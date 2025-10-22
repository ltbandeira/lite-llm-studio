"""
Module core.ml.training
----------------------

This module handles fine-tuning of causal language models using LoRA adapters
with Unsloth for optimization on modest hardware (CPU and consumer GPUs).
"""

import os
import platform

os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["HF_DATASETS_CACHE"] = r"D:\hf-cache"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("RAYON_NUM_THREADS", str(os.cpu_count() or 4))
os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"

# Disable PyTorch compilation on Windows to avoid Triton compatibility issues
if platform.system() == "Windows":
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
import unsloth

logger = logging.getLogger("app.ml.training")


@dataclass
class TrainingConfig:
    """Configuration for fine‑tuning a causal language model.

    Attributes:
        dataset_dir: Directory containing ``train.jsonl`` and
            ``validation.jsonl`` files. A ``test.jsonl`` file is optional
            but may be used for evaluation after training.
        base_model_path: Path to the directory or file containing the
            pre‑trained base model. This should be compatible with
            ``transformers.AutoModelForCausalLM.from_pretrained``.
        output_dir: Directory where the fine‑tuned model and tokenizer
            will be saved. The directory will be created if it does not
            exist.
        epochs: Number of training epochs.
        batch_size: Batch size per device. Gradient accumulation
            determines the effective batch size. Default is 4.
        learning_rate: Base learning rate for the optimiser.
        max_seq_length: Maximum sequence length (context window) for
            training. Typical values are 512–2048.
        lora_r: Rank of LoRA adapters. Smaller values reduce memory
            usage but may limit expressiveness. Typical range: 4–16.
        lora_alpha: Alpha parameter for LoRA scaling. Usually set to
            ``lora_r * 2``.
        lora_dropout: Dropout probability for LoRA layers.
        gradient_accumulation_steps: Number of gradient accumulation
            steps. Default is 1. Increasing this value reduces memory
            usage at the cost of longer training.
        warmup_steps: Number of warmup steps for learning rate scheduler.
        weight_decay: Weight decay for optimizer.
        use_gradient_checkpointing: Enable gradient checkpointing to save memory.
        use_4bit: Use 4-bit quantization for model loading.
    """

    dataset_dir: str
    base_model_path: str
    output_dir: str
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 1024
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 10
    weight_decay: float = 0.01
    use_gradient_checkpointing: bool = True
    use_4bit: bool = True


def _detect_hardware() -> dict[str, any]:
    """Detect available hardware for training.

    Returns:
        Dictionary with hardware information:
            - has_cuda: bool
            - has_mps: bool (Apple Silicon)
            - device: str (cuda, mps, or cpu)
            - gpu_memory_gb: float (if applicable)
    """
    import torch

    hardware = {
        "has_cuda": torch.cuda.is_available(),
        "has_mps": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
        "device": "cpu",
        "gpu_memory_gb": 0.0,
    }

    if hardware["has_cuda"]:
        hardware["device"] = "cuda"
        try:
            # Get total GPU memory in GB
            hardware["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            pass
    elif hardware["has_mps"]:
        hardware["device"] = "mps"

    return hardware


def _optimize_config_for_hardware(config: TrainingConfig, hardware: dict[str, any]) -> TrainingConfig:
    """Optimize training configuration based on available hardware.

    Args:
        config: Initial training configuration
        hardware: Hardware information from _detect_hardware

    Returns:
        Optimized training configuration
    """
    # For CPU-only training, reduce batch size and enable more aggressive memory optimizations
    if hardware["device"] == "cpu":
        logger.info("CPU-only mode detected. Applying CPU-optimized settings.")
        config.batch_size = min(config.batch_size, 2)
        config.gradient_accumulation_steps = max(config.gradient_accumulation_steps, 4)
        config.use_gradient_checkpointing = True
        config.use_4bit = True

    # For low-memory GPUs (< 8GB), adjust settings
    elif hardware["device"] == "cuda" and hardware["gpu_memory_gb"] < 8:
        logger.info(f"Low-memory GPU detected ({hardware['gpu_memory_gb']:.1f}GB). Applying memory-efficient settings.")
        config.batch_size = min(config.batch_size, 2)
        config.gradient_accumulation_steps = max(config.gradient_accumulation_steps, 2)
        config.use_gradient_checkpointing = True
        config.use_4bit = True

    # For mid-range GPUs (8-16GB), moderate settings
    elif hardware["device"] == "cuda" and hardware["gpu_memory_gb"] < 16:
        logger.info(f"Mid-range GPU detected ({hardware['gpu_memory_gb']:.1f}GB). Using balanced settings.")
        config.use_4bit = True

    return config


def load_jsonl_dataset(file_path: str) -> list[dict[str, str]]:
    """Load a JSONL file into a list of dictionaries.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        A list of dictionaries, each representing a sample.
    """
    samples: list[dict[str, str]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                samples.append(data)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON line: {line[:80]}")
                continue
    return samples


def train_lora_model(config: TrainingConfig, progress_callback: Optional[Callable[[str, float], None]] = None) -> str:
    """Fine‑tune a causal LM using LoRA adapters on the provided dataset.

    This function performs parameter‑efficient fine‑tuning using the
    Unsloth and PEFT libraries. It will load the train and validation
    splits from the dataset directory, prepare the model and tokenizer
    from ``base_model_path``, configure LoRA layers, and then train
    using the SFTTrainer from the ``trl`` library. After training, the
    fine‑tuned model and tokenizer are saved to ``output_dir``.

    The function automatically detects available hardware (CUDA GPU, Apple
    Silicon MPS, or CPU) and optimizes training parameters accordingly for
    best performance on modest hardware.

    Args:
        config: Training configuration with dataset and hyperparameters.
        progress_callback: Optional callback function for progress updates.
            Called with (message: str, progress: float) where progress is 0-100.

    Returns:
        Path to the directory containing the fine‑tuned model.

    Raises:
        RuntimeError: If required dependencies are missing.
        ValueError: If dataset or model paths are invalid.
        Exception: If training fails.
    """

    # Import modules after verifying dependencies
    import torch
    from unsloth import FastLanguageModel
    from datasets import load_dataset, disable_caching
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback

    # Disable torch compilation on Windows to avoid Triton compatibility issues
    if platform.system() == "Windows":
        logger.info("Configuring PyTorch for Windows compatibility")
        # Disable dynamic compilation that requires Triton
        torch._dynamo.config.suppress_errors = True
        # Ensure CUDNN is enabled for better performance
        torch.backends.cudnn.enabled = True
        logger.info("PyTorch JIT compilation disabled to avoid Triton errors")

    # Detect hardware and optimize configuration
    hardware = _detect_hardware()
    logger.info(f"Hardware detected: {hardware['device']}")
    if hardware["gpu_memory_gb"] > 0:
        logger.info(f"GPU memory: {hardware['gpu_memory_gb']:.1f}GB")

    config = _optimize_config_for_hardware(config, hardware)

    # Report progress
    if progress_callback:
        progress_callback("Initializing training environment...", 5.0)

    # Resolve and validate paths
    dataset_dir = Path(config.dataset_dir)
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    base_model_path = Path(config.base_model_path)
    if not base_model_path.exists():
        raise ValueError(f"Base model path not found: {base_model_path}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate dataset files
    train_file = dataset_dir / "train.jsonl"
    if not train_file.exists():
        raise ValueError(f"Training file not found: {train_file}")

    val_file = dataset_dir / "validation.jsonl"
    data_files = {"train": str(train_file)}
    if val_file.exists():
        data_files["validation"] = str(val_file)
        logger.info("Validation set found and will be used for evaluation")
    else:
        logger.warning("No validation set found. Training without evaluation.")

    # Load dataset
    logger.info(f"Loading dataset from {dataset_dir}")
    if progress_callback:
        progress_callback("Loading dataset...", 10.0)

    try:
        ds = load_dataset("json", data_files=data_files)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

    # Ensure dataset has correct column structure
    for split_name in ds.keys():
        sample = ds[split_name][0]
        if "text" not in sample:
            # Try to find a suitable text field
            text_field = None
            for key in sample.keys():
                if key.lower() in {"text", "content", "prompt", "input"}:
                    text_field = key
                    break
            if text_field is None:
                raise RuntimeError(f"Dataset split '{split_name}' does not contain a 'text' field. " f"Available fields: {list(sample.keys())}")
            logger.info(f"Renaming column '{text_field}' to 'text' in split '{split_name}'")
            ds[split_name] = ds[split_name].rename_column(text_field, "text")

    train_samples = len(ds["train"])
    logger.info(f"Training samples: {train_samples}")
    if "validation" in ds:
        logger.info(f"Validation samples: {len(ds['validation'])}")

    # Load base model and tokenizer with Unsloth optimization
    logger.info(f"Loading base model from {base_model_path}")
    if progress_callback:
        progress_callback("Loading base model...", 20.0)

    try:
        # Determine dtype based on hardware
        # Unsloth only accepts None, torch.float16, or torch.bfloat16
        # For CPU or when we want auto-selection, use None
        if config.use_4bit or hardware["device"] == "cpu":
            dtype = None  # Let Unsloth choose the best dtype
        else:
            # For non-quantized GPU training, use float16
            import torch

            dtype = torch.float16

        logger.info(f"Loading model with dtype={dtype}, 4-bit={config.use_4bit}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(base_model_path),
            max_seq_length=config.max_seq_length,
            dtype=dtype,
            load_in_4bit=config.use_4bit,
        )

        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")

        # Log tokenizer special tokens for debugging
        logger.info(f"Tokenizer special tokens:")
        logger.info(f"  EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
        logger.info(f"  BOS token: {tokenizer.bos_token} (id: {getattr(tokenizer, 'bos_token_id', None)})")
        logger.info(f"  PAD token: {tokenizer.pad_token} (id: {getattr(tokenizer, 'pad_token_id', None)})")

        try:
            tokenizer.model_max_length = int(config.max_seq_length)
            if hasattr(tokenizer, "init_kwargs"):
                tokenizer.init_kwargs["model_max_length"] = int(config.max_seq_length)
        except Exception:
            logger.warning("Could not set tokenizer.model_max_length; continuing.")

    except Exception as e:
        logger.error(f"Model loading error: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load model: {e}")

    # Apply LoRA configuration
    logger.info("Applying LoRA configuration")
    logger.info(f"  LoRA rank (r): {config.lora_r}")
    logger.info(f"  LoRA alpha: {config.lora_alpha}")
    logger.info(f"  LoRA dropout: {config.lora_dropout}")

    if progress_callback:
        progress_callback("Configuring LoRA adapters...", 30.0)

    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            random_state=42,
        )
    except Exception as e:
        logger.warning(f"Failed to apply LoRA to all target modules: {e}")
        logger.info("Retrying with minimal target modules (q_proj, v_proj)")
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            random_state=42,
        )

    # Calculate total training steps for progress tracking
    steps_per_epoch = train_samples // (config.batch_size * config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.epochs
    logger.info(f"Total training steps: {total_steps} ({steps_per_epoch} per epoch)")

    # On Windows, completely disable multiprocessing to prevent subprocess crashes
    if platform.system() == "Windows":
        logger.info("Windows detected: configuring for single-process operation")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"
        # Force datasets library to use single process
        import datasets
        datasets.disable_caching()

    # Define training arguments with optimizations
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        eval_strategy="steps" if "validation" in ds else "no",
        eval_steps=max(50, steps_per_epoch // 2) if "validation" in ds else None,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=max(1, steps_per_epoch // 10),
        logging_first_step=True,
        fp16=(hardware["device"] == "cuda" and not config.use_4bit),
        bf16=False,
        optim="adamw_bnb_8bit" if config.use_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        seed=42,
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
        remove_unused_columns=False,
        report_to=None,
        dataset_text_field="text",
        max_length=config.max_seq_length,
        dataset_num_proc=1,  # 1 means single process, not multiprocessing
        packing=False,
        eval_packing=False,
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token,
    )

    # Debug: verify the config values
    logger.info(f"SFTConfig eos_token: {sft_config.eos_token}")
    logger.info(f"SFTConfig pad_token: {sft_config.pad_token}")
    logger.info(f"Tokenizer eos_token: {tokenizer.eos_token}")
    logger.info(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")

    # Custom callback for progress reporting
    class ProgressCallback(TrainerCallback):
        def __init__(self, total_steps, callback_fn):
            self.total_steps = total_steps
            self.callback_fn = callback_fn

        def on_step_end(self, args, state, control, **kwargs):
            if self.callback_fn and state.global_step > 0:
                # Progress from 30% to 90% during training
                progress = 30.0 + (state.global_step / self.total_steps) * 60.0
                self.callback_fn(f"Training: Step {state.global_step}/{self.total_steps}", min(progress, 90.0))
            return control

    # Initialize trainer
    logger.info("Starting fine‑tuning")
    logger.info(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")

    if progress_callback:
        progress_callback("Starting training...", 35.0)

    callbacks = []
    if progress_callback:
        callbacks.append(ProgressCallback(total_steps, progress_callback))

    try:
        # On Windows, monkey-patch the dataset.map to remove num_proc parameter
        # This prevents the subprocess crash with Unsloth's compiled modules
        if platform.system() == "Windows":
            logger.info("Applying Windows-specific dataset.map patch to prevent multiprocessing")
            from datasets import Dataset
            original_map = Dataset.map
            
            def patched_map(self, *args, **kwargs):
                """Remove num_proc from kwargs to prevent multiprocessing on Windows"""
                if 'num_proc' in kwargs:
                    logger.debug(f"Removing num_proc={kwargs['num_proc']} from dataset.map call")
                    del kwargs['num_proc']
                return original_map(self, *args, **kwargs)
            
            Dataset.map = patched_map

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=ds["train"],
            eval_dataset=ds.get("validation"),
            processing_class=tokenizer,
            callbacks=callbacks,
        )

        # Train the model
        trainer.train()

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        raise RuntimeError(f"Training failed: {e}")

    # Save model and tokenizer
    logger.info("Training complete, saving model and tokenizer")
    if progress_callback:
        progress_callback("Saving fine-tuned model...", 95.0)

    try:
        # Save the final model
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # Also save training configuration for reproducibility
        config_dict = {
            "dataset_dir": str(dataset_dir),
            "base_model_path": str(base_model_path),
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "max_seq_length": config.max_seq_length,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "hardware_used": hardware["device"],
        }
        config_file = output_dir / "training_config.json"
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)

    except Exception as e:
        raise RuntimeError(f"Failed to save model: {e}")

    logger.info(f"Fine-tuned model saved successfully to {output_dir}")
    if progress_callback:
        progress_callback("Training completed!", 100.0)

    return str(output_dir)
