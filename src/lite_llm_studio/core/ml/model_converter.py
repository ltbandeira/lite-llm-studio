"""
Module core.ml.model_converter
-------------------------------

This module handles conversion of fine-tuned LoRA models to GGUF format.
Merges LoRA adapters with base model and converts to standalone GGUF.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger("app.ml.converter")


def merge_lora_to_base_model(
    adapter_path: str,
    base_model_path: str,
    output_path: str,
) -> str:
    """Merge LoRA adapters into base model to create standalone model.

    Args:
        adapter_path: Path to LoRA adapter directory
        base_model_path: Path to base model
        output_path: Where to save merged model

    Returns:
        Path to merged model directory

    Raises:
        RuntimeError: If merge fails
    """
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info("Merging LoRA adapters into base model...")
        logger.info(f"  Base model: {base_model_path}")
        logger.info(f"  Adapter: {adapter_path}")

        # Load base model
        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        # Load adapter
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_path)

        # Merge
        logger.info("Merging adapter into base model...")
        model = model.merge_and_unload()

        # Save merged model
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving merged model to: {output_dir}")
        model.save_pretrained(str(output_dir), safe_serialization=True)

        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(str(output_dir))

        logger.info("✓ Model merge completed")

        # Clean up GPU memory
        del model
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return str(output_dir)

    except Exception as e:
        logger.error(f"Model merge failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to merge LoRA adapters: {e}")


def convert_hf_to_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "f16",
) -> str:
    """Convert HuggingFace model to GGUF format using llama.cpp.

    Args:
        model_path: Path to HuggingFace model directory
        output_path: Where to save GGUF file
        quantization: Quantization type (f32, f16, bf16, q8_0, etc.)

    Returns:
        Path to GGUF file

    Raises:
        RuntimeError: If conversion fails
    """
    try:
        # Find llama.cpp conversion script
        llama_cpp_root = Path(__file__).parent.parent.parent.parent.parent / "llama.cpp"
        convert_script = llama_cpp_root / "convert_hf_to_gguf.py"

        if not convert_script.exists():
            raise RuntimeError(
                f"llama.cpp not found at: {llama_cpp_root}\n"
                "Please clone llama.cpp in project root:\n"
                "  git clone https://github.com/ggerganov/llama.cpp.git"
            )

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Output filename
        quant_type = quantization.lower()
        output_file = output_dir / f"model-{quant_type}.gguf"

        logger.info(f"Converting HuggingFace model to GGUF...")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Output: {output_file}")
        logger.info(f"  Quantization: {quant_type}")

        # Get Python executable
        import sys, os

        python_exe = sys.executable

        # Build command
        cmd = [
            python_exe,
            str(convert_script),
            str(model_path),
            "--outfile",
            str(output_file),
            "--outtype",
            quant_type,
        ]

        # Run conversion
        logger.info("Running llama.cpp HuggingFace to GGUF conversion...")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(llama_cpp_root),
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(f"GGUF conversion failed:\n{result.stderr}\n{result.stdout}")

        if result.stdout:
            logger.info(f"Conversion output:\n{result.stdout}")

        if not output_file.exists():
            raise RuntimeError(f"GGUF file was not created at: {output_file}")

        logger.info(f"✓ GGUF conversion completed: {output_file}")
        return str(output_file)

    except Exception as e:
        logger.error(f"GGUF conversion failed: {e}", exc_info=True)
        raise RuntimeError(f"GGUF conversion failed: {e}")


def convert_finetuned_model_to_gguf(
    adapter_path: str,
    base_model_path: str,
    output_name: Optional[str] = None,
    quantization: str = "f16",
    models_dir: Optional[str] = None,
) -> dict[str, str]:
    """Convert fine-tuned LoRA model to standalone GGUF format.

    Process:
    1. Merge LoRA adapters with base model
    2. Convert merged model to GGUF
    3. Clean up intermediate files

    Args:
        adapter_path: Path to the LoRA adapter directory (fine-tuned model)
        base_model_path: Path to the original base model
        output_name: Optional name for output (defaults to adapter directory name)
        quantization: GGUF quantization type (f32, f16, bf16, q8_0, auto)
        models_dir: Optional directory where to save GGUF models

    Returns:
        Dictionary with paths:
            - merged_model: Path to merged HuggingFace model (intermediate)
            - gguf_model: Path to final GGUF file

    Raises:
        RuntimeError: If conversion fails
    """
    adapter_dir = Path(adapter_path)
    model_name = output_name or adapter_dir.name

    # Determine output directory
    if models_dir:
        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        output_base = models_path / model_name
    else:
        output_base = adapter_dir.parent / f"{model_name}_gguf"

    output_base.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting fine-tuned model to standalone GGUF")
    logger.info(f"  Adapter: {adapter_path}")
    logger.info(f"  Base model: {base_model_path}")
    logger.info(f"  Quantization: {quantization}")
    logger.info(f"  Output: {output_base}")

    try:
        # Step 1: Merge LoRA with base model
        logger.info("=" * 60)
        logger.info("STEP 1/2: Merging LoRA adapters with base model")
        logger.info("=" * 60)

        merged_dir = output_base / "merged_model"
        merged_path = merge_lora_to_base_model(
            adapter_path=adapter_path,
            base_model_path=base_model_path,
            output_path=str(merged_dir),
        )

        # Step 2: Convert merged model to GGUF
        logger.info("=" * 60)
        logger.info("STEP 2/2: Converting merged model to GGUF")
        logger.info("=" * 60)

        gguf_path = convert_hf_to_gguf(
            model_path=merged_path,
            output_path=str(output_base),
            quantization=quantization,
        )

        logger.info("=" * 60)
        logger.info("✅ CONVERSION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"📁 Merged model: {merged_path}")
        logger.info(f"📦 GGUF model: {gguf_path}")

        # Create README
        readme_path = output_base / "README.md"
        gguf_filename = Path(gguf_path).name

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# {model_name}\n\n")
            f.write(f"Fine-tuned model converted to standalone GGUF format.\n\n")
            f.write(f"## 📁 Files\n\n")
            f.write(f"- `{gguf_filename}`: Standalone GGUF model ({quantization})\n")
            f.write(f"- `merged_model/`: Intermediate merged HuggingFace model\n")
            f.write(f"- Compatible with llama.cpp, Ollama, LM Studio, etc.\n\n")
            f.write(f"## 📊 Model Information\n\n")
            f.write(f"- **Base Model**: `{base_model_path}`\n")
            f.write(f"- **LoRA Adapter**: `{adapter_path}`\n")
            f.write(f"- **Quantization**: {quantization}\n")
            f.write(f"- **Conversion**: LoRA merge + HuggingFace to GGUF\n\n")
            f.write(f"## 🚀 Usage\n\n")
            f.write(f"This is a **standalone model** - no need for base model or adapter!\n\n")
            f.write(f"### llama.cpp\n")
            f.write(f"```bash\n")
            f.write(f'./llama-cli -m {gguf_filename} -p "Your prompt here"\n')
            f.write(f"```\n\n")
            f.write(f"### Ollama\n")
            f.write(f"```bash\n")
            f.write(f"# Create Modelfile\n")
            f.write(f"echo 'FROM {gguf_filename}' > Modelfile\n\n")
            f.write(f"# Import model\n")
            f.write(f"ollama create {model_name} -f Modelfile\n\n")
            f.write(f"# Run\n")
            f.write(f"ollama run {model_name}\n")
            f.write(f"```\n\n")
            f.write(f"### Python (llama-cpp-python)\n")
            f.write(f"```python\n")
            f.write(f"from llama_cpp import Llama\n\n")
            f.write(f'model = Llama(model_path="{gguf_filename}")\n')
            f.write(f'response = model("Your prompt", max_tokens=256)\n')
            f.write(f"print(response['choices'][0]['text'])\n")
            f.write(f"```\n")

        logger.info(f"📝 Created README at: {readme_path}")

        return {
            "merged_model": merged_path,
            "gguf_model": gguf_path,
        }

    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        raise RuntimeError(
            f"Failed to convert model to GGUF.\n\n"
            f"Error: {str(e)}\n\n"
            f"Common issues:\n"
            f"1. Insufficient disk space (needs ~2x model size)\n"
            f"2. Insufficient RAM/VRAM for model merge\n"
            f"3. llama.cpp not found in project root\n"
            f"4. Adapter or base model corrupted\n\n"
            f"Supported quantization: f32, f16, bf16, q8_0, auto"
        )
