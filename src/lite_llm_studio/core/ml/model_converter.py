"""
Module core.ml.model_converter
-------------------------------

This module handles conversion of fine-tuned models to different formats,
particularly GGUF format for use with llama.cpp.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger("app.ml.converter")


def check_llama_cpp_available() -> tuple[bool, str]:
    """Check if llama.cpp conversion tools are available.
    
    Returns:
        Tuple of (available: bool, message: str)
    """
    # Check for convert.py from llama.cpp
    llama_cpp_paths = [
        Path("llama.cpp/convert.py"),
        Path("../llama.cpp/convert.py"),
        Path.home() / "llama.cpp" / "convert.py",
    ]
    
    for path in llama_cpp_paths:
        if path.exists():
            return True, f"Found llama.cpp at: {path.parent}"
    
    return False, "llama.cpp not found. Please clone and setup llama.cpp repository."


def merge_lora_adapters(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
) -> str:
    """Merge LoRA adapters back into the base model.
    
    This creates a full model with the LoRA weights merged, which can then
    be converted to GGUF format.
    
    Args:
        base_model_path: Path to the base model
        adapter_path: Path to the LoRA adapter directory
        output_path: Where to save the merged model
        
    Returns:
        Path to the merged model
        
    Raises:
        RuntimeError: If merge fails
    """
    try:
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch
        
        logger.info(f"Loading base model from {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        logger.info(f"Loading LoRA adapters from {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        logger.info("Merging LoRA adapters into base model...")
        model = model.merge_and_unload()
        
        logger.info(f"Saving merged model to {output_path}")
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(output_path)
        
        # Also save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(output_path)
        
        logger.info("Model merge completed successfully")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to merge LoRA adapters: {e}", exc_info=True)
        raise RuntimeError(f"Model merge failed: {e}")


def convert_to_gguf(
    model_path: str,
    output_path: Optional[str] = None,
    quantization: str = "Q4_K_M",
) -> str:
    """Convert a PyTorch model to GGUF format.
    
    Args:
        model_path: Path to the PyTorch model directory
        output_path: Optional output path (defaults to model_path/gguf/)
        quantization: Quantization type (Q4_K_M, Q5_K_M, Q8_0, etc.)
        
    Returns:
        Path to the generated GGUF file
        
    Raises:
        RuntimeError: If conversion fails or llama.cpp is not available
    """
    available, message = check_llama_cpp_available()
    if not available:
        raise RuntimeError(
            f"{message}\n\n"
            "Para converter modelos para GGUF:\n"
            "1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp\n"
            "2. Instale dependências: pip install -r llama.cpp/requirements.txt\n"
            "3. Tente novamente"
        )
    
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise ValueError(f"Model directory not found: {model_path}")
    
    # Determine output path
    if output_path is None:
        output_dir = model_dir / "gguf"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"model-{quantization.lower()}.gguf"
    else:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Find llama.cpp convert script
    convert_script = None
    for path in [Path("llama.cpp/convert.py"), Path("../llama.cpp/convert.py")]:
        if path.exists():
            convert_script = path
            break
    
    if not convert_script:
        raise RuntimeError("Could not locate llama.cpp/convert.py")
    
    try:
        # Step 1: Convert to fp16 GGUF
        logger.info("Step 1/2: Converting to FP16 GGUF...")
        fp16_file = output_file.parent / "model-fp16.gguf"
        
        cmd = [
            "python",
            str(convert_script),
            str(model_dir),
            "--outfile", str(fp16_file),
            "--outtype", "f16",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Conversion failed: {result.stderr}")
        
        logger.info(f"FP16 model saved to: {fp16_file}")
        
        # Step 2: Quantize if requested
        if quantization.upper() != "F16":
            logger.info(f"Step 2/2: Quantizing to {quantization}...")
            
            # Find quantize binary
            quantize_bin = None
            for path in [
                Path("llama.cpp/build/bin/quantize.exe"),
                Path("llama.cpp/build/bin/quantize"),
                Path("../llama.cpp/build/bin/quantize.exe"),
            ]:
                if path.exists():
                    quantize_bin = path
                    break
            
            if not quantize_bin:
                logger.warning("Quantize binary not found. Skipping quantization.")
                logger.info("To enable quantization, build llama.cpp with: make")
                output_file = fp16_file
            else:
                cmd = [
                    str(quantize_bin),
                    str(fp16_file),
                    str(output_file),
                    quantization.upper(),
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Quantization failed: {result.stderr}")
                
                logger.info(f"Quantized model saved to: {output_file}")
                
                # Clean up fp16 file
                if fp16_file.exists() and fp16_file != output_file:
                    fp16_file.unlink()
        else:
            output_file = fp16_file
        
        logger.info("GGUF conversion completed successfully")
        return str(output_file)
        
    except Exception as e:
        logger.error(f"GGUF conversion failed: {e}", exc_info=True)
        raise RuntimeError(f"GGUF conversion failed: {e}")


def convert_finetuned_model_to_gguf(
    adapter_path: str,
    base_model_path: str,
    output_name: Optional[str] = None,
    quantization: str = "Q4_K_M",
) -> dict[str, str]:
    """Complete workflow: merge LoRA adapters and convert to GGUF.
    
    Args:
        adapter_path: Path to the LoRA adapter directory (fine-tuned model)
        base_model_path: Path to the original base model
        output_name: Optional name for output (defaults to adapter directory name)
        quantization: GGUF quantization type
        
    Returns:
        Dictionary with paths:
            - merged_model: Path to merged PyTorch model
            - gguf_model: Path to GGUF file
    """
    adapter_dir = Path(adapter_path)
    
    if output_name is None:
        output_name = adapter_dir.name + "-merged"
    
    # Step 1: Merge LoRA adapters
    merged_dir = adapter_dir.parent / output_name
    merged_path = merge_lora_adapters(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        output_path=str(merged_dir),
    )
    
    # Step 2: Convert to GGUF
    gguf_path = convert_to_gguf(
        model_path=merged_path,
        quantization=quantization,
    )
    
    return {
        "merged_model": merged_path,
        "gguf_model": gguf_path,
    }
