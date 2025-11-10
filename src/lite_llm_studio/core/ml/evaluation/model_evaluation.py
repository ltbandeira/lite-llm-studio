"""
Module core.ml.model_evaluation
-------------------------------

This module provides model evaluation utilities, including perplexity calculation
for language models in GGUF format.
"""

import json
import logging
import math
from pathlib import Path

import torch
from tqdm import tqdm

logger = logging.getLogger("app.ml.evaluation")


def evaluate_model_perplexity(
    model_path: str,
    validation_file: str,
    max_samples: int = 100,
    max_length: int = 512,
) -> float:
    """Calculate perplexity of a GGUF model on validation data.

    Perplexity measures how well a language model predicts a sample of text.
    Lower perplexity indicates better performance.

    Args:
        model_path: Path to the GGUF model file
        validation_file: Path to validation JSONL file
        max_samples: Maximum number of samples to evaluate (for speed)
        max_length: Maximum sequence length for evaluation

    Returns:
        float: Perplexity score (lower is better)

    Raises:
        FileNotFoundError: If model or validation file not found
        RuntimeError: If evaluation fails
    """
    logger.info(f"Evaluating perplexity: model={model_path}, samples={max_samples}")

    # Validate inputs
    model_path_obj = Path(model_path)
    validation_path = Path(validation_file)

    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not validation_path.exists():
        raise FileNotFoundError(f"Validation file not found: {validation_file}")

    try:
        # Import llama-cpp-python
        from llama_cpp import Llama

        # Load model
        logger.info("Loading model...")
        llm = Llama(
            model_path=str(model_path),
            n_ctx=max_length,
            n_batch=max_length,  # Match context size to avoid batch issues
            n_threads=8,
            verbose=False,
            logits_all=True,  # Need logits for perplexity calculation
            n_gpu_layers=-1,  # Use GPU if available
        )

        # Load validation data
        logger.info(f"Loading validation data from {validation_file}")
        validation_samples = []
        with open(validation_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text.strip():
                        validation_samples.append(text)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line {i+1}")
                    continue

        if not validation_samples:
            raise ValueError("No valid samples found in validation file")

        logger.info(f"Evaluating {len(validation_samples)} samples...")

        # Calculate perplexity
        total_nll = 0.0  # Negative log-likelihood
        total_tokens = 0

        for text in tqdm(validation_samples, desc="Calculating perplexity"):
            try:
                # Tokenize
                tokens = llm.tokenize(text.encode("utf-8"), add_bos=True)

                # Limit length
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]

                if len(tokens) < 2:
                    continue

                # Reset model state
                llm.reset()

                # Evaluate sequence in one pass
                # Feed all tokens at once and get logits for each position except the last
                llm.eval(tokens[:-1])

                # Get logits for all positions
                logits_seq = list(llm.eval_logits)
                if not logits_seq or len(logits_seq) != len(tokens) - 1:
                    # Fallback
                    logits_seq = logits_seq[: max(0, len(tokens) - 1)]

                # Calculate log probabilities for actual next tokens
                with torch.no_grad():
                    for i, logits in enumerate(logits_seq):
                        next_token = tokens[i + 1]
                        log_probs = torch.log_softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)
                        total_nll -= float(log_probs[next_token])
                        total_tokens += 1

            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
                # Reset and continue with next sample
                llm.reset()
                continue

        if total_tokens == 0:
            raise RuntimeError("No tokens were successfully evaluated")

        # Calculate perplexity: exp(average negative log-likelihood)
        avg_nll = total_nll / total_tokens
        perplexity = math.exp(avg_nll)

        logger.info(f"Perplexity: {perplexity:.2f} (evaluated {total_tokens} tokens)")

        return perplexity

    except ImportError as e:
        raise RuntimeError("llama-cpp-python is required for model evaluation. " "Please install it: pip install llama-cpp-python") from e
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to evaluate model: {e}") from e


def compare_models(
    base_model_path: str,
    finetuned_model_path: str,
    validation_file: str,
    max_samples: int = 100,
) -> dict[str, float]:
    """Compare two models using perplexity.

    Args:
        base_model_path: Path to base GGUF model
        finetuned_model_path: Path to fine-tuned GGUF model
        validation_file: Path to validation JSONL file
        max_samples: Maximum samples to evaluate

    Returns:
        Dictionary with comparison results:
        - base_perplexity: Base model perplexity
        - finetuned_perplexity: Fine-tuned model perplexity
        - improvement: Absolute improvement
        - improvement_pct: Percentage improvement
    """
    logger.info("Comparing models...")

    base_ppl = evaluate_model_perplexity(base_model_path, validation_file, max_samples)
    finetuned_ppl = evaluate_model_perplexity(finetuned_model_path, validation_file, max_samples)

    improvement = base_ppl - finetuned_ppl
    improvement_pct = (improvement / base_ppl) * 100 if base_ppl > 0 else 0

    results = {
        "base_perplexity": base_ppl,
        "finetuned_perplexity": finetuned_ppl,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
    }

    logger.info(f"Comparison results: {results}")

    return results
