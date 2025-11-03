"""
Module core.ml.registry
-----------------------

"""

import json
from pathlib import Path
from typing import Literal

from lite_llm_studio.core.configuration.model_schema import ArtifactInfo, DiscoveryConfig, ModelCard


class ModelRegistry:
    def __init__(self, discovery: DiscoveryConfig):
        self.discovery = discovery
        self._cards: dict[str, ModelCard] = {}

    def scan(self) -> list[ModelCard]:
        root = Path(self.discovery.models_root).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Models root not found or not a directory: {root}")

        found: list[ModelCard] = []
        found_dirs = set()  # Track directories already processed

        # metadata.json -> preferido
        for meta in root.rglob("metadata.json"):
            try:
                card = self._load_card_from_metadata(meta)
                self._cards[card.slug] = card
                found.append(card)
                found_dirs.add(str(meta.parent))
            except Exception as e:
                print(f"[registry] Skipping metadata {meta}: {e}")

        # Modelos PyTorch/Transformers (config.json + .safetensors)
        for config in root.rglob("config.json"):
            model_dir = config.parent
            
            # Skip if already processed
            if str(model_dir) in found_dirs:
                continue
                
            # Verifica se tem arquivos .safetensors
            safetensors_files = list(model_dir.glob("*.safetensors"))
            
            if safetensors_files:
                try:
                    card = self._mk_minimal_card_from_pytorch(model_dir, safetensors_files)
                    self._cards[card.slug] = card
                    found.append(card)
                    found_dirs.add(str(model_dir))
                except Exception as e:
                    print(f"[registry] Skipping PyTorch model {model_dir}: {e}")

        # arquivos .gguf sem metadata
        for gguf in root.rglob("*.gguf"):
            # Skip if directory already processed
            if str(gguf.parent) in found_dirs:
                continue
            # se já carregado via metadata, ignora
            if any(c.model_file() == gguf for c in found):
                continue
            # cria mínimo
            card = self._mk_minimal_card_from_gguf(gguf)
            self._cards[card.slug] = card
            found.append(card)
            found_dirs.add(str(gguf.parent))

        # ordena por nome
        found.sort(key=lambda c: (c.name.lower(), c.version))
        return found

    def all(self) -> list[ModelCard]:
        return list(self._cards.values())

    def _detect_model_family(self, filename: str) -> Literal["LLaMA", "Qwen", "Mistral", "Phi", "Unknown"]:
        """
        Auto-detect model family from filename patterns.

        Args:
            filename: The model filename (without extension)

        Returns:
            Literal["LLaMA", "Qwen", "Mistral", "Phi", "Unknown"]: Detected model family name
        """
        filename_lower = filename.lower()

        # Common model family patterns
        if any(pattern in filename_lower for pattern in ["llama", "llama2", "llama3", "code-llama"]):
            return "LLaMA"
        elif any(pattern in filename_lower for pattern in ["qwen", "qwen2", "qwen-chat"]):
            return "Qwen"
        elif any(pattern in filename_lower for pattern in ["mistral", "mixtral", "codestral"]):
            return "Mistral"
        elif any(pattern in filename_lower for pattern in ["phi", "phi-2", "phi-3"]):
            return "Phi"
        else:
            return "Unknown"

    def get(self, slug: str) -> ModelCard | None:
        return self._cards.get(slug)

    def _load_card_from_metadata(self, metadata_path: Path) -> ModelCard:
        with metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # ajusta root_dir e caminhos relativos
        root_dir = metadata_path.parent
        # valida e instancia
        card = ModelCard(**data)
        card.root_dir = str(root_dir)
        # valida se arquivo existe
        if not card.model_file().exists():
            raise FileNotFoundError(f"Model file not found: {card.model_file()}")
        return card

    def _mk_minimal_card_from_gguf(self, gguf_path: Path) -> ModelCard:
        name = gguf_path.stem
        slug = name.lower().replace(" ", "-")
        artifacts = ArtifactInfo(model_path=str(gguf_path.name))

        # Auto-detect model family from filename
        family = self._detect_model_family(name)

        return ModelCard(
            name=name,
            slug=slug,
            artifacts=artifacts,
            runtime="llamacpp",
            root_dir=str(gguf_path.parent),
            family=family,
            task="chat",
            quantization={"format": "gguf"},
            tags=["discovered"],
            system_prompt="You are a helpful AI assistant. Please provide clear, accurate, and concise responses to user questions.",
        )

    def _mk_minimal_card_from_pytorch(self, model_dir: Path, safetensors_files: list[Path]) -> ModelCard:
        """Create a minimal ModelCard from a PyTorch/HuggingFace model directory."""
        name = model_dir.name
        slug = name.lower().replace(" ", "-")
        
        # Try to read config.json for more info
        model_type = None
        try:
            config_path = model_dir / "config.json"
            if config_path.exists():
                with config_path.open("r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    model_type = config_data.get("model_type", "")
                    architectures = config_data.get("architectures", [])
                    
                    # Add architecture info if available
                    if architectures:
                        arch_name = architectures[0] if isinstance(architectures, list) else architectures
                        name = f"{name} ({arch_name})"
                    elif model_type:
                        name = f"{name} ({model_type})"
        except Exception as e:
            print(f"[registry] Could not read config.json for {model_dir}: {e}")
        
        # Use directory path for PyTorch models (loaded by directory, not individual file)
        artifacts = ArtifactInfo(model_path=".")

        # Auto-detect model family from directory name
        family = self._detect_model_family(model_dir.name)

        return ModelCard(
            name=name,
            slug=slug,
            artifacts=artifacts,
            runtime="transformers",
            root_dir=str(model_dir),
            family=family,
            task="chat",
            quantization={"format": "safetensors"},
            tags=["pytorch", "transformers", "finetunable"],
            system_prompt="You are a helpful AI assistant. Please provide clear, accurate, and concise responses to user questions.",
        )
