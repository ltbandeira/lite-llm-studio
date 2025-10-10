"""
Module core.configuration.model_schema
--------------------------------------

"""

from pathlib import Path
from typing import Literal

from pydantic import Field, StrictStr, PositiveInt, field_validator

from .base_config import BaseConfigModel


class RuntimeSpec(BaseConfigModel):
    """
    Runtime specifications for model loading and inference.
    """

    n_ctx: PositiveInt = Field(4096, description="Context size")
    n_threads: PositiveInt | None = Field(None, description="CPU threads (None => auto)")
    n_gpu_layers: int = Field(-1, description="Number of layers on GPU (-1 = all)")

    def to_dict(self):
        return super().to_dict()

    def to_json(self):
        return super().to_json()


class GenParams(BaseConfigModel):
    """
    Generation parameters for model inference.
    """

    temperature: float = Field(0.6, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_new_tokens: PositiveInt = Field(256)
    stop: list[StrictStr] = Field(default_factory=list)

    def to_dict(self):
        return super().to_dict()

    def to_json(self):
        return super().to_json()


class ArtifactInfo(BaseConfigModel):
    """
    Information about the model artifacts.
    """

    model_path: StrictStr = Field(..., description="Absolute or relative path to the .gguf / HF directory")
    tokenizer_json: StrictStr | None = None

    def path_obj(self, root: Path | None = None) -> Path:
        p = Path(self.model_path)
        return (root / p) if (root and not p.is_absolute()) else p

    def to_dict(self):
        # Pydantic v2
        return self.model_dump()

    def to_json(self):
        # Pydantic v2
        return self.model_dump_json(indent=2, ensure_ascii=False)


class ModelCard(BaseConfigModel):
    """
    Model card with metadata and configuration.
    """

    name: StrictStr
    slug: StrictStr
    version: StrictStr = "0.1.0"
    family: Literal["LLaMA", "Qwen", "Mistral", "Phi", "Unknown"] = "LLaMA"
    task: Literal["chat", "completion"] = "chat"

    runtime: Literal["llamacpp"] = "llamacpp"
    quantization: dict | None = None

    artifacts: ArtifactInfo
    chat_template: StrictStr | None = None
    system_prompt: StrictStr | None = None
    tags: list[StrictStr] = Field(default_factory=list)

    root_dir: StrictStr | None = Field(
        default=None,
        description="Root directory of the model (filled by the registry during discovery).",
    )

    @field_validator("slug")
    @classmethod
    def _norm_slug(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "-")

    def model_file(self) -> Path:
        return self.artifacts.path_obj(Path(self.root_dir) if self.root_dir else None)

    def to_dict(self):
        d = super().to_dict()
        return d

    def to_json(self):
        return super().to_json()


class DiscoveryConfig(BaseConfigModel):
    models_root: StrictStr = Field(..., description="Diretório com subpastas de modelos")

    def to_dict(self):
        return super().to_dict()

    def to_json(self):
        return super().to_json()
