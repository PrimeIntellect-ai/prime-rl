"""Launch a Flow plugin with the same config and run-directory conventions as eval."""

from pathlib import Path
from uuid import uuid4

from pydantic import Field, NonNegativeInt, SerializeAsAny, model_validator
from verifiers.v1.configs.flow import FlowConfig as PipelineConfig
from verifiers.v1.flow.events import Status
from verifiers.v1.utils.loaders import flow_config_type, narrow_plugin_field

from prime_rl.configs.shared import LogConfig, RunConfig
from prime_rl.utils.config import BaseConfig, default_output_dir


class FlowConfig(BaseConfig):
    flow: SerializeAsAny[PipelineConfig]
    """Installed package id and its pipeline-specific configuration."""
    run: RunConfig = Field(default_factory=RunConfig)
    output_dir: Path = Field(default_factory=default_output_dir)
    dashboard: bool = True
    log: LogConfig = LogConfig()

    @model_validator(mode="before")
    @classmethod
    def resolve_flow(cls, data):
        if isinstance(data, dict):
            narrow_plugin_field(data, "flow", flow_config_type)
        return data

    @model_validator(mode="after")
    def resolve_run(self):
        if not self.flow.id:
            raise ValueError("flow.id must name an installed Flow plugin")
        if self.run.name is None:
            self.run.name = f"{self.flow.id}--{uuid4().hex[:8]}"
        if self.run.dir is None:
            self.run.dir = self.run.name
        return self

    @property
    def run_dir(self) -> Path:
        assert self.run.dir is not None
        return self.output_dir / self.run.dir


class InspectConfig(BaseConfig):
    root: Path
    job: str | None = None


class ApplyConfig(BaseConfig):
    root: Path
    job: str
    stage: str | None = None
    status: Status | None = None
    reason: str | None = None
    note: str | None = None
    outcome: str | None = None
    report: str | None = None
    data_file: Path | None = None
    """Complete job data snapshot, validated against its recorded model."""
    expected: NonNegativeInt | None = None
    """Inspected workflow revision; required for data updates."""


class DrainConfig(BaseConfig):
    root: Path
