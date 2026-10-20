"""Versioned task, experiment, and receipt contracts.

[Role] Define the task, native harness, execution and measurement interfaces.
[Invariants] Missing measurements stay distinct from measured zero or success.
[Timeouts & Locks] Models declare budgets; parsing owns no locks or subprocesses.
[Side-Effect Policy] Loads declared local contracts; never starts an experiment.
"""

from __future__ import annotations

import re
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mindie_bench.io import read_json, safe_relative, task_hash


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)


class Group(StrEnum):
    G0 = "G0"
    G1 = "G1"
    G2 = "G2"
    G3 = "G3"


class TerminalStatus(StrEnum):
    PASS = "completed_pass"
    FAIL = "completed_fail"
    INFRA = "infra_failed"
    UNKNOWN = "unknown"


class TaskInfo(StrictModel):
    instance_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    domain_category: str
    difficulty: Literal["easy", "medium", "hard"] = "hard"
    evidence: Literal["synthetic", "candidate", "verified"] = "candidate"
    cluster: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    role: Literal["seed", "eval"] = "eval"
    source: dict[str, Any] = Field(default_factory=dict)


class KnowledgeSpec(StrictModel):
    corpus_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    knowledge_gradient: Literal["near_exact", "analogical", "compositional", "stale", "no_hit"]
    same_origin_task_hashes: list[str] = Field(default_factory=list)
    analogical_task_hashes: list[str] = Field(default_factory=list)
    oracle_block_ids: list[str] = Field(default_factory=list)
    target_files: list[str] = Field(default_factory=list)
    conditions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("same_origin_task_hashes", "analogical_task_hashes")
    @classmethod
    def package_ids(cls, values: list[str]) -> list[str]:
        if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) for value in values):
            raise ValueError("invalid corpus package ID")
        if len(values) != len(set(values)):
            raise ValueError("knowledge package IDs must be unique within each relation")
        return values

    @model_validator(mode="after")
    def distinct_relations(self):
        if set(self.same_origin_task_hashes) & set(self.analogical_task_hashes):
            raise ValueError("same-origin and analogical knowledge packages must be distinct")
        return self

    @field_validator("target_files")
    @classmethod
    def confined_files(cls, values: list[str]) -> list[str]:
        return [safe_relative(value) for value in values]


class Budget(StrictModel):
    max_turns: int = Field(default=25, gt=0)
    max_npu_runs: int = Field(default=4, gt=0)
    timeout_sec: float = Field(default=1080, gt=0)
    verifier_timeout_sec: float = Field(default=120, gt=0)
    max_output_bytes: int = Field(default=32 * 1024 * 1024, gt=0)


class SourceEnvironment(StrictModel):
    repo_path: str
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    npu_model: str | None = None
    npu_count: int = Field(default=0, ge=0)
    replay_commands: list[str] = Field(default_factory=list)

    @field_validator("replay_commands")
    @classmethod
    def command_names(cls, values: list[str]) -> list[str]:
        if any(not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in values):
            raise ValueError("replay commands must be executable names")
        return values


class OracleProbe(StrictModel):
    """A task-owned negative control, executed outside the foreground process."""

    name: str = Field(min_length=1)
    command: list[str] = Field(min_length=1)
    expected_returncode: int
    stdout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class OracleContract(StrictModel):
    # verifier/ and oracle/ are always protected, including newly added files.
    protected_paths: list[str] = Field(default_factory=list)
    probes: list[OracleProbe] = Field(default_factory=list)

    @field_validator("protected_paths")
    @classmethod
    def confined_paths(cls, values: list[str]) -> list[str]:
        return [safe_relative(value) for value in values]


class PerformanceContract(StrictModel):
    """Admission requirements; shape contents stay in the private judge."""

    private_shape_sha256: list[str] = Field(min_length=2)
    min_buffers: int = Field(default=2, ge=2)
    samples_per_shape: int = Field(default=2, ge=2)
    require_aclgraph_capture: bool = True
    require_workspace_measurement: bool = True
    require_e2e_warm_server: bool = True
    min_speedup_ratio: float = Field(default=1.0, gt=0)
    min_e2e_warm_server_speedup: float = Field(default=1.0, gt=0)
    max_workspace_delta_bytes: int | None = None

    @field_validator("private_shape_sha256")
    @classmethod
    def distinct_private_shapes(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)) or any(
            not re.fullmatch(r"[0-9a-f]{64}", value) for value in values
        ):
            raise ValueError("performance requires distinct private shape SHA-256 identities")
        return values

    @model_validator(mode="after")
    def rotation_is_admissible(self):
        if self.samples_per_shape < self.min_buffers:
            raise ValueError("performance samples must cover every required buffer")
        return self


class TaskSpec(StrictModel):
    schema_version: Literal["1.0"]
    task: TaskInfo
    mindie_knowledge: KnowledgeSpec
    budget: Budget = Field(default_factory=Budget)
    environment: dict[Literal["l1", "l2", "l3"], SourceEnvironment]
    oracle: OracleContract = Field(default_factory=OracleContract)
    performance: PerformanceContract | None = None

    @model_validator(mode="after")
    def performance_admission(self):
        labels = [self.task.domain_category, str(self.task.source.get("task_type", ""))]
        if any("performance" in label.casefold() for label in labels) and self.performance is None:
            raise ValueError("performance task requires an explicit performance contract")
        return self

    @classmethod
    def load(cls, root: Path) -> TaskSpec:
        spec = cls.model_validate(tomllib.loads((root / "task.toml").read_text()))
        for environment in spec.environment.values():
            if not Path(environment.repo_path).is_absolute():
                environment.repo_path = str((root / environment.repo_path).resolve())
        if not (root / "instruction.md").is_file():
            raise ValueError("task is missing instruction.md")
        if not (root / "verifier/test.sh").is_file():
            raise ValueError("task is missing external verifier/test.sh")
        if spec.task.evidence == "verified":
            if not (root / "soundness_receipt.json").is_file():
                raise ValueError("verified task has no soundness receipt")
            proof = read_json(root / "soundness_receipt.json")
            if (
                proof.get("task_sha256") != task_hash(root)
                or proof.get("before_reward") != 0.0
                or proof.get("after_reward") != 1.0
                or proof.get("status") != "completed"
            ):
                raise ValueError("verified task soundness proof is stale or incomplete")
        return spec


class MCPServer(StrictModel):
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    component: Literal["remote-dev", "knowledge", "npu-top", "coordinator"]
    required_tools: list[str] = Field(default_factory=list)


class UnknownExternalEffects(RuntimeError):
    """A model/remote write may have happened; automatic repetition is forbidden."""


class NativeDSHPlugin(StrictModel):
    package_path: str
    runtime_python: str
    redactor_executable: str
    domain: str = "vllm-ascend"
    contribution_enabled: Literal[False] = False


class GroupConfig(StrictModel):
    mcp_servers: dict[str, MCPServer] = Field(default_factory=dict)
    skills_path: str | None = None
    capture_command: list[str] = Field(default_factory=list)
    capture_model: str | None = None
    capture_max_model_calls: int = Field(default=1, gt=0)
    capture_protocol: Literal["material-package", "graded-c6"] = "material-package"
    capture_timeout_sec: float = Field(default=1200, gt=0)
    dsh_summary_plugin: str | None = None
    native_dsh_plugin: NativeDSHPlugin | None = None
    knowledge_mode: Literal["advisory", "explicit", "oracle"] = "advisory"
    oracle_context_files: list[str] = Field(default_factory=list)

    @field_validator("oracle_context_files")
    @classmethod
    def confined_oracle_context(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("oracle context paths must be unique")
        for value in values:
            if not safe_relative(value).startswith("tasks/") or not value.endswith(".md"):
                raise ValueError("oracle context must select task Markdown in the filtered corpus")
        return values

    @model_validator(mode="after")
    def explicit_knowledge_condition(self):
        if (self.knowledge_mode == "oracle") != bool(self.oracle_context_files):
            raise ValueError("only oracle mode requires explicitly selected context files")
        return self

    @field_validator("mcp_servers")
    @classmethod
    def confined_server_names(cls, values):
        if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name) for name in values):
            raise ValueError("MCP server names must be safe artifact identifiers")
        return values


class CorpusConfig(StrictModel):
    path: str
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")


class HarnessConfig(StrictModel):
    name: Literal["codex", "claude-code", "dsh", "kimi", "fixture"]
    executable: str
    model: str
    expected_version: str
    reasoning_effort: str | None = None
    model_catalog_path: str | None = None
    provider_url: str | None = None
    api_key_env: str | None = None
    api_protocol: Literal["openai-responses", "openai-completions", "anthropic-messages"] = (
        "openai-responses"
    )
    input_usd_per_million: float | None = Field(default=None, ge=0)
    cached_usd_per_million: float | None = Field(default=None, ge=0)
    output_usd_per_million: float | None = Field(default=None, ge=0)
    cache_creation_usd_per_million: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def explicit_auth(self) -> HarnessConfig:
        if self.name != "fixture" and (not self.provider_url or not self.api_key_env):
            raise ValueError("native harnesses require an explicit model endpoint and API-key env name")
        return self


class RemoteConfig(StrictModel):
    host: str = Field(pattern=r"^[A-Za-z0-9_.@-]+$")
    container: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    repo_path: str
    slots_root: str = "/tmp/mindie-bench-slots"
    devices: list[int] = Field(min_length=1)
    memory_limit_mb: int = Field(default=500, ge=0)
    probe_command: list[str] = Field(min_length=1)
    runtime_read_paths: list[str] = Field(default_factory=list)
    component_runtime: dict[str, Any] | None = None

    @field_validator("devices")
    @classmethod
    def unique_devices(cls, values):
        if len(set(values)) != len(values) or any(v < 0 for v in values):
            raise ValueError("NPU IDs must be unique and nonnegative")
        return values


class ExperimentConfig(StrictModel):
    schema_version: Literal["1.0"]
    harness: HarnessConfig
    groups: dict[Group, GroupConfig] = Field(default_factory=dict)
    corpora: dict[str, CorpusConfig] = Field(default_factory=dict)
    component_revisions: dict[str, str] = Field(default_factory=dict)
    runtime_read_paths: list[str] = Field(default_factory=list)
    knowledge_runtime_read_paths: list[str] = Field(default_factory=list)
    backend: Literal["native", "fixture"] = "native"
    remote: RemoteConfig | None = None

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        return cls.model_validate(tomllib.loads(path.read_text()))


class VerifierResult(StrictModel):
    reward: Literal[0.0, 1.0]
    speedup_ratio: float | None = Field(default=None, ge=0)
    max_abs_error: float | None = Field(default=None, ge=0)
    cosine_similarity: float | None = Field(default=None, ge=-1, le=1)
    aclgraph_capture_compatible: bool | None = Field(default=None, strict=True)
    memory_workspace_delta_mb: float | None = Field(
        default=None,
        strict=True,
        description="Candidate minus baseline workspace; MB = 1,000,000 bytes.",
    )
    e2e_warm_server_speedup: float | None = Field(default=None, ge=0)
    performance_evidence: dict[str, Any] | None = None
    oracle_status: Literal["intact", "oracle_tampered"] | None = None
    timed_out: bool = False
    failure_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("reward", mode="before")
    @classmethod
    def numeric_reward(cls, value):
        if type(value) not in {int, float}:
            raise ValueError("reward must be a numeric 0 or 1, not a truthy value")
        return value

    @model_validator(mode="after")
    def tamper_is_failure(self):
        if self.oracle_status == "oracle_tampered" and (
            self.reward != 0.0 or not self.failure_reason or not self.warnings
        ):
            raise ValueError("oracle_tampered requires reward 0, a failure reason and warning")
        return self


class ToolResultObservation(StrictModel):
    """Bounded result facts; original tool payloads stay in the private native log."""

    schema_version: Literal["mindie-bench-tool-observation/1"] = "mindie-bench-tool-observation/1"
    source: Literal[
        "codex_mcp", "claude_tool_result", "dsh_tool_result", "native_command", "native_file"
    ]
    classification: Literal[
        "success",
        "business_success",
        "business_failure",
        "pending",
        "unknown",
        "transport_error",
        "protocol_error",
        "component_error",
        "unclassified",
    ]
    operation_kind: Literal["execution", "observation", "control", "file", "knowledge", "other"]
    result_schema: str | None = Field(default=None, max_length=96)
    native_status: str | None = Field(default=None, max_length=96)
    mcp_is_error: bool | None = None
    remote_outcome: str | None = Field(default=None, max_length=96)
    remote_status: str | None = Field(default=None, max_length=96)
    exit_code: int | None = Field(default=None, strict=True)
    job_id: str | None = Field(default=None, max_length=256)
    invocation_id: str | None = Field(default=None, max_length=256)
    target_id: str | None = Field(default=None, max_length=256)
    submission_state: str | None = Field(default=None, max_length=96)
    error_category: str | None = Field(default=None, max_length=96)
    error_domain: Literal["transport", "protocol", "component"] | None = None
    operation_completed: bool | None = None
    component_duration_ms: float | None = Field(default=None, ge=0)
    result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    classification_reason: str = Field(max_length=96)


class ToolCall(StrictModel):
    id: str
    name: str
    component: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    turn: int = Field(ge=1)
    started_sec: float | None = None
    ended_sec: float | None = None
    is_error: bool | None = None
    result_observation: ToolResultObservation | None = None


class Tokens(StrictModel):
    input_uncached: int = Field(default=0, ge=0)
    input_cached: int = Field(default=0, ge=0)
    cache_creation: int = Field(default=0, ge=0)
    output: int = Field(default=0, ge=0)
    indexer_small_model_total: int = Field(default=0, ge=0)
    complete: bool = False

    @model_validator(mode="after")
    def cache_writes_are_subset(self):
        if self.cache_creation > self.input_uncached:
            raise ValueError("cache creation tokens must be included in non-read input tokens")
        return self


class Trajectory(StrictModel):
    calls: list[ToolCall] = Field(default_factory=list)
    tokens: Tokens = Field(default_factory=Tokens)
    total_turns: int = Field(default=0, ge=0)
    human_interventions: int | None = None
    native_completed: bool = False
    native_error: str | None = None
    native_session_id: str | None = None
    public_messages: list[dict[str, str]] = Field(default_factory=list)
    tool_observation_schema: Literal["mindie-bench-tool-observation/1"] | None = None


class TrialReceipt(StrictModel):
    schema_version: Literal["mindie-bench-receipt/1.0"] = "mindie-bench-receipt/1.0"
    trial_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
    instance_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    repetition: int = Field(ge=1)
    tier: Literal["l1", "l2", "l3"]
    ablation_group: Group
    knowledge_gradient: str
    task_role: Literal["seed", "eval"] = "eval"
    cluster: str | None = None
    task_digest: str
    experiment_digest: str
    environment_fingerprint: dict[str, Any]
    terminal_status: TerminalStatus
    failure_reason: str | None = None
    primary_metrics: dict[str, Any] = Field(default_factory=dict)
    auxiliary_metrics: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, str]] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    evidence: Literal["synthetic", "candidate", "verified"]

    @model_validator(mode="after")
    def consistent_outcome(self) -> TrialReceipt:
        reward = self.primary_metrics.get("reward")
        resolved = self.primary_metrics.get("resolved")
        if self.terminal_status == TerminalStatus.PASS and (reward != 1.0 or resolved is not True):
            raise ValueError("completed_pass requires external reward=1 and resolved=true")
        if self.terminal_status == TerminalStatus.FAIL and (reward != 0.0 or resolved is not False):
            raise ValueError("completed_fail requires external reward=0 and resolved=false")
        if (
            self.terminal_status in {TerminalStatus.INFRA, TerminalStatus.UNKNOWN}
            and resolved is not None
        ):
            raise ValueError("infrastructure and unknown attempts must not claim a resolved outcome")
        if self.terminal_status != TerminalStatus.PASS and not self.failure_reason:
            raise ValueError("non-pass receipt must retain a failure reason")
        return self
