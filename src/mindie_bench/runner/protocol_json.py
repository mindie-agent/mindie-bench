"""JSON encoding for the benchmark's immutable protocol records.

[Role] Preserve each nested record's public result schema during JSON export.
[Invariants] Derived denominators and interpretation fields survive nesting;
None remains missing and enum values use their declared protocol spelling.
[Timeouts & Locks] Pure in-memory conversion, no locks or waits.
[Side-Effect Policy] No I/O and no mutation of source records.
"""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any


def json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, JsonRecord):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return record_payload(value)
    if isinstance(value, Mapping):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((json_value(item) for item in value), key=str)
    return value


def record_payload(record: Any) -> dict:
    return {field.name: json_value(getattr(record, field.name)) for field in fields(record)}


class JsonRecord:
    def to_dict(self) -> dict:
        return record_payload(self)
