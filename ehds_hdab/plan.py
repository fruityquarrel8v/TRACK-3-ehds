from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QueryPlan:
    concepts: dict[str, dict[str, Any]] = field(default_factory=dict)
    group_by: list[dict[str, Any]] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    cohort: list[dict[str, Any]] = field(default_factory=list)
    record_types: list[str] = field(default_factory=list)
    patient_level_listing: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @property
    def empty(self):
        return not self.concepts and not self.metrics and not self.cohort
