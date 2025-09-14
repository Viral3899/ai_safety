from dataclasses import dataclass

@dataclass
class SafetyResult:
    label: str
    score: float
    details: dict | None = None
