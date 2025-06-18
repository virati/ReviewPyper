from dataclasses import dataclass


@dataclass
class candidate_article:
    title: str
    goal: str
    relevant: float
