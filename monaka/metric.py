from typing import List, Any
from prettytable import PrettyTable

class MetricReporter:

    def __init__(self, name: str) -> None:
        self.g = 0
        self.c = 0
        self.s = 0
        self.name = name

    def reset(self):
        self.g = 0
        self.c = 0
        self.s = 0

    def update(self, golds: List[Any], systems: List[Any]):
        self.g += len(golds)
        self.s += len(systems)
        for g, s in zip(golds, systems):
            if g == s:
                self.c += 1

    def to_json(self):
        return {
            "gold": self.g,
            "correct": self.c,
            "system": self.s,
            "precision": self.c / self.s if self.s > 0 else None,
            "recall": self.c / self.g if self.g > 0 else None,
            "f1":  2.0 * self.c / (self.g + self.s) if self.g + self.s > 0 else None
        }
    
    def pretty(self):
        js = self.to_json()
        keys = ["name"] + list(js.keys())
        js["name"] = self.name
        table = PrettyTable(field_names=keys)
        table.add_row([js[k] for k in keys])
        print(table)

class SpanBasedMetricReporter(MetricReporter):

    def update(self, golds: List[Any], systems: List[Any]):
        self.g += len(golds)
        self.s += len(systems)
        correct = [1 for g in golds if g in systems]
        self.c += len(correct)
