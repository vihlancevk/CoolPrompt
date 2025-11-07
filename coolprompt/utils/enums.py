from enum import Enum


class Method(Enum):
    HYPE = "hype"
    REFLECTIVE = "reflective"
    DISTILL = "distill"

    def is_data_driven(self) -> bool:
        return self is not Method.HYPE

    def __str__(self) -> str:
        return self.value


class Task(Enum):
    CLASSIFICATION = "classification"
    GENERATION = "generation"

    def __str__(self) -> str:
        return self.value
