import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from evaluate import load
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage

from coolprompt.utils.arithmetics import clip, extract_number_from_text, mean
from coolprompt.utils.enums import Task
from coolprompt.utils.logging_config import logger
from coolprompt.utils.parsing import extract_answer


class HFEvaluateMetric:
    _HF_EVALUATE_METRIC_DIR: ClassVar[Path] = Path(os.getenv("HF_EVALUATE_METRIC_DIR", default=""))

    def __init__(self, name: str) -> None:
        """Initialize metric with specified evaluate library metric name.

        Args:
            name (str): Name of metric to load from evaluate library
        """

        self._return_parameter = name
        self._metric = load(str(self._HF_EVALUATE_METRIC_DIR / name))
        self._compute_kwargs_func = lambda _, __: {}
        super().__init__()

    def _compute_raw(
        self,
        outputs: list[int | str],
        targets: list[int | str],
        _: list[str] | None = None,
    ) -> float:
        """Compute metric value from preprocessed model answers.

        Args:
            outputs (list[int | str]): Model predictions (text for generation,
            labels for classification)
            targets (list[int | str]): Ground truth labels
        Returns:
            float: Computed metric value
        """

        return self._metric.compute(
            predictions=outputs, references=targets, **self._compute_kwargs_func(outputs, targets)
        )[self._return_parameter]


class BaseMetric(ABC):
    """Abstract base class for implementing evaluation metrics.

    Provides common infrastructure for loading metrics
    from HuggingFace's evaluate library and defining
    metric computation interfaces.

    Attributes:
        ANS_TAGS: tuple - Start and end tags for answer extraction
        FORMAT_MISMATCH_LABEL: int - Special value indicating parsing failure
    """

    ANS_TAGS = ("<ans>", "</ans>")

    def __init__(self) -> None:
        """Initialize metric"""

        super().__init__()

    @abstractmethod
    def _compute_raw(
        self, outputs: list[str | int], targets: list[str | int], dataset: list[str] | None = None
    ) -> float:
        """Compute metric value from preprocessed model answers.

        Args:
            outputs (list[str|int]): Model predictions (text for generation,
            labels for classification)
            targets (list[str|int]): Ground truth labels
        Returns:
            float: Computed metric value
        """

    @abstractmethod
    def _encode_labels(
        self, output_labels: list[str | int], targets: list[str | int]
    ) -> tuple[list[int] | list[str], list[int] | list[str]]:
        """Encode labels into internal representation for both
        outputs and targets.

        Args:
            output_labels (list[str|int]): Extracted labels from model outputs.
            targets (list[str|int]): Ground truth labels.
        Returns:
            tuple[list[int], list[int]]: Encoded output labels
            and encoded targets.
        """

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        pass

    def compute(self, outputs: list[str | int], targets: list[str | int], dataset: list[str] | None = None) -> float:
        """Compute metric value from text model outputs

        Must be implemented by subclasses to handle input formatting.

        Args:
            outputs (list[str|int]): Model predictions (just text)
            targets (list[str|int]): Ground truth labels
            dataset (list[str] | None): Dataset used for evaluation
        Returns:
            float: Computed metric value
        """
        output_labels: list[int | str] = [
            extract_answer(output, self.ANS_TAGS, self.FORMAT_MISMATCH_LABEL) for output in outputs
        ]
        targets = list(map(str, targets))
        encoded_output_labels, encoded_targets = self._encode_labels(output_labels, targets)
        return self._compute_raw(encoded_output_labels, encoded_targets, dataset)

    def __str__(self) -> str:
        return self.get_name()

    def __eq__(self, other: type["BaseMetric"]) -> bool:
        if type(self) is not type(other):
            return False
        return self.get_name() == other.get_name()

    def __hash__(self) -> int:
        return hash((type(self), self.get_name()))


class ClassificationMetric(BaseMetric, ABC):
    """Base class for classification metrics with answer parsing functionality.

    Handles extraction of labels from model outputs
    containing XML-style <ans> tags
    and label encoding for metric computation.
    """

    FORMAT_MISMATCH_LABEL = -1

    def __init__(self) -> None:
        """Initialize metric"""

        super().__init__()
        self.label_to_id = None

    def _encode_labels(self, output_labels: list[str | int], targets: list[str | int]) -> tuple[list[int], list[int]]:
        """Encode string labels into integer IDs for both outputs and targets.

        Args:
            output_labels (list[str|int]): Extracted labels from model outputs.
            targets (list[str|int]): Ground truth labels.
        Returns:
            tuple[list[int], list[int]]: Encoded output labels
            and encoded targets.
        """

        if self.label_to_id is None:
            self.extract_labels(targets)

        encoded_output_labels = [self.label_to_id.get(label, -1) for label in output_labels]
        encoded_targets = [self.label_to_id[label] for label in targets]
        return encoded_output_labels, encoded_targets

    def extract_labels(self, targets: list[str | int]) -> None:
        """Extract unique labels from targets and encode them into IDs.

        Args:
            targets (list[str  |  int]): Ground truth labels.
        """

        self.label_to_id = {}
        for x in targets:
            label = str(x)
            if label not in self.label_to_id:
                self.label_to_id[label] = len(self.label_to_id)


class GenerationMetric(BaseMetric):
    """Base class for generation metrics.

    Provides a generic implementation for metrics that compare generated text
    to reference text.
    """

    FORMAT_MISMATCH_LABEL = ""

    def __init__(self) -> None:
        """Initialize metric"""

        super().__init__()

    def _encode_labels(
        self, output_labels: list[str | int], targets: list[str | int]
    ) -> tuple[list[int] | list[str], list[int] | list[str]]:
        """Returns labels without encoding for generation metrics.

        Args:
            output_labels (list[str|int]): Extracted labels from model outputs.
            targets (list[str|int]): Ground truth labels.
        Returns:
            tuple[list[str], list[str]]: input values
        """

        return output_labels, targets


class AccuracyMetric(HFEvaluateMetric, ClassificationMetric):
    """Accuracy metric for classification tasks."""

    @staticmethod
    def get_name() -> str:
        return "accuracy"

    def __init__(self) -> None:
        super().__init__(self.get_name())


class F1Metric(HFEvaluateMetric, ClassificationMetric):
    """F1 metric for classification tasks with macro averaging."""

    @staticmethod
    def get_name() -> str:
        return "f1"

    def __init__(self) -> None:
        super().__init__(self.get_name())
        self._compute_kwargs_func = lambda _, __: {"average": "macro"}


class BleuMetric(HFEvaluateMetric, GenerationMetric):
    """BLEU metric for generation tasks."""

    @staticmethod
    def get_name() -> str:
        return "bleu"

    def __init__(self) -> None:
        super().__init__(self.get_name())


class RougeMetric(HFEvaluateMetric, GenerationMetric):
    """ROUGE metric for generation tasks."""

    @staticmethod
    def get_name() -> str:
        return "rouge"

    def __init__(self) -> None:
        super().__init__(self.get_name())
        self._return_parameter = "rougeL"


class MeteorMetric(HFEvaluateMetric, GenerationMetric):
    """METEOR metric for generation tasks."""

    @staticmethod
    def get_name() -> str:
        return "meteor"

    def __init__(self) -> None:
        super().__init__(self.get_name())


class BertScoreMetric(HFEvaluateMetric, GenerationMetric):
    """BertScore metric for generation tasks."""

    @staticmethod
    def get_name() -> str:
        return "bertscore"

    def __init__(self) -> None:
        super().__init__(self.get_name())
        self._compute_kwargs_func = lambda _, __: {"model_type": "bert-base-multilingual-cased"}
        self._return_parameter = "f1"

    def _compute_raw(self, outputs: list[int | str], targets: list[int | str], _: list[str] | None = None) -> float:
        f1_list = super()._compute_raw(outputs, targets)
        return sum(f1_list) / len(f1_list)


class GEvalMetric(HFEvaluateMetric, GenerationMetric):
    """GEval metric for generation tasks."""

    @staticmethod
    def get_name() -> str:
        return "geval"

    def __init__(self, model: BaseLanguageModel, prompt_template: str, metric_ceil: int = 10) -> None:
        super().__init__(self.get_name())
        self.model = model
        self.prompt_template = prompt_template
        self.metric_ceil = metric_ceil

    def _compute_raw(self, outputs: list[int | str], _: list[int | str], dataset: list[str] | None = None) -> float:
        requests = [
            self.prompt_template.format(metric_ceil=self.metric_ceil, request=request, responce=response)
            for request, response in zip(dataset, outputs, strict=True)
        ]

        answers = self.model.batch(requests)

        answers = [(int(a.content) if a.content.isdigit() else 0) if isinstance(a, AIMessage) else a for a in answers]

        answers = [clip(ans, 0, self.metric_ceil) / self.metric_ceil for ans in answers]

        return sum(answers) / len(answers)


class ExactMatchMetric(HFEvaluateMetric, GenerationMetric):
    """EM Metric for generation tasks."""

    @staticmethod
    def get_name() -> str:
        return "em"

    def __init__(self) -> None:
        super().__init__(self.get_name())

    def _compute_raw(self, outputs: list[int | str], targets: list[int | str], _: list[str] | None = None) -> float:
        targets = [extract_number_from_text(item) for item in targets]
        outputs = [extract_number_from_text(item) for item in outputs]
        return float(mean([o == t for o, t in zip(outputs, targets, strict=True)]))


CLASSIFICATION_METRIC_NAME_MAPPING = {metric.get_name(): metric for metric in ClassificationMetric.__subclasses__()}

GENERATION_METRIC_NAME_MAPPING = {metric.get_name(): metric for metric in GenerationMetric.__subclasses__()}


def validate_and_create_metric(task: Task, metric: str | None, model: BaseLanguageModel | None = None) -> BaseMetric:
    """
    Validates given metric in order to correspond the given task.
    Returns the given metric name back if the validation succeeded.

    Args:
        task (Task): The type of task, either "classification" or "generation".
        metric (str): Name of the metric to validate.
        model (BaseLanguageModel): model to use for evaluation (for GEval)
    Returns:
        str: the name of the metric.
    Raises:
        ValueError: If the specified task name is not recognized
        ValueError: If the specified metric name is not
            matched to the specified task name.
    """

    if metric is None:
        metric = get_default_metric(task)
    match task:
        case Task.CLASSIFICATION:
            if metric in CLASSIFICATION_METRIC_NAME_MAPPING:
                return CLASSIFICATION_METRIC_NAME_MAPPING[metric]()
            error_msg = (
                f"Invalid metric for {task} task: {metric}. "
                f"Available metrics: {', '.join(CLASSIFICATION_METRIC_NAME_MAPPING.keys())}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        case Task.GENERATION:
            if metric == "geval":
                if model is None:
                    error_msg = "Model for GEval metric must not be None"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                return GENERATION_METRIC_NAME_MAPPING[metric](model)
            if metric in GENERATION_METRIC_NAME_MAPPING:
                return GENERATION_METRIC_NAME_MAPPING[metric]()
            error_msg = (
                f"Invalid metric for {task} task: {metric}. "
                f"Available metrics: {', '.join(GENERATION_METRIC_NAME_MAPPING.keys())}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    error_msg = f"Invalid task: {task}Available tasks: classification, generation"
    logger.error(error_msg)
    raise ValueError(error_msg)


def get_default_metric(task: Task) -> str:
    """
    Returns default metric names for the provided task name.

    Args:
        task (Task): The type of task, either "classification" or "generation".
    Returns:
        str: the name of the default metric for the specified task.
    """

    match task:
        case Task.CLASSIFICATION:
            return "f1"
        case Task.GENERATION:
            return "meteor"
