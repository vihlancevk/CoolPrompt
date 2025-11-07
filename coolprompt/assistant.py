from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from langchain_core.language_models.base import BaseLanguageModel
from sklearn.model_selection import train_test_split

from coolprompt.data_generator.generator import SyntheticDataGenerator
from coolprompt.evaluator import Evaluator, validate_and_create_metric
from coolprompt.language_model.llm import DefaultLLM
from coolprompt.optimizer.distill_prompt.run import distillprompt
from coolprompt.optimizer.hype import hype_optimizer
from coolprompt.optimizer.reflective_prompt import reflectiveprompt
from coolprompt.prompt_assistant.prompt_assistant import PromptAssistant
from coolprompt.task_detector.detector import TaskDetector
from coolprompt.utils.correction.corrector import correct
from coolprompt.utils.correction.rule import LanguageRule
from coolprompt.utils.enums import Method, Task
from coolprompt.utils.logging_config import logger, set_verbose, setup_logging
from coolprompt.utils.prompt_templates.default_templates import (
    CLASSIFICATION_TASK_TEMPLATE,
    GENERATION_TASK_TEMPLATE,
)
from coolprompt.utils.prompt_templates.hype_templates import (
    CLASSIFICATION_TASK_TEMPLATE_HYPE,
    GENERATION_TASK_TEMPLATE_HYPE,
)
from coolprompt.utils.var_validation import (
    validate_method,
    validate_model,
    validate_run,
    validate_task,
    validate_verbose,
)


class PromptTuner:
    """Prompt optimization tool supporting multiple methods."""

    TEMPLATE_MAP: ClassVar[dict[tuple[Task, Method], str]] = {
        (Task.CLASSIFICATION, Method.HYPE): CLASSIFICATION_TASK_TEMPLATE_HYPE,
        (Task.CLASSIFICATION, Method.REFLECTIVE): CLASSIFICATION_TASK_TEMPLATE,
        (Task.CLASSIFICATION, Method.DISTILL): CLASSIFICATION_TASK_TEMPLATE,
        (Task.GENERATION, Method.HYPE): GENERATION_TASK_TEMPLATE_HYPE,
        (Task.GENERATION, Method.REFLECTIVE): GENERATION_TASK_TEMPLATE,
        (Task.GENERATION, Method.DISTILL): GENERATION_TASK_TEMPLATE,
    }

    def __init__(
        self,
        target_model: BaseLanguageModel = None,
        system_model: BaseLanguageModel = None,
        logs_dir: str | Path | None = None,
    ) -> None:
        """Initializes the tuner with a LangChain-compatible language model.

        Args:
            target_model (BaseLanguageModel): Any LangChain BaseLanguageModel
                instance which supports invoke(str) -> str. Used for
                optimization processes. Will use DefaultLLM if not provided.
            system_model (BaseLanguageModel): Any LangChain BaseLanguageModel
                instance which supports invoke(str) -> str. Used for
                synthetic data generation, feedback generation, etc.
                Will use the `target_model` if not provided.
            logs_dir (str | Path, optional): logs saving directory.
                Defaults to None.
        """
        setup_logging(logs_dir)
        self._target_model = target_model or DefaultLLM.init()
        self._system_model = system_model or self._target_model

        self.init_metric = None
        self.init_prompt = None
        self.final_metric = None
        self.final_prompt = None
        self.assistant_feedback = None

        self.synthetic_dataset = None
        self.synthetic_target = None

        logger.info("Validating the target model")
        validate_model(self._target_model)

        if self._system_model is not self._target_model:
            logger.info("Validating the system model")
            validate_model(self._system_model)

        logger.info("PromptTuner successfully initialized")

    def get_task_prompt_template(self, task: str, method: str) -> str:
        """Returns the prompt template for the given task.

        Args:
            task (str):
                The type of task, either "classification" or "generation".
            method (str):
                Optimization method to use.
                Available methods are: ['hype', 'reflective', 'distill']

        Returns:
            str: The prompt template for the given task.
        """

        logger.debug(f"Getting prompt template for {task} task and {method} method")
        task = validate_task(task)
        method = validate_method(method)
        return self.TEMPLATE_MAP[(task, method)]

    def run(  # noqa: PLR0913
        self,
        start_prompt: str,
        task: str | None = None,
        dataset: Iterable[str] | None = None,
        target: Iterable[int] | Iterable[str] | None = None,
        method: str = "hype",
        metric: str | None = None,
        problem_description: str | None = None,
        validation_size: float = 0.25,
        generate_num_samples: int = 10,
        verbose: int | None = None,
        *,
        train_as_test: bool = False,
        feedback: bool = True,
        **kwargs: dict[str, Any],
    ) -> str:
        """Optimizes prompts using provided model.

        Args:
            start_prompt (str): Initial prompt text to optimize.
            task (str | None):
                Type of task to optimize for (classification or generation).
            dataset (Iterable[str] | None):
                Dataset iterable object for autoprompting optimization.
            target (Iterable[int] | Iterable[str] | None):
                Target iterable object for autoprompting optimization.
            method (str): Optimization method to use.
                Available methods are: ['hype', 'reflective', 'distill']
                Defaults to hype.
            metric (str | None): Metric to use for optimization.
            problem_description (str | None): a string that contains
                short description of problem to optimize.
            validation_size (float):
                A float that must be between 0.0 and 1.0 and
                represent the proportion of the dataset
                to include in the validation split.
                Defaults to 0.25.
            generate_num_samples (int):
                A number of dataset and target samples to generate with PromptAssistant
            verbose (int | None): Parameter for logging configuration:
                0 - no logging
                1 - steps logging
                2 - steps and prompts logging
            train_as_test (bool):
                Either to use all the provided data as
                the train and the test dataset at the same time or not.
                If sets to True, the validation_size parameter will be ignored.
                Defaults to False.
            feedback (bool):
                PromptAssistant interpretation of optimization results
                Defaults to True.
            **kwargs (dict[str, Any]): other key-word arguments.

        Returns:
            final_prompt: str - The resulting optimized prompt
            after applying the selected method.

        Raises:
            ValueError: If an invalid task type is provided.
            ValueError: If a problem description is not provided
                for ReflectivePrompt.

        Note:
            Uses HyPE optimization
            when dataset or method parameters are not provided.

            Uses default metric for the task type
            if metric parameter is not provided:
            f1 for classification, meteor for generation.

            if dataset is not None, you can find evaluation results
            in self.init_metric and self.final_metric
        """
        if verbose is not None:
            validate_verbose(verbose)
            set_verbose(verbose)

        task_detector = TaskDetector(self._system_model)
        if task is None:
            task = task_detector.generate(start_prompt)

        logger.info("Validating args for PromptTuner running")
        task, method = validate_run(
            start_prompt,
            task,
            dataset,
            target,
            method,
            problem_description,
            validation_size,
        )
        metric = validate_and_create_metric(task, metric)
        evaluator = Evaluator(self._target_model, task, metric)
        generator = SyntheticDataGenerator(self._system_model)

        if dataset is None:
            dataset, target, problem_description = generator.generate(
                prompt=start_prompt,
                task=task,
                problem_description=problem_description,
                num_samples=generate_num_samples,
            )
            self.synthetic_dataset = dataset
            self.synthetic_target = target

        if problem_description is None:
            problem_description = generator.generate_problem_description(prompt=start_prompt)

        dataset_split: tuple[Iterable[str], Iterable[str], Iterable[str], Iterable[str]] = self._get_dataset_split(
            dataset=dataset,
            target=target,
            validation_size=validation_size,
            train_as_test=train_as_test,
        )

        logger.info("=== Starting Prompt Optimization ===")
        logger.info(f"Method: {method}, Task: {task}")
        logger.info(f"Metric: {metric}, Validation size: {validation_size}")
        if dataset:
            logger.info(f"Dataset: {len(dataset)} samples")
        else:
            logger.info("No dataset provided")
        if target:
            logger.info(f"Target: {len(target)} samples")
        else:
            logger.info("No target provided")
        if kwargs:
            logger.debug(f"Additional kwargs: {kwargs}")

        final_prompt: str = self._optimize_prompt(method, start_prompt, problem_description, dataset_split, evaluator)

        logger.info("Running the prompt format checking...")
        final_prompt = correct(
            prompt=final_prompt,
            rule=LanguageRule(self._system_model),
            start_prompt=start_prompt,
        )

        logger.debug(f"Final prompt:\n{final_prompt}")
        template = self.TEMPLATE_MAP[(task, method)]
        logger.info(f"Evaluating on given dataset for {task} task...")
        self.init_metric = evaluator.evaluate(
            prompt=start_prompt,
            dataset=dataset_split[1],
            targets=dataset_split[3],
            template=template,
        )
        self.final_metric = evaluator.evaluate(
            prompt=final_prompt,
            dataset=dataset_split[1],
            targets=dataset_split[3],
            template=template,
        )
        logger.info(f"Initial {metric} score: {self.init_metric}, final {metric} score: {self.final_metric}")

        self.init_prompt = start_prompt
        self.final_prompt = final_prompt

        logger.info("=== Prompt Optimization Completed ===")

        if feedback:
            self._process_feedback(start_prompt, final_prompt)

        return final_prompt

    @staticmethod
    def _get_dataset_split(
        dataset: Iterable[str],
        target: Iterable[str],
        validation_size: float,
        *,
        train_as_test: bool,
    ) -> tuple[Iterable[str], Iterable[str], Iterable[str], Iterable[str]]:
        """Provides a train/val dataset split.

        Args:
            dataset (Iterable[str]):
                Provided dataset.
            target (Iterable[str]):
                Provided targets for the dataset.
            validation_size (float):
                Provided size of validation subset.
            train_as_test (bool):
                Either to use all data for train and validation or split it.

        Returns:
            Tuple[Iterable[str], Iterable[str], Iterable[str], Iterable[str]]:
                a tuple of train dataset, validation dataset,
                train targets and validation targets.
        """
        if train_as_test:
            return dataset, dataset, target, target
        train_data, val_data, train_targets, val_targets = train_test_split(dataset, target, test_size=validation_size)
        return train_data, val_data, train_targets, val_targets

    def _optimize_prompt(
        self,
        method: Method,
        start_prompt: str,
        problem_description: str,
        dataset_split: tuple[Iterable[str], Iterable[str], Iterable[str], Iterable[str]],
        evaluator: Evaluator,
        **kwargs: dict[str, Any],
    ) -> str:
        if method is Method.HYPE:
            return hype_optimizer(
                model=self._target_model,
                prompt=start_prompt,
                problem_description=problem_description,
            )
        if method is Method.REFLECTIVE:
            return reflectiveprompt(
                model=self._target_model,
                dataset_split=dataset_split,
                evaluator=evaluator,
                problem_description=problem_description,
                initial_prompt=start_prompt,
                **kwargs,
            )
        if method is Method.DISTILL:
            return distillprompt(
                model=self._target_model,
                dataset_split=dataset_split,
                evaluator=evaluator,
                initial_prompt=start_prompt,
                **kwargs,
            )
        return ""

    def _process_feedback(self, start_prompt: str, final_prompt: str) -> None:
        prompt_assistant: PromptAssistant = PromptAssistant(self._target_model)
        self.assistant_feedback: str = correct(
            prompt=prompt_assistant.get_feedback(start_prompt, final_prompt),
            rule=LanguageRule(self._system_model),
            start_prompt=start_prompt,
        )

        logger.info("=== Assistant's feedback ===")
        logger.info(self.assistant_feedback)
