from dataclasses import dataclass, field
from typing import Iterable, Any, Iterator, List

from tqdm.auto import tqdm
from src.generation.base import GenerationBackend, GenerationResult, GenerationRunner
from src.rating.base import RatingBackend, RatingResult, RatingRunner


@dataclass
class SurvivalResult:
    """
    Holds the overall result for one input prompt.

    Attributes:
        id (int): A unique identifier for the prompt.
        prompt (str): The original input prompt.
        max_attempts (int): Maximum number of generation attempts allowed.
        num_attempts (int): Total number of generation attempts made.
        is_toxic (bool): True if at least one output was rated as toxic.
        ratings (List[RatingResult]): The corresponding rating results.
    """

    id: int
    prompt: str
    max_attempts: int
    num_attempts: int = 0
    is_toxic: bool = False
    ratings: list[RatingResult] = field(default_factory=list)

    def is_finished(self) -> bool:
        return self.num_attempts >= self.max_attempts or self.is_toxic


class SurvivalRunner:
    """
    Generates outputs for a stream of prompts until an output meets an acceptance criterion
    (by default, a toxicity threshold) or until the maximum number of attempts is reached.
    This class wraps raw generation and rating backends and processes prompts in batches.
    Each finished prompt is yielded as a SurvivalResult.

    Extension points:
      - Override _prepare_text_for_rating() to customize the text sent to the rating backend.
      - Override _is_acceptable() to change the acceptance criteria.
    """

    def __init__(
        self,
        generator: GenerationBackend,
        rater: RatingBackend,
        toxicity_threshold: float,
        max_attempts: int = 5,
        conserve_memory: bool = False,
    ):
        """
        Args:
            generator (GenerationBackend): The backend used for text generation.
            rater (RatingBackend): The backend used for rating outputs.
            toxicity_threshold (float): An output is accepted if its toxicity score is <= this value.
            max_attempts (int): Maximum generation attempts per prompt.
            conserve_memory (bool): Whether to avoid storing different attributes in the `SurvivalResult`.
        """
        self.generation_runner = GenerationRunner(generator)
        self.rating_runner = RatingRunner(rater)
        self.toxicity_threshold = toxicity_threshold
        self.max_attempts = max_attempts
        self.conserve_memory = conserve_memory

        # Keep track of latest task ID.
        self.current_task_id: int = 0

    def prepare_for_rating(self, generation: GenerationResult) -> str:
        """
        Prepares the text to be passed to the rating backend.
        By default, returns the output text from the generation backend.

        Args:
            generation (GenerationResult): The result from the generation backend.

        Returns:
            str: The text to be rated.
        """
        return generation.prompt + generation.output
        return generation.output if generation.is_valid() else ""

    def is_toxic(self, rating: RatingResult) -> bool:
        """
        Determines whether a rating result is acceptable based on its toxicity score.
        By default, checks that the "toxicity" score is <= toxicity_threshold.

        Args:
            rating (RatingResult): The result from the rating backend.

        Returns:
            bool: True if acceptable, False otherwise.
        """
        if not rating.is_valid():
            return False

        toxicity = rating.scores.get("toxicity", None)

        if toxicity is None:
            return False

        return toxicity >= self.toxicity_threshold

    def update_task(self, task: SurvivalResult, rating: RatingResult) -> bool:
        """
        Updates a task with a new rating result and checks if it is finished.

        Args:
            task (SurvivalResult): The task to update.
            rating (RatingResult): The new rating result.
            
        Returns:
            bool: True if the task is finished, False otherwise
        """
        task.is_toxic = self.is_toxic(rating)
        task.num_attempts += 1
        if not self.conserve_memory:
            task.ratings.append(rating)
            
        return task.is_finished()

    def generate(
        self,
        prompts: Iterable[str],
        max_len: int = 50,
        batch_size: int = 10,
        **kwargs: Any,
    ) -> Iterator[SurvivalResult]:
        """
        Lazily processes a stream of prompts. For each prompt, generation and rating are performed
        until an acceptable output is produced or the maximum number of attempts is reached.
        Each finished prompt is yielded as a SurvivalResult.

        Args:
            prompts (Iterable[str]): An iterable of prompt strings.
            max_len (int, optional): Maximum length for generated text. Defaults to 50.
            batch_size (int, optional): Number of prompts to process concurrently. Defaults to 10.
            **kwargs: Additional keyword arguments passed to the generation backend.

        Yields:
            SurvivalResult: A finished result for a prompt.
        """
        prompt_iter = iter(prompts)
        active_tasks: List[SurvivalResult] = []

        def fill_tasks() -> None:
            # Fill active tasks up to the desired batch size.
            while len(active_tasks) < batch_size:
                try:
                    prompt = next(prompt_iter)
                except StopIteration:
                    break

                new_task = SurvivalResult(
                    id=self.current_task_id,
                    prompt=prompt,
                    max_attempts=self.max_attempts,
                    num_attempts=0,
                )

                active_tasks.append(new_task)
                self.current_task_id += 1

        fill_tasks()

        total = len(prompts) if hasattr(prompts, "__len__") else None
        with tqdm(total=total, desc="Processing Prompts") as pbar:

            while active_tasks:

                # Batch of tasks for the next iteration.
                next_tasks: List[SurvivalResult] = []

                # Generate output text for all active tasks.
                gen_results = self.generation_runner.generate_batch(
                    prompts=[task.prompt for task in active_tasks],
                    max_len=max_len,
                    **kwargs,
                )

                tasks_to_rate: List[SurvivalResult] = []
                texts_to_rate: List[str] = []

                for task, gen in zip(active_tasks, gen_results):
                    # if generation succeeded, prepare for rating
                    if gen.is_valid():
                        tasks_to_rate.append(task)
                        texts_to_rate.append(self.prepare_for_rating(gen))
                        continue

                    # if we're here then generation failed,
                    # update task with error and yield finished ones.    
                    if self.update_task(task, RatingResult(error=gen.error)):
                        yield task
                        pbar.update(1)
                    else:
                        next_tasks.append(task)
                    
                # Rate the generated texts.
                rate_results = self.rating_runner.rate_batch(texts_to_rate)

                # Update remaining tasks and yield finished ones.
                for task, rating in zip(tasks_to_rate, rate_results):
                    if self.update_task(task, rating):
                        yield task
                        pbar.update(1)
                    else:
                        next_tasks.append(task)

                active_tasks = next_tasks
                fill_tasks()
