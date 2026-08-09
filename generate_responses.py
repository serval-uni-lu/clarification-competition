import os
import fire
import json

import importlib.util
import inspect
import traceback

from tqdm import tqdm

from clarify.baselines.base import ClarificationAlgorithmBase

from clarify.env import ClarificationEnvironment, ClarificationConfiguration
from clarify.data import preprocess_benchmark
from clarify.utils import BatchParallelProcessor, BatchSequentialProcessor

from rich.console import Console
from rich.panel import Panel

console = Console()


def _load_clarification_algorithm(path_to_algorithm: str):
    """
    Parses and compiles the Python file at path_to_algorithm, executing it
    into its own module namespace, then returns the ClarificationAlgorithmBase
    subclass defined in that file.
    """
    if not os.path.exists(path_to_algorithm):
        raise FileNotFoundError(f"{path_to_algorithm} does not exist.")

    module_name = os.path.splitext(os.path.basename(path_to_algorithm))[0]
    spec = importlib.util.spec_from_file_location(module_name, path_to_algorithm)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # parses + compiles + runs the file

    candidates = [
        obj for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, ClarificationAlgorithmBase) and obj is not ClarificationAlgorithmBase
        and obj.__module__ == module_name  # exclude re-imported subclasses from elsewhere
    ]

    if not candidates:
        raise ValueError(
            f"No ClarificationAlgorithmBase subclass defined in {path_to_algorithm}."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Expected exactly one ClarificationAlgorithmBase subclass in "
            f"{path_to_algorithm}, found: {[c.__name__ for c in candidates]}"
        )

    return candidates[0]

# Hooks ---------------

def llm_console_hook(message: dict[str, str]):
    if message["role"] == "user":
        console.print(Panel(message["content"], title="Prompt", border_style="cyan"))
    if message["role"] == "assistant":
        console.print(Panel(message["content"], title="Model Response", border_style="purple"))


def clarification_console_hook(message: dict[str, str]):
    if message["role"] == "user":
        console.print(Panel(message["content"], title="Question", border_style="green"))
    if message["role"] == "assistant":
        console.print(Panel(message["content"], title="Clarification", border_style="orange3"))

# ---------------------


class SimulationFunction:

    def __init__(self, **config):
        self.config = config

    def __call__(self, environment_definition):
        environment = ClarificationEnvironment(
            self.config["environment_config"], environment_definition,
            llm_api_hook = llm_console_hook,
            clarification_api_hook = clarification_console_hook,
        )

        problem_definition = {
            "prompt": environment_definition["prompt"],
            "entry_point": environment_definition.get("entry_point", "unknown")
        }

        result = {"task_id": environment_definition["task_id"], 
                  "prompt": environment_definition["prompt"]}

        try:
            clarification_algorithm = self.config["clarification_algorithm"]
            algorithm_name = clarification_algorithm.__class__.__name__
            with console.status(f"Run {algorithm_name}..."):
                prompt_result = clarification_algorithm.run(
                    environment, problem_definition
                )

            result["prompt_result"] = prompt_result
            result["clarification_history"] = environment.history
            result["prompt_cost"] = environment.prompt_cost
            if environment.clarification_cost:
                result["clarification_cost"] = environment.clarification_cost

            return result

        except Exception as e:
            if self.config["fail_on_exception"]:
                raise

            traceback.print_exc()
            result["prompt_result"] = f"[EXCEPTION] {e}"
            return result


def main(
    clarify_py : str,
    dataset_path : str = "data/mbpp_test.jsonl",
    output_path : str = "data/mbpp_test_clarify_output.jsonl",
    batch_size : int = 1,
    max_workers : int = 1,
    max_clarification_turns : int = 1,
    language_model      : str = "gpt-4.1-mini",
    temperature : float = 0.7,
    clarification_model : str | None = None,
    fail_on_exception : bool = False,
    **kwargs
):
    batch_size = max(batch_size, max_workers)

    environment_config = ClarificationConfiguration(
        max_clarification_turns = max_clarification_turns,
        language_model = language_model,
        temperature = temperature,
        clarification_model = clarification_model
    )
    
    # Load clarification algorithm and use kwargs as config options
    clarification_algorithm = _load_clarification_algorithm(
        clarify_py
    )(kwargs)

    algorithm_name = clarification_algorithm.__class__.__name__
    print(f"Loaded `{algorithm_name}` clarification algorithm...")

    # Load data ----------------------
    with open(dataset_path, "r") as lines:
        benchmark = [json.loads(line) for line in lines]

    benchmark = preprocess_benchmark(benchmark)
    print(f"Loaded {len(benchmark)} instances...")

    simulation_function = SimulationFunction(
        clarification_algorithm = clarification_algorithm,
        environment_config = environment_config,
        fail_on_exception = fail_on_exception
    )

    def batched_iterator():
        current_batch = []
        for example in benchmark.values():
            current_batch.append(example)
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []
        
        if len(current_batch) > 0:
            yield current_batch

    if max_workers > 1:
        batched_worker = BatchParallelProcessor(simulation_function, max_workers = max_workers)
    else:
        batched_worker = BatchSequentialProcessor(simulation_function)

    prompt_cost, clarification_cost = 0.0, 0.0
    try:
        with open(output_path, "w") as o, tqdm(total = len(benchmark)) as pbar:
            for task_batch in batched_iterator():
                results = batched_worker(task_batch)
                for result in results:
                    o.write(json.dumps(result, default = str) + "\n")

                    prompt_cost += result.get("prompt_cost", 0.0) 
                    clarification_cost += result.get("clarification_cost", 0.0) 

                    pbar.set_description(f"{algorithm_name} | Prompt ($): {prompt_cost+clarification_cost:.4f}, Clarification ($): {clarification_cost:.4f}")
                    pbar.update(1)
        
    finally:
        if batched_worker: batched_worker.close()
    
    

if __name__ == "__main__":
    fire.Fire(main)