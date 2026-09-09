import os
import fire
import json


from tqdm import tqdm

from clarify.data import preprocess_benchmark, load_split
from clarify.utils import BatchParallelProcessor, BatchSequentialProcessor
from clarify.runtime import init_evalplus_evaluator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ---------------------

DEFAULT_DATASET_PATH = "data/mbpp_demo_test.jsonl"

DEFAULT_SPLITS = {
    "train" : "data/splits/train.txt",
    "val"   : "data/splits/validation.txt"
}


class EvaluationFunction:

    def __init__(self, **config):
        self.config = config

    def __call__(self, result):
        evaluator = result["evaluator"]
        del result["evaluator"]

        prompt_result = result["prompt_result"]
        if "```python" not in prompt_result:
            prompt_result = f"```python\n{prompt_result.strip()}\n```"

        # Execute
        with console.status(f"Run tests for {result['task_id']}"):
            success, status = evaluator.evaluate(
                None, prompt_result
            )

        if success:
            console.print(Panel(prompt_result, title=f"✅ Success ({result['task_id']})", border_style="green"))
        else:
            output = f"{prompt_result}\n\nTest Result:\n{status}"
            console.print(Panel(output, title=f"❌ Failure ({result["task_id"]})", border_style="red"))

        result["success"], result["test_result"] = success, status
        return result

# Statistics ---------

def _print_test_overview(results):
    clarify_map = {"3": "✅", "2": "🌀", "1": "☠️"}
    algorithm = None
    for result in results:
        algorithm = result.get("algorithm", None)

    if algorithm:
        table = Table(title=f"Test Results ({algorithm})")
    else:
        table = Table(title="Test Results")

    table.add_column("Task ID")
    table.add_column("Status", justify = "center")
    table.add_column("Clarify?", justify = "center")
    table.add_column("Output", no_wrap = True)

    for result in sorted(results, key = lambda x: x["task_id"]):
        output = result["test_result"]

        clarification = "❌"
        if result["clarification_history"]:
            clarification = max(e[1] for e in result["clarification_history"])
            clarification = clarify_map.get(clarification, clarification)

        table.add_row(
            result["task_id"], 
            "✅" if result["success"] else ("⏲️" if "timeout" in output else "❌"),
            clarification,
            output[:40] + "..." if len(output) > 40 else output
        )

    console.print(table)


def print_statistics(output_path):
    with open(output_path, "r") as lines:
        results = [json.loads(line) for line in lines]

    _print_test_overview(results)

    table = Table(title="Run Statistics")
    table.add_column("Metric")
    table.add_column("Results")

    total = len(results)
    table.add_row("Total", str(total))

    pass_at_1 = sum(r["success"] for r in results)
    table.add_row("Pass@1", f"{100 * pass_at_1 / total:2f}")

    clarification_rate = sum(len(r["clarification_history"]) > 0 for r in results)
    table.add_row("Clarification rate", f"{100 * pass_at_1 / total:2f}%")

    if clarification_rate > 0:
        high_quality_clarification = sum(all(e[1] == "3" for e in r["clarification_history"])
                                        for r in results if len(r["clarification_history"]) > 0)
        table.add_row("High quality clarification", f"{100 * high_quality_clarification / clarification_rate:2f}%")

        clarification_length = sum(sum(len(e[2]) for e in r["clarification_history"]) for r in results)
        table.add_row("Average clarification length", f"{clarification_length / clarification_rate:2f}")

    console.print(table)
    
# --------------------

def main(
    benchmark_path : str = "data/mbpp_demo_test.jsonl",
    generation_path : str = "data/mbpp_demo_test_clarify_output.jsonl",
    output_path : str = "data/mbpp_demo_test_clarify_output_results.jsonl",
    batch_size : int = 1,
    max_workers : int = 1,
    force_rerun : bool = False,
    split : str | None = None,
):
    if os.path.exists(output_path) and not force_rerun:
        print_statistics(output_path)
        exit(0)
    
    batch_size = max(batch_size, max_workers)

    # Load data ----------------------
    if split:
        if benchmark_path != DEFAULT_DATASET_PATH:
            print(f"WARNING: split '{split}' is overwritting your benchmark path.")
        split = DEFAULT_SPLITS.get(split, split)
        benchmark = load_split(split)
    else:
        with open(benchmark_path, "r") as lines:
            benchmark = [json.loads(line) for line in lines]

        benchmark = list(preprocess_benchmark(benchmark).values())
    
    evaluator = init_evalplus_evaluator(benchmark)
    print(f"Loaded {len(benchmark)} instances...")

    with open(generation_path, "r") as lines:
        results = [json.loads(line) for line in lines]

    for result in results:
        result["evaluator"] = evaluator.partial(result)

    print(f"Loaded {len(results)} result instances...")

    evaluation_function = EvaluationFunction()

    def batched_iterator():
        current_batch = []
        for example in results:
            current_batch.append(example)
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []
        
        if len(current_batch) > 0:
            yield current_batch

    if max_workers > 1:
        batched_worker = BatchParallelProcessor(evaluation_function, max_workers = max_workers)
    else:
        batched_worker = BatchSequentialProcessor(evaluation_function)

    num_success, total = 0, 0
    try:
        with open(output_path, "w") as o, tqdm(total = len(benchmark)) as pbar:
            for task_batch in batched_iterator():
                results = batched_worker(task_batch)
                for result in results:
                    o.write(json.dumps(result, default = str) + "\n")

                    num_success += 1.0 if result.get("success", False) else 0.0
                    total += 1

                    pbar.set_description(f"Pass@1: {100 * num_success / total:.2f}")
                    pbar.update(1)
        
    finally:
        if batched_worker: batched_worker.close()
        print_statistics(output_path)
    

if __name__ == "__main__":
    fire.Fire(main)