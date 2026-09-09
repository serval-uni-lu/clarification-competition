import os
import json
from glob import glob

try:
    import evalplus

    from evalplus.data.mbpp import get_mbpp_plus, mbpp_deserialize_inputs
    from evalplus.data import get_human_eval_plus
except ImportError:
    evalplus = None


def _load_humaneval_datasets(base_path):
    for humaneval_path in glob(os.path.join(base_path, "humaneval", "*.jsonl")):
        with open(humaneval_path, "r") as lines:
            dataset = [json.loads(line) for line in lines]
        yield preprocess_benchmark(dataset)

def _load_mbpp_datasets(base_path):
    for mbpp_path in glob(os.path.join(base_path, "mbpp", "*.jsonl")):
        with open(mbpp_path, "r") as lines:
            dataset = [json.loads(line) for line in lines]
        yield preprocess_benchmark(dataset)

def _load_datasets(base_path, dataset_id):
    if dataset_id == "HumanEval":
        return _load_humaneval_datasets(base_path)
    if dataset_id == "Mbpp":
        return _load_mbpp_datasets(base_path)

    raise ValueError(f"Unknown dataset `{dataset_id}`")


def load_split(split_path):
    with open(split_path, "r") as lines:
        task_ids = set(line.strip() for line in lines)

    base_dir = split_path
    while "split" in base_dir:
        base_dir  = os.path.dirname(base_dir)

    datasets  = set(task_id.split("/", 1)[0] for task_id in task_ids)

    benchmark = []
    for dataset_id in datasets:
        for dataset in _load_datasets(base_dir, dataset_id):
            for example in dataset.values():
                if example["task_id"] in task_ids:
                    benchmark.append(example)

    return benchmark


def preprocess_benchmark(dataset):
    tasks = set(example["task_id"].split("/", 1)[0]
                    for example in dataset)
    
    assert len(tasks) == 1, f"Can only support one type of tasks, but got {tasks}"

    task_identifier = next(iter(tasks))

    if task_identifier == "Mbpp":
        return _preprocess_mbpp(dataset)

    if task_identifier == "HumanEval":
        return _preprocess_humaneval(dataset)

    raise ValueError(f"Unknown task `{task_identifier}`")


def _preprocess_mbpp(dataset):
    if not evalplus:
        raise ImportError("To support MBPP, you need to install the Python package `evalplus`.")

    dataset = _preprocess_references(dataset)
    
    mbpp = get_mbpp_plus()
    preprocessed_benchmark = {}
    for task in dataset:
        if not task["prompt"]:
            # Prompt is none existing; skip
            continue

        if task["task_id"] in mbpp:
            preprocessed_example = mbpp[task["task_id"]]
            preprocessed_example["reference_prompt"] = preprocessed_example["prompt"]
            preprocessed_example.update(task)
        elif task["task_id"] in ["Mbpp/617"]:
            preprocessed_example = task
            preprocessed_example["plus_input"] = []

        elif task["task_id"] in ["Mbpp/114", "Mbpp/490"]:
            preprocessed_example = task
            if "base_input" in task:
                preprocessed_example["base_input"] = [
                    [[tuple(lst) for lst in lst_lst] for lst_lst in inp] for inp in task["base_input"]
                ]
                preprocessed_example["plus_input"] = []

        elif task["task_id"] in ["Mbpp/642"]:
            preprocessed_example = task
            if "base_input" in task:
                preprocessed_example["base_input"] = [
                    [[[tuple(l) for l in lst] for lst in lst_lst] for lst_lst in inp] for inp in task["base_input"]
                ]
                preprocessed_example["plus_input"] = []

        else:
            preprocessed_example = task
            if "base_input" in task:
                preprocessed_example["base_input"] = mbpp_deserialize_inputs(
                    task["task_id"], task["base_input"]
                )
            if "plus_input" in task:
                preprocessed_example["plus_input"] = mbpp_deserialize_inputs(
                    task["task_id"], task["plus_input"]
                )
            else:
                preprocessed_example["plus_input"] = []

        if "atol" not in preprocessed_example:
            preprocessed_example["atol"] = 0

        preprocessed_example["prompt"] = preprocessed_example["prompt"].strip() + "\n"
        preprocessed_benchmark[task["task_id"]] = preprocessed_example
    
    return preprocessed_benchmark


def _preprocess_humaneval(benchmark):
    if not evalplus:
        raise ImportError("To support HumanEval, you need to install the Python package `evalplus`.")

    benchmark = _preprocess_references(benchmark)
    
    humaneval = get_human_eval_plus()

    preprocessed_benchmark = {}
    for task in benchmark:
        if not task["prompt"]:
            continue

        if task["task_id"] in humaneval:
            preprocessed_example = humaneval[task["task_id"]]
            preprocessed_example["reference_prompt"] = preprocessed_example["prompt"]
            preprocessed_example.update(task)
        else:
            raise ValueError(f"Unknown HumanEval task `{task['task_id']}`")

        preprocessed_benchmark[task["task_id"]] = preprocessed_example
    
    return preprocessed_benchmark


def _preprocess_references(benchmark):
    import base64, zlib

    for task in benchmark:
        if "reference_prompt" in task and task["reference_prompt"].startswith("REF_"):
            reference_prompt = task["reference_prompt"][len("REF_"):]
            reference_prompt = zlib.decompress(
                base64.b64decode(
                    reference_prompt.encode("ascii")
                )
            ).decode("utf-8")
            task["reference_prompt"] = reference_prompt

    return benchmark
