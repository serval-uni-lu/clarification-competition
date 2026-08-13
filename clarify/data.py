
try:
    import evalplus

    from evalplus.data.mbpp import get_mbpp_plus, mbpp_deserialize_inputs
    from evalplus.data import get_human_eval_plus
except ImportError:
    evalplus = None


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
    
    mbpp = get_mbpp_plus()
    preprocessed_benchmark = {}
    for task in dataset:
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
    
    humaneval = get_human_eval_plus()

    preprocessed_benchmark = {}
    for task in benchmark:
        if task["task_id"] in humaneval:
            preprocessed_example = humaneval[task["task_id"]]
            preprocessed_example["reference_prompt"] = preprocessed_example["prompt"]
            preprocessed_example.update(task)
        else:
            raise ValueError(f"Unknown HumanEval task `{task['task_id']}`")

        preprocessed_benchmark[task["task_id"]] = preprocessed_example
    
    return preprocessed_benchmark