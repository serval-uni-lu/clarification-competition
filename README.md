# The Clarification Competition


## The Clarification SDK
An SDK to implement clarification algorithms that can be judged within the clarification competition. We expect that every participant submits a single Python file `clarifier.py` which implements `ClarificationAlgorithmBase` (`clarify.ClarificationAlgorithmBase`). 

### Installation
We use `uv` to develop this project. Follow the steps to install the project:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# Optional: Install everything in a venv
uv venv --python 3.13
source .venv/bin/activate

# Sync the repository
uv sync
```

### Direct LLM
> Algorithm: An LLM is tasked to implement a given task specification. If the LLM does not provide an implementation, DirectLLM interprets the response as a clarifying question which is clarified.

A simplistic baseline for code generation with clarifications. 
```bash
python generate_responses.py clarify/baselines/direct.py [input file] [output file]
```

### ClarifyGPT-modern
> Algorithm: ClarifyGPT incrementally generates candidate implementation and clusters them with respect to the test behavior. Once at least two clusters are identified, ClarifyGPT produces clarifying questions to discriminate between them. After clarification, ClarifyGPT restarts the process with the clarified task specification. 

A modernized version of [ClarifyGPT](https://arxiv.org/abs/2310.10996) for clarification question generation. The modernized version implements incremental clustering, self-repair, and LLM-based test generation. 
```bash
python generate_responses.py clarify/baselines/clarifygpt.py [input file] [output file]
```
