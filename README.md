# CSC2541-Compute-Optimal-TTS

Basic python virtual environment (you can alternatively use conda/micromamba etc).

```bash
uv venv --python 3.10
source .venv/bin/activate
```

## Verifier Models
Install dependencies with uv, make sure you have cuda-12.4
```bash
uv sync --extra build
uv sync --extra build --extra compile
```

If the above fails, revert to the openr installation.