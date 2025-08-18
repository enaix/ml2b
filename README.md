Infrastructure code for the machine learning LLM benchmark

## Python

### Setup

Clear the submission folder after each run

Make sure that submission folder is a Python module:

`echo "" > python/submission/__init__.py`

Put the respecting dataset to the `python/data` directory and the submission file to `python/submission/??.py`

### Executing

`export BENCH_NAME=abc`

Execute the benchmark and put results to `python/submission/results.txt`:

`docker compose run bench_python    # to enable non-blocking mode add -d`
