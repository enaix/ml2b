Infrastructure code for the machine learning LLM benchmark

## Requirements

- kaggle cli

- Docker with Compose plugin

## Python

You may run `test_submission.sh` to automatically download and run the dataset

### Setup

Clear the submission folder after each run

Make sure that submission folder is a Python module:

`echo "" > python/submission/__init__.py`

Copy `competitions.json` to the container:

`cp competitions.json python/data/`

Put the respecting dataset to the `python/data/${COMPETITION_ID}` directory and the submission file to `python/submission/code.py`. You may download the dataset using kaggle cli


### Executing

`export COMPETITION_ID=abc`

`export BENCH_LANG=English`

Execute the benchmark and put results to `python/submission/results.txt`:

`docker compose run bench_python    # to enable non-blocking mode add -d`
