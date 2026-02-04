# Manual grading

In some cases grader may be executed manually for a particular submission file, instead of executing the whole benchmark pipeline. Competition data is generated using the benchmark pipeline, while the grader container is executed using `docker compose` with the provided configuration. Note that the script is going to build the separate [grading container](../python/Dockerfile) with the default [`requirements.txt`](../python/requirements.txt).

Install the base enviroment and execute `grade_manual.py` script. It has the following positional arguments:

- `competition_id`: ID of the competition to run
- `lang`: Competition language
- `code_path`: Path to the Python submission file. Submission file is copied to the newly created folder

Optional arguments:

- `-m/--mode`: Execution mode: mono (monolithic) or modular (default: modular)
- `-e/--extended_schema`: Use extended schema for submission code (**y**/n)
- `-f/--folds`: Set custom number of folds, overrides value from `competitions.json`
- `-r/--rebuild`: Rebuild the grader container 
- `-s/--seed`: Set RNG seed (default: 42)

The output is written to `python/submission/submission_{competition_id}-{lang}-python-only_code`
