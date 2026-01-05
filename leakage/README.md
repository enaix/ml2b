## Data leakage analysis

### Installation

Clone the `leakage-analysis` tool:

`git clone --recursive https://github.com/malusamayo/leakage-analysis`

This project requires `python3.8` interpreter to parse the AST.

`cd leakage-analysis`

Follow the instructions provided in the repository to build the package:


`pip install -r requirements.txt`

Install the [souffle](https://souffle-lang.github.io/install) package

#### Build the `pyright` package:

`cd pyright`

`npm install`

`npm run build:cli:dev`

#### Install tools deps

`pip install pandas bs4`

### Processing

Go to the parent `leakage` directory. You may want to install commonly used packages like `scikit-learn` to import necessary type hints, otherwise the analyzer would not be able to process the code correctly

Recursively find the submission (`submission.py`) files, process them and write to the output directory:

`python3 submission_mgr.py --add-entrypoint INPUT_FOLDER OUTPUT_FOLDER`

Run the analyzer:

`cd leakage-analysis`

`python3 -m src.run -o OUTPUT_FOLDER_PATH` - Generate HTML analysis result for each submission

`cd ..`

`python3 analyze_results.py OUTPUT_FOLDER_PATH` - Generate `.csv` file with results

Results analysis is done in [leakages.ipynb](leakages.ipynb)

### Multi-core processing

You may speed up the processing by running multiple instances of the analyzer. Split the directory into N batches using `../make_subfolders.sh N OUTPUT_FOLDER_PATH`. Note that the remainder is added to the N+1-th subfolder

Execute the analyzer in parallel using `../run_parallel.sh "python3 -m src.run -o" OUTPUT_FOLDER_PATH`. Logs are written to `subfolder*.log`.

Move the resulting files into a singular directory: `mv OUTPUT_FOLDER_PATH/subfolder?/* FINAL_OUTPUT_FOLDER_PATH` (for N<9)

You may now generate the `.csv` file and analyze the results

