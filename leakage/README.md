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

Go to the parent `leakage` directory

Recursively find the submission (`submission.py`) files, process them and write to the output directory:

`python3 submission_mgr.py --add-entrypoint INPUT_FOLDER OUTPUT_FOLDER`

Run the analyzer:

`cd leakage-analysis`

`python3 -m src.run -o OUTPUT_FOLDER_PATH` - Generate HTML analysis result for each submission

`cd ..`

`python3 analyze_results.py OUTPUT_FOLDER_PATH` - Generate `.csv` file with results
