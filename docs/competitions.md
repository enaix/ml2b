# Competitions and graders

## Competition structure

Competitions are defined in `competitions/competitions.json`. Its files are processed by [splitters](python/splitters.py), which produce train and validation splits. [Loaders](loaders/) read the splits into dataframes. [Graders](python/grade_functions.py) analyze the results produced by the code.

### Basic fields

Each competition defines the following fields:

- `metric`: Metric used by the default grader. Defined in [`METRICS`](python/grade_functions.py) and [`METRICS_EXTRA`](python/grade_functions.py) variables
- `target_col`: Column containing the target variable, handled in splitters and loaders
- `direction`: Either `minimize` or `maximize`, defines if the score should be minimized or maximized
- `cv_folds`: Number of cross-validation folds to use
- `output_hint`: Description of the output format 

### Advanced fields

- `grader`: Competition grader, defined in [`GRADERS`](python/grade_functions.py) variable. `default` is used if not set
- `split_strategy`: Competition splitter, defined in [`DATA_SPLITTERS`](python/splitters.py). `csv` is used by default
- `load_strategy`: Competition loader, defined in [`DATA_LOADERS`](loaders/__init__.py). `default` is used if not set
- `stratified_split`: Use StratifiedKFold split, false by default
- `exclude_cols`: Columns to exclude from train and test splits, used by the default grader, splitter and loader. Passed as extra data to the grader

### Defining extra files

Competitions with several files may be defined with `files` and `file_mapping` fields. Each file is defined as the following:

`'files': {'name': {'type': 'data', 'required': true, 'extensions': ['...']}}`

`'file_mapping': {'name': {'filename': '...', 'type': 'data', 'required': true}}`

- `name`: Internal file name
- `type`: TODO describe file type field
- `required`: bool
- `extensions`: List of file extensions
- `filename`: File name in competition directory

