# Dataset

This specification describes how to change, customize, and add new competitions to the benchmark.

## Agent Split

To run an agent, we use the `competitions.json` file, which specifies competition settings for each task.

### Configuration Fields

* **metric** - The metric name to use for evaluation. We expose the following standard metrics:
  * `roc_auc_score`
  * `accuracy_score`
  * `f1_score`
  * `log_loss`
  * `mean_absolute_error`
  * `mean_squared_error`
  * `precision_score`
  * `recall_score`
  * `matthews_corrcoef`
  * `balanced_accuracy_score`
  * `root_mean_squared_error`
  * `r2_score`
  * `mean_squared_log_error`
  * `fbeta_score`
  
  You may also implement custom metrics. For examples, see [grade_functions.py](../python/grade_functions.py).

* **target_col** - The target column name in the training data.

* **files** (Optional) - A dictionary describing the data files to be used. Example:
```json
  {
    "train": {
      "type": "data",
      "required": true,
      "extensions": [".csv"]
    }
  }
```

* **file_mapping** (Optional) - Used by the splitter to split data. Example:
```json
  {
    "train": {
      "filename": "train_labels.csv",
      "type": "data",
      "required": true
    }
  }
```

* **grader** (Optional) - Selects the grading mechanism. For standard metrics, the default grader is used automatically.
  
You can also customize data splitting and loading. For examples, see [splitters.py](../python/splitters.py) for splitting logic and [loaders](../loaders/) for custom loading implementations.

## Data Structure

We use a split structure for storing tasks and data.

### Tasks Directory

The `tasks` directory should contain a `<Language>.csv` file with the following required fields:

* **comp_name** - The competition name (human-readable).
* **competition** - A pythonic key used for running the competition, referenced as the competition header in `competitions.json`.
* **data_card** - A description of the training data.
* **description** - The task description used to build the agent prompt.

### Data Directory

The `data` directory should contain folders named according to the `competition` field in the `<Language>.csv` file. Each folder should include the corresponding `train.csv` data file.

### Example Structure
```
competitions/
├── data/
│   ├── competition_key_1/
│   │   └── train.csv
│   └── competition_key_2/
│       └── train.csv
├── tasks/
│   ├── English.csv
│   └── Russian.csv
└── competitions.json
```