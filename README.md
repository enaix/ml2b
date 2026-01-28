# ML2B

This repository provides the implementation accompanying the paper *“MULTI-LINGUAL ML BENCHMARK FOR AUTOML”*.  
It includes the code for dataset construction, the evaluation framework, and the agents assessed within this benchmark.

## Usage

### Requirements

We use [`uv`](https://github.com/astral-sh/uv) for environment management.  
Install `uv` once, then run `uv sync` (or `uv pip install -r requirements.txt`) inside the project to create the virtual environment.

### Prepare Environment

1. Install dependencies:
   ```bash
   uv sync
   ```
2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
3. Build the agent runtime:
   ```bash
   python run.py build-runtime -i aide --agent-dir agents/aide
   ```
   *(If you use another agent, keep the same file structure and command. See `python run.py build-runtime --help` for details.)*
4. Download and prepare the dataset:
   ```bash
   python run.py prepare-data huggingface    # preferred way, you may delete the cache by passing the --remove-cache option
   # OR
   python run.py prepare-data gdrive
   ```
   *(The dataset can also be downloaded manually from the [hugginface hub](https://huggingface.co/datasets/enaix/ml2b) and placing the `data`,`tasks` directories into `competitions`)*
   
   *(If you wish to download from GDrive and encounter an error with `gdown`, manually download the data from [Google Drive](https://drive.google.com/drive/folders/18QoNa3vjdJouI4bAW6wmGbJQCrWprxyf). You would also need to download task descriptions manually)*

After these steps, you should see the following structure:

```
.
├── run.py
└── competitions/
    ├── data/
    ├── competitions.json
    └── tasks/
```

### Running the Benchmark

1. Configure agent parameters in the corresponding directory (e.g. `agents/aide/config.yaml`).  
   Make sure environment variables such as `$OPENAI_API_KEY` are exported in your shell.

2. Run the benchmark (see `python run.py bench --help` for more options):
   ```bash
   python run.py bench -i aide -w 3 --agent-dir agents/aide --seed 42 --args-variant extended --code-variant extended
   ```

### Documentation

General documentation can be found in [docs](docs/)
