# ML2B

This repository provides the implementation accompanying the paper *"MULTI-LINGUAL ML BENCHMARK FOR AUTOML"*.  
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
   python ml2b.py build-runtime -i aide --agent-dir agents/aide

   # For ARM platforms use:
   python ml2b.py build-runtime -i react --agent-dir agents/react --platform "linux/arm64"
```
   *(If you use another agent, maintain the same file structure and command pattern.)*
   
   For proxy settings, see `python ml2b.py build-runtime --help` for details.

4. Download and prepare the dataset:
```bash
   python ml2b.py prepare-data
```
   *(The dataset can also be downloaded manually from [Hugging Face Hub](https://huggingface.co/datasets/enaix/ml2b) by placing the `data` and `tasks` directories into `competitions`.)*

   For customization details, see [dataset implementation documentation](/competitions/README.md).

5. Unpack the competition files:
```bash
   cd competitions/data
   chmod +x ./unpack_files.sh
   ./unpack_files.sh
```
   This will extract `.tar.gz` archives with competition files

After completing the preparation steps, you should see the following folder structure:
```
.
├── ml2b.py
└── competitions/
    ├── data/
    ├── competitions.json
    └── tasks/
```

### Running the Benchmark

1. Configure agent parameters in the corresponding directory (e.g., `agents/aide/config.yaml`).  

   Ensure necessary environment variables such as `OPENAI_API_KEY` are exported in your shell.

2. Configure Docker runtime limitations in [runtime_config.json](/environments/runtime/runtime_config.json).

   **Optional:** You may change proxy settings for the validation container in [squid.conf](/environments/runtime/squid.conf).

3. Run the benchmark (see `python ml2b.py bench --help` for more options):
```bash
   python ml2b.py bench -i aide -w 3 --agent-dir agents/aide --seed 42 --args-variant extended --code-variant extended
```

### Documentation

General documentation can be found in [docs](docs/)
