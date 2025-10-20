# ResearchBench Cluster Execution

This directory contains scripts for running ResearchBench evaluations on your cluster with Singularity and Slurm.

## Setup

1. **Configure your cluster settings** in `cluster_config.py`:
   - Paths to workspace, Singularity image, logs
   - Slurm parameters (partition, resources)

2. **Prepare experiment configs** in `experiment_configs/`:
   - `experiment_configs/base/` - Expert papers (base mode)
   - `experiment_configs/syw/` - ShowYourWork papers (base mode)

## Workflow

### 1. Setup Paper Workspaces

Download data and install dependencies for papers:

```bash
# Setup all showyourwork papers
python setup_workspace.py --filter_source showyourwork

# Setup specific papers
python setup_workspace.py --papers gw_nsbh astm3 bayes_cal

# Only download data (no dependency installation)
python setup_workspace.py --papers gw_nsbh --data_only

# Only install dependencies (no data download)
python setup_workspace.py --papers gw_nsbh --deps_only
```

### 2. Launch Evaluations

Submit evaluation jobs to Slurm:

```bash
# Launch full batch with config
python launch.py --config experiment_configs/base/o4_mini_config.json

# Override run name
python launch.py --config experiment_configs/base/o4_mini_config.json --run_name my_experiment

# Run single paper
python launch.py --config experiment_configs/base/o4_mini_config.json --paper_id gw_nsbh
```

### 3. Monitor Jobs

```bash
# Check Slurm queue
squeue -u $USER

# View job output
cat /home/users/cye/slurm_logs/RUN_NAME/JOBID_0_log.out

# View evaluation logs
inspect view --log-dir /home/users/cye/researchbench/logs/RUN_NAME/logs
```

## Configuration Files

Experiment config JSON format:

```json
{
    "MODEL": "openai/o4-mini",
    "RUN_NAME": "o4-mini-base",
    "TASK_TYPES": ["numeric"],
    "FILTERS": {"source": "expert"},
    "MASKING": true,
    "MODE": "base",
    "MESSAGE_LIMIT": 10000,
    "TOKEN_LIMIT": 5000000,
    "EXECUTION_TIMEOUT": 14400,
    "TIME_LIMIT": 21600,
    "ATTEMPTS": 1,
    "CACHE": true,
    "INCLUDE_WORKSPACE": true
}
```

**Note:** All cluster-specific settings (paths, Slurm params, etc.) are now in `cluster_config.py`, not in the experiment configs.

## Directory Structure

```
experiments/run/
├── cluster_config.py       # Your cluster-specific configuration
├── launch.py               # Launch evaluations on cluster
├── setup_workspace.py      # Setup paper workspaces
├── experiment_configs/     # Experiment configurations
│   ├── base/               # Expert papers
│   └── syw/                # ShowYourWork papers
└── README.md               # This file
```

## Notes

- Jobs are automatically assigned to CPU or GPU partitions based on paper requirements
- Workspaces are read-only during evaluation to prevent data corruption
- All evaluation outputs are written to temporary directories
- Overlays allow per-paper package installations without conflicts
