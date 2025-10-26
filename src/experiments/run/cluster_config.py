"""Cluster-specific configuration for your setup."""

from dataclasses import dataclass


@dataclass
class ClusterConfig:
    """Configuration for cluster execution."""

    home_dir: str = "/home/users/cye"
    workspace_base: str = "/oak/stanford/projects/c4u/researchbench/workspace"
    singularity_image: str = "/oak/stanford/projects/c4u/researchbench/workspace/inspect_final_tool.sif"
    tmp_base: str = "/scratch/users/cye/tmp"
    slurm_log_dir: str = "/home/users/cye/slurm_logs"
    inspect_log_dir: str = "/home/users/cye/researchbench/logs"
    api_key_path: str = "/home/users/cye/researchbench/config.sh"

    n_parallel: int = 20
    cpu_partition: str = "kipac"
    gpu_partition: str = "owners"
    time_hours: int = 7
    cpus_per_task: int = 4
    mem_gb: int = 256


DEFAULT_CONFIG = ClusterConfig()