"""Slurm job submission utilities using submitit."""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def get_slurm_executor(
    log_dir: str,
    partition: str = "owners",
    time_hours: int = 7,
    nodes: int = 1,
    cpus_per_task: int = 4,
    mem_gb: int = 128,
    job_name: str = "researchbench",
    array_parallelism: Optional[int] = None,
    enable_gpu: bool = False
):
    """Create submitit executor for Slurm.

    Args:
        log_dir: Directory for Slurm logs
        partition: Slurm partition
        time_hours: Time limit in hours
        nodes: Number of nodes
        cpus_per_task: CPUs per task
        mem_gb: Memory in GB
        job_name: Job name
        array_parallelism: Max parallel array jobs
        enable_gpu: Request GPU

    Returns:
        submitit AutoExecutor configured for Slurm
    """
    import submitit

    executor = submitit.AutoExecutor(folder=log_dir)

    slurm_kwargs = {
        "slurm_time": f"{time_hours}:00:00",
        "nodes": nodes,
        "slurm_ntasks_per_node": 1,
        "cpus_per_task": cpus_per_task,
        "slurm_mem": f"{mem_gb}GB",
        "slurm_job_name": job_name,
        "slurm_partition": partition
    }

    if array_parallelism:
        slurm_kwargs["slurm_array_parallelism"] = array_parallelism

    if enable_gpu:
        slurm_kwargs["slurm_gres"] = "gpu:1"

    executor.update_parameters(**slurm_kwargs)

    return executor


def submit_job(executor, func, *args, **kwargs):
    """Submit job to Slurm.

    Args:
        executor: submitit executor
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        submitit Job object
    """
    job = executor.submit(func, *args, **kwargs)
    logger.info(f"Submitted job {job.job_id}")
    return job
