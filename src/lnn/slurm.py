import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

SBATCH_DEFAULT_ARGS = [
    "--partion=single",
    "--time=06:00:00",
    "--ntasks-per-node=1",
    "--cpus-per-task=64",
    "--mem=64gb",
]

SACCT_TIMESTAMP_FORMAT = "%Y-%m-%d"


def job_is_finished(
    job_id: Optional[int] = None, job_name: Optional[str] = None
) -> bool:
    now = datetime.now()
    last_month = now.replace(month=now.month - 1)
    start_timestamp = last_month.strftime(SACCT_TIMESTAMP_FORMAT)
    end_timestamp = last_month.strftime(SACCT_TIMESTAMP_FORMAT)
    sacct_args = ["sacct", "-P", "-S", start_timestamp, "-E", end_timestamp]
    sacct_output = subprocess.run(
        [" ".join(sacct_args)], shell=True, capture_output=True, check=True
    ).stdout.decode("utf-8")
    joblist = sacct_output.split("\n")[1:]
    for job in joblist:
        if not job.strip():
            continue
        entries = job.split("|")
        id = entries[0]
        name = entries[1]
        state = entries[-2]
        id = id.split(".")[0]
        if job_id is not None:
            if str(job_id) == id and state.strip() == "COMPLETED":
                return True
        else:
            assert job_name is not None, "Provide either job_name or job_id"
            if name.strip().lower() == job_name.strip().lower():
                return True
    return False


def get_jobs() -> dict[str, str | int]:
    squeue_output = subprocess.run(
        ["squeue", "--nohead", "--format", "%i %j %.10M %L %T"],
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    jobs = []
    for line in squeue_output.split("\n"):
        if not line.strip():
            continue
        id, name, run_time, time_left, state = line.split()
        id = int(id)
        jobs.append(
            {
                "id": id,
                "name": name,
                "run_time": run_time,
                "time_left": time_left,
                "state": state,
            }
        )
    return jobs


def get_current_timestamp():
    return datetime.now().strftime("%Y:%m:%d-%H:%M:%S")


def path_to_string(p: Path) -> str:
    return str(p.resolve())


def sbatch(
    script_path: str,
    sbatch_args: Optional[list[str]] = None,
    env_vars: Optional[dict[str, str]] = None,
    verbose: bool = False,
) -> subprocess.Popen:
    if sbatch_args is None:
        sbatch_args = SBATCH_DEFAULT_ARGS
    # Sanitize batch args
    sbatch_args = shlex.split(" ".join(sbatch_args))
    # Convert path into absolute path
    script_path = str(Path(script_path).expanduser().absolute())
    complete_args = ["sbatch"] + sbatch_args + [script_path]
    # Prepare environment variables
    env = os.environ.copy()
    if env_vars is not None:
        env |= env_vars
    if verbose:
        msg = f"Scheduling job using the following configuration: {complete_args}"
        msg += f"\n Using the following extra environment variables: {env_vars}"
        print(msg)
    process = subprocess.Popen(
        [" ".join(complete_args)],
        env=env,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process


def slurm_guardian(watch_config: list[dict], every: int = 30):
    while True:
        jobs = get_jobs()
        for entry in watch_config:
            job_name = entry["name"]
            is_scheduled = False
            for job in jobs:
                if job["name"] == job_name:
                    is_scheduled = True
                    print(
                        f"Found job '{job_name}'! Current status is '{job['state']}'."
                    )
                    break
            if not is_scheduled:
                is_already_finished = job_is_finished(job_name=job_name)
                force = entry.get("force", False)
                if is_already_finished and not force:
                    print(
                        f"Job '{job_name}'is not scheduled but seems to be finished..."
                        " Skipping resubmission! Set force=True in watch config to force resubmission."
                    )
                    continue
                print(f"Did not found job '{job_name}'! Scheduling it now.")
                sbatch_prc = sbatch(**entry["sbatch_kwargs"], verbose=True)
                stdout, stderr = sbatch_prc.communicate()
                print(stdout)
                print()
                print(stderr)

        time.sleep(every)


@click.command()
@click.argument(
    "watch_config",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
)
@click.option(
    "-e",
    "--every",
    default=30,
    help="Check job status every N seconds. Defaults to 30 seconds.",
)
def sguardian(watch_config, every):
    """Watches over slurm jobs every N seconds.

    If the job cannot be found, it is (re-)submitted.

    Specify jobs to watch and how to submit them via a watch-config JSON file:

    \b
    [
        {
            "name": "slurm-job-that-should-keep-running.sh",
            "sbatch_kwargs": {
                "script_path": "~/slurm-scripts/slurm-job-that-should-keep-running.sh",
                "sbatch_args": [
                    "--partition=single",
                    "--gres='gpu:4'",
                    "--cpus-per-task=64",
                    "--mem=64gb",
                    "--time=120:00:00",
                    "--ntasks-per-node=1",
                ],
                "env_vars": {
                    "HF_DATASETS_CACHE": "/store/user/hf_cache"
                },
            },
        },
        ...
    ]
    Usage:
        $ sguardian --every 20 my-watch-config.json
    """
    watch_config = json.loads(Path(watch_config).read_text())
    slurm_guardian(watch_config=watch_config, every=every)


@click.command()
@click.argument(
    "script-dir", default=".", type=click.Path(file_okay=False, exists=True)
)
@click.argument("script-pattern", default="*", type=str)
@click.option("-r", "--regex", is_flag=True, help="Use regex as script-pattern")
@click.option("-d", "--dry-run", is_flag=True, help="Run without submitting")
@click.option("-v", "--verbose", is_flag=True, help="Print out helpful messages")
@click.option(
    "-n",
    "--no-dump",
    is_flag=True,
    help="Disable dumping filenames of scripts that failed to submit",
)
def sbatch_submit(script_dir, script_pattern, regex, dry_run, verbose, no_dump):
    script_dir = Path(script_dir)
    if verbose:
        click.echo(
            f"Searching for slurm-scripts with pattern {script_pattern} in {script_dir.resolve()}"
        )
    if not regex:
        if verbose:
            click.echo("Searching for scripts using glob mode")
        scripts = list(script_dir.glob(script_pattern))
    else:
        if verbose:
            click.echo("Searching for scripts using regex mode")
        scripts = [
            s for s in script_dir.glob("*") if re.match(rf"{script_pattern}", s.name)
        ]
    if verbose or dry_run:
        click.echo(f"Found {len(scripts)}")
    if dry_run:
        click.echo("Dry run: Would submit the following scripts")
        click.echo("\n".join([path_to_string(s) for s in scripts]))
    else:
        submit_successes, submit_failures = 0, 0
        failed_scripts = []
        for script in scripts:
            script_path = path_to_string(script)
            if not dry_run:
                if verbose:
                    click.echo(f"Submitting: {script_path}")
                try:
                    subprocess.run(["sbatch", script_path], check=True)
                    submit_successes += 1
                except subprocess.CalledProcessError:
                    submit_failures += 1
                    failed_scripts.append(script_path)
        if verbose:
            click.echo(f"Successfully submitted {submit_successes} scripts.")
        if submit_failures:
            click.echo(f"Failed to submit {submit_failures} scripts.")
            if not no_dump:
                log_filename = (
                    Path().cwd() / f"sbatch-submit-log-{get_current_timestamp()}"
                )
                log_filename.write_text("\n".join(failed_scripts))
                click.echo(f"Wrote a detailed list of filenames to {log_filename}")
