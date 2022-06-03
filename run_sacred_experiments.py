import sys
import subprocess
import os
import json
import time
import signal
import argparse
import itertools
import datetime
from multiprocessing import Pool
from random import randint
from sacred.observers import MongoObserver

try:
    from tensorflow.python.client import device_lib

    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == "GPU"]

    tensorflow_available = True


except ImportError:
    tensorflow_available = False


def get_entry_or_default(label, config, default):
    try:
        return config[label]
    except KeyError:
        return default


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to an experiment config file (json)",
    )
    parser.add_argument(
        "--n_jobs", type=int, help="Number of jobs to launch in parallel", default=1
    )
    parser.add_argument(
        "--mult_gpus",
        action="store_true",
        help="Distribute tensorflow jobs over all available GPUs",
    )
    parser.add_argument(
        "--bsub",
        action="store_true",
        help="Use bsub to submit cluster jobs instead of multiprocessing "
        "(if set, --n_jobs, and --mult_gpus is ignored)",
    )
    parser.add_argument(
        "--bsub_n", type=int, help="Number of CPUs to use per bsub job", default=1
    )
    parser.add_argument(
        "--bsub_W", type=str, help="Timelimit to use per bsub job", default="23:59"
    )
    parser.add_argument(
        "--bsub_mem",
        type=int,
        help="Memory (MB) to use per bsub job (for each core)",
        default=4500,
    )
    parser.add_argument(
        "--bsub_gpus",
        type=int,
        help="Number of GPUs to use per bsub job",
        default=0,
    )
    parser.add_argument(
        "--drop_output",
        action="store_true",
        help="Redirect all stdout and stderr output to /dev/null",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    experiment_script = config["experiment_script"]
    experiment_label = get_entry_or_default("experiment_label", config, "no_label")
    N_seeds = get_entry_or_default("N_seeds", config, 1)
    del config["N_seeds"]
    del config["experiment_script"]

    config_list_entries = []

    for key in config.keys():
        if isinstance(config[key], list):
            config_list_entries.append([(key, val) for val in config[key]])
        else:
            config_list_entries.append([(key, config[key])])

    job_list = []

    if args.mult_gpus and tensorflow_available:
        devices = get_available_gpus()
        print("Distributing over tf devices:", devices)
        devices_iter = itertools.cycle(range(len(devices)))

    seeds = [randint(1, 100000) for _ in range(N_seeds)]
    for seed in seeds:
        for config_update in itertools.product(*config_list_entries):
            config_update_dict = dict(config_update)
            config_update_dict["seed"] = seed
            env = os.environ.copy()

            if args.mult_gpus and tensorflow_available:
                env["CUDA_VISIBLE_DEVICES"] = str(next(devices_iter))

            job_list.append((config_update_dict, env))

    # we use subprocesses for parallelization instead of python multiprocessing
    # because the former caused results to be missing in mongodb
    def run_experiment(job):
        config_updates, env = job
        arguments = ["{}={}".format(k, v) for k, v in config_updates.items()]
        command = [sys.executable, experiment_script, "with"] + arguments
        if args.drop_output:
            command += ["&>", "/dev/null"]
        experiment_label = config_updates["experiment_label"]

        if args.bsub:
            command_str = subprocess.list2cmdline(command)
            os.makedirs("jobs", exist_ok=True)
            os.makedirs(os.path.join("jobs", "output"), exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            label = f"{experiment_label}_{timestamp}"
            scriptfile = os.path.join(os.getcwd(), "jobs", f"{label}.sh")
            outfile = os.path.join(os.getcwd(), "jobs", "output", f"{label}.out")
            errfile = os.path.join(os.getcwd(), "jobs", "output", f"{label}.err")
            print(command_str)
            with open(scriptfile, "w") as f:
                f.write(command_str)
            bsub_command = (
                f"bsub -W {args.bsub_W} "
                f"-n {args.bsub_n} "
                f'-R "rusage[mem={args.bsub_mem}]" '
                f'-R "rusage[ngpus_excl_p={args.bsub_gpus}]" '
                f"-oo {outfile} "
                f"-eo {errfile} "
                f"-J {experiment_label} "
                f"{scriptfile}"
            )
            os.system(f"chmod +x {scriptfile}")
            os.system(bsub_command)
        else:
            print(" ".join(command))
            stderr = []
            with subprocess.Popen(
                command,
                env=env,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
            ) as p:
                for line in p.stderr:
                    stderr.append(line)
            returncode = p.wait()
            print("\n".join(stderr))

    t = time.time()
    n_jobs = len(job_list)
    print()
    method = "bsub" if args.bsub else "multiprocessing"
    if args.n_jobs == 1:
        response = "yes"
    else:
        response = input(f"Starting {n_jobs} jobs using {method}. OK? [yes/no] ")
    print()
    if response == "yes":
        if args.n_jobs == 1 or args.bsub:
            for job in job_list:
                run_experiment(job)
        else:
            with Pool(args.n_jobs + 1) as p:  # +1 for base process
                p.map(run_experiment, job_list)
        t = time.time() - t

        print("Done in {:.2f} seconds".format(t))
    else:
        print("No jobs started.")
