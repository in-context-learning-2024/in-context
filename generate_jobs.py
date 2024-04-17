import os
import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-j", default=4, type=int)
parser.add_argument("--name", type=str)
parser.add_argument("--conda_env_name", type=str)
parser.add_argument("--partition", default="savio4_gpu", type=str)
parser.add_argument("--qos", default="a5k_gpu4_normal", type=str)
parser.add_argument("--cpus_per_task", default=4, type=int)
parser.add_argument("--gpu", default="A5000", type=str)
parser.add_argument("--memory", default=24, type=int)

args, unknown = parser.parse_known_args()

name = args.name
conda_env_name = args.conda_env_name

print(unknown)


def parse(args):
    prefix = ""
    for index in range(len(args)):
        prefix += " "
        arg = args[index]
        i = arg.find("=")
        if i == -1:
            content = arg
        else:
            prefix += arg[: i + 1]
            content = arg[i + 1 :]

        if "," in content:
            elements = content.split(",")
            for r in parse(args[index + 1 :]):
                for element in elements:
                    yield prefix + element + r
            return
        else:
            prefix += content
    yield prefix


python_command_list = list(parse(unknown))
print("python_command_list", python_command_list)

num_jobs = len(python_command_list)

num_arr = (num_jobs - 1) // args.j + 1

print("\n".join(python_command_list))

path = os.getcwd()

d_str = "\n ".join(
    [
        "[{}]='{}'".format(i + 1, f'singularity exec --userns --fakeroot --nv --writable-tmpfs -B /usr/lib64 -B /var/lib/dcv-gl --overlay $SCRATCH/singularity/overlay-50G-10M.ext3:ro $SCRATCH/singularity/cuda11.5-cudnn8-devel-ubuntu18.04.sif /bin/bash -c " source ~/.bashrc && conda activate {conda_env_name} && cd {path} && MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false {command[1:]}"')
        for i, command in enumerate(python_command_list)
    ]
)

sbatch_str = f"""#!/bin/bash

#SBATCH --job-name=in-context-learning
#SBATCH --open-mode=append
#SBATCH --output=logs/out/%x_%j.txt
#SBATCH --error=logs/err/%x_%j.txt
#SBATCH --time=24:00:00
#SBATCH --mem={args.memory}G
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --gres=gpu:{args.gpu}:1
#SBATCH --account=fc_ocow
#SBATCH --partition={args.partition}
#SBATCH --qos={args.qos}
#SBATCH --array=1-{num_arr}
#SBATCH -o icl-%A_%a.out
#SBATCH -e icl-%A_%a.err

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N={args.j}
JOB_N={num_jobs}

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))

module load gnu-parallel

declare -a commands=(
 {d_str}
)

parallel --delay 20 --linebuffer -j {args.j} {{1}} ::: \"${{commands[@]:$COM_ID_S:$PARALLEL_N}}\"
"""

filename = f"sbatch/{name}.sh"
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, "w") as f:
    f.write(sbatch_str)
