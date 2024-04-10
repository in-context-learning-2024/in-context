# Instructions to Set Up BRC Savio Cluster

## Getting Started with Savio

1. **Create a Savio Account and Request Access**
   - Visit [https://mybrc.brc.berkeley.edu](https://mybrc.brc.berkeley.edu) to make an account.
   - Sign the cluster user access agreement.
   - Request access to `fc_ocow` (Professor Sahai’s project group).

2. **Setup Google Authenticator**
   - To create one-time passwords for logging in to BRC, follow the instructions [here](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/setting-otp/).

## Accessing BRC via SSH
First, export your username (will need to do this in both BRC environment and local machine) so commands in this guide can work through copy and paste: 
```
export UNAME=<yourusername>
```

To access BRC, use the following SSH command:
```
ssh $UNAME@hpc.brc.berkeley.edu
```

- At the `Password:` prompt, enter your password (called Token Pin in the official documentation) followed immediately (without spaces) by the 6-digit one-time password from the Google Authenticator app.
  - **Example**: If your password is `ilovebrc` and the one-time code is `123456`, you would type `ilovebrc123456`.

## Transferring Singularity Containers

1. **Download Required Containers**
   - Libraries container: [Download Link](https://drive.google.com/file/d/1syZVEb0l5d75q7NDaQRyS7FJQCqSxlDq/view?usp=sharing)
   - CUDA container: [Download Link](https://drive.google.com/file/d/1gF3-sQoQewP6Dhv1jVxbYvqRbrLT2idS/view?usp=sharing)
   - **Note**: Store files other than code repositories and some models in the scratch directory (`/global/scratch/users/$UNAME/`).

2. **Upload Containers to Scratch Directory**
   - Use the `scp` command to transfer the downloaded containers to your scratch directory on BRC:
     ```
     scp -r ~/Downloads/overlay-50G-10M.ext3.gz $UNAME@dtn.brc.berkeley.edu:/global/scratch/users/$UNAME/singularity/
     scp -r ~/Downloads/cuda11.5-cudnn8-devel-ubuntu18.04.sif $UNAME@dtn.brc.berkeley.edu:/global/scratch/users/$UNAME/singularity/
     ```
   - Unzip the file in the target directory:
     ```
     gunzip /global/scratch/users/$UNAME/singularity/overlay-50G-10M.ext3.gz
     ```

## Setting Up the Environment

1. **Obtain GPU Access**
   - Use the `srun` command to allocate a GPU for setting up your conda environment:
     ```
     srun --account=fc_ocow --partition=savio4_gpu --cpus-per-task=4 --gres=gpu:A5000:1 --qos=a5k_gpu4_normal -t 3:00:00 --pty /bin/bash
     ```
   - For different GPU/partition options, check available associations with:
     ```
     sacctmgr show associations where account=fc_ocow format=Account,Partition,qos
     ```

2. **Start Singularity Container**
   - Set the scratch directory variable and execute the singularity container:
     ```
     export SCRATCH=/global/scratch/users/$UNAME
     singularity exec --userns --fakeroot --nv -B /usr/lib64 -B /var/lib/dcv-gl --overlay $SCRATCH/singularity/overlay-50G-10M.ext3:rw $SCRATCH/singularity/cuda11.5-cudnn8-devel-ubuntu18.04.sif /bin/bash
     ```

3. **Install Conda Environment**
   - Navigate to `/ext3` and install Miniconda:
     ```
     cd /ext3
     mkdir miniconda3
     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh
     bash miniconda3/miniconda.sh -b -u -p miniconda3
     rm -rf miniconda3/miniconda.sh
     miniconda3/bin/conda init bash
     ```
   - Use `conda env export > environment.yml` on your machine, adjust the prefix in `environment.yml`, and then create the conda environment on BRC with `conda env create -f environment.yml`.

## Running Experiments

1. **Generate Job Scripts**
   - Copy `generate_jobs.py` to your project directory.
   - Use the template command to generate jobs:
     ```
     python3 generate_jobs.py -j JOBS_PER_GPU --name PROJECT_NAME --conda_env_name CONDA_ENV_NAME --partition PARTITION --qos QOS --cpus_per_task CPUS_PER_TASK --gpu GPU --memory CPU_MEMORY PYTHON_COMMAND
     ```
   - Example usage to create multiple job combinations.

2. **Execute Experiments**
   - Navigate to the `sbatch` directory and run the shell script for your experiment:
     ```
     cd sbatch
     ./name.sh
     ```
   - Check log files in the `sbatch` folder if issues arise and try again as necessary.

This guide should help you get started with your projects on the BRC Savio Cluster. Good luck!


