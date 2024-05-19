"""

Before running:

'pip install wandb'
'wandb login [YOUR API KEY]'

If loading llama or gpt2 modification-free fails, then comment out the line where vocab size is set in models/transformers.py

"""


import wandb
import os

api = wandb.Api()

entity = 'in-context-learning-2024'
project = 'neurips'

save_dir = 'checkpoints'
os.makedirs(save_dir, exist_ok=True)

def get_user_inputs():
    print("Enter the run names, each on a new line. Enter a blank line to finish:")
    run_names = []
    while True:
        line = input().strip()
        if line == "":
            break
        run_names.append(line)
    
    checkpoint_numbers = input("Enter the checkpoint numbers separated by spaces: ").strip().split()
    return run_names, [cp.strip() for cp in checkpoint_numbers if cp.strip()]

specified_runs, specified_checkpoint_numbers = get_user_inputs()

def get_run_id_by_name(run_name):
    runs = api.runs(f"{entity}/{project}")
    for run in runs:
        if run_name in run.name:
            return run.id
    return None

for run_name in specified_runs:
    run_id = get_run_id_by_name(run_name)
    if run_id is None:
        print(f"Run with name '{run_name}' not found.")
        exit(1)
    
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        print(f"Fetching files for run: {run.name} ({run.id})")
        files = run.files()
        checkpoints_downloaded = False
        for file in files:
            if file.name.startswith("models/"):
                checkpoint_name = file.name.split('/')[-1]
                checkpoint_number = checkpoint_name.split('_')[-1].split('.')[0]
                if checkpoint_number in specified_checkpoint_numbers:
                    file_path = os.path.join(save_dir, run_name, "models", checkpoint_name)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    file.download(root=os.path.dirname(file_path), replace=True)
                    print(f"Downloaded {file.name} to {file_path}")
                    checkpoints_downloaded = True
        if not checkpoints_downloaded:
            print(f"No specified checkpoints found for run {run_name}")

    except Exception as e:
        print(f"Error downloading files for run {run_name}: {e}")

print("Download complete.")
