import wandb

api = wandb.Api()

entity = 'in-context-learning-2024'  
project = 'neurips'  

def get_gpu_type_from_tags(run):
    for tag in run.tags:
        if 'gpu' in tag.lower():
            return tag.split(':')[-1]
    return 'Unknown'  

gpu_hours_dict = {}

try:
    runs = api.runs(f"{entity}/{project}")
    for run in runs:
        try:
            runtime_seconds = run.summary.get('_runtime', 0)  
            gpu_type = get_gpu_type_from_tags(run)
            num_gpus = run.config.get('gpu', 1)  
            gpu_hours = (runtime_seconds / 3600) * num_gpus  

            if gpu_type in gpu_hours_dict:
                gpu_hours_dict[gpu_type] += gpu_hours
            else:
                gpu_hours_dict[gpu_type] = gpu_hours

            print(f"Run: {run.name}, GPU Type: {gpu_type}, GPU Hours: {gpu_hours:.2f}")
        except Exception as e:
            print(f"Error processing run {run.id}: {e}")

except Exception as e:
    print(f"Error fetching runs for project {project}: {e}")

print("Total GPU Hours by GPU Type:")
for gpu_type, hours in gpu_hours_dict.items():
    print(f"{gpu_type}: {hours:.2f} hours")
