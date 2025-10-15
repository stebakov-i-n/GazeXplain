from clearml import Task, Dataset
import subprocess
import os
import json

task = Task.init()

print('Test task')

dataset = Dataset.get(dataset_name='GazeXplain_dataset', dataset_project='GazeXplain')
dataset_path = dataset.get_local_copy()
print(f'Dataset downloaded to: {dataset_path}')
print(os.listdir('.'))
print(os.listdir(dataset_path))

subprocess.run(
    ['bash', 'GazeXplain/bash/train.sh', dataset_path]
)

# with open('runs/COCO_TP_runX_baseline/history.json', 'r') as fp:
#     history = json.load(fp)

# with open('runs/COCO_TP_runX_baseline/hparams.json', 'r') as fp:
#     hparams = json.load(fp)

# for i in range(len(history['Train'])):
#     # Log a simple metric
#     task.logger.report_scalar(
#         title="loss",
#         series="Training",
#         value=history['Train'][i],
#         iteration=i
#     )

#     task.logger.report_scalar(
#         title="AUC",
#         series="Validation",
#         value=history['Val'][i],
#         iteration=i
#     )

# # Log a hyperparameter
# task.connect(hparams)

# task.upload_artifact('trained_model', artifact_object='runs/COCO_TP_runX_baseline/checkpoints/ckpt_best/model.safetensors')
# task.upload_artifact('evaluation', artifact_object='runs/COCO_TP_runX_baseline/validation/metric.json')
