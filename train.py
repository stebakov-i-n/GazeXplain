from clearml import Task, Dataset
import subprocess
import os

task = Task.init()

print('Test task')

dataset = Dataset.get(dataset_name='GazeXplain_dataset', dataset_project='GazeXplain')
dataset_path = dataset.get_local_copy()
print(f'Dataset downloaded to: {dataset_path}')
print(os.listdir('.'))
print(os.listdir(dataset_path))

subprocess.run(
    ['bash', 'GazeXplain/bash/train.sh', dataset_path],
    capture_output=True,
    text=True,
    check=True
)

# # # Инициализация задачи ClearML
# # task = Task.init(
# #     project_name='GazeXplain',
# #     task_name='GazeXplain_Training',
# #     auto_connect_frameworks=True
# # )

# # Логируем сам sh-скрипт
# task.upload_artifact(name='training_script', artifact_object='GazeXplain/bash/train.sh')

# # Запускаем shell-скрипт
# try:
#     result = subprocess.run(
#         ['bash', 'GazeXplain/bash/train.sh'],
#         capture_output=True,
#         text=True,
#         check=True
#     )

#     task.upload_artifact('trained_model', artifact_object='runs/COCO_TP_runX_baseline/checkpoints/ckpt_best/model.safetensors')
#     task.upload_artifact('evaluation', artifact_object='runs/COCO_TP_runX_baseline/validation/metric.json')


        
# except subprocess.CalledProcessError as e:
#     task.get_logger().report_text(f"Script failed: {e}")
#     raise