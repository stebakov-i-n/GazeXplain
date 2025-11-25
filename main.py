from clearml import Task, Dataset
import subprocess
import os
import json

LOCAL = False
DATASET = 'CocoSearch'
UNSEEN_SET = 'set_1'
EXP_NAME = 'VLM_experiments_unseen_set_1'

def main():
    if not LOCAL:
        dataset = Dataset.get(dataset_name='GazeXplain_dataset', dataset_project='GazeXplain')
        dataset_path = dataset.get_local_copy()
        task = Task.init()
    else:
        dataset_path = '/repo'

    subprocess.run(
        ['bash', 'repo/bash/train.sh', dataset_path, DATASET, UNSEEN_SET, EXP_NAME]
    )

    subprocess.run(
        ['bash', 'repo/bash/test.sh', dataset_path, DATASET, EXP_NAME]
    )

    if not LOCAL:
        with open(f'runs/{EXP_NAME}/history.json', 'r') as fp:
            history = json.load(fp)

        with open(f'runs/{EXP_NAME}/hparams.json', 'r') as fp:
            hparams = json.load(fp)

        for i in range(len(history['Train'])):
            # Log a simple metric
            task.logger.report_scalar(
                title="loss",
                series="Training",
                value=history['Train'][i],
                iteration=i
            )

            task.logger.report_scalar(
                title="AUC",
                series="Validation",
                value=history['Val'][i],
                iteration=i
            )

        # Log a hyperparameter
        task.connect(hparams)

        task.upload_artifact('checkpoint_best', artifact_object='runs/COCO_TP_runX_baseline/checkpoints/ckpt_best')
        task.upload_artifact('checkpoint_spv', artifact_object='runs/COCO_TP_runX_baseline/checkpoints/ckpt_supervised_end')
        task.upload_artifact('history_record', artifact_object='runs/COCO_TP_runX_baseline/history_record.json')
        task.upload_artifact('hparams', artifact_object='runs/COCO_TP_runX_baseline/hparams.json')
        task.upload_artifact('validation', artifact_object='runs/COCO_TP_runX_baseline/validation')
        task.upload_artifact('test', artifact_object='runs/COCO_TP_runX_baseline/test')

if __name__ == "__main__":
    main()