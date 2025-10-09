# The name of this experiment.
DATASET_NAME='COCO_TP'
MODEL_NAME='runX_baseline'

# Save logs and models under snap/; make backup.
output=runs/${DATASET_NAME}_${MODEL_NAME}
mkdir -p $output/src
mkdir -p $output/bash
rsync -av  GazeXplain/src/* $output/src/
cp $0 $output/bash/run.bash

# TORCH_DISTRIBUTED_DEBUG=DETAIL
TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --config_file GazeXplain/src/config.yaml --main_process_port 29600 GazeXplain/src/train_explanation_alignment.py --project_dir runs/${DATASET_NAME}_${MODEL_NAME} \
  --project_name ExplanationScanpath --checkpoint_every 1 --checkpoint_every_rl 1 --epochs 2 --start_rl_epoch 1  --batch 16 --test_batch 32
