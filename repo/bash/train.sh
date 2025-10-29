# The name of this experiment.
apt-get install -y libgl1
apt-get install -y default-jre

export PATH="$PATH:/root/.clearml/venvs-builds/3.10/bin"

MODEL_NAME="$3"

# Save logs and models under snap/; make backup.
output=runs/${MODEL_NAME}
mkdir -p $output/src
mkdir -p $output/bash
apt install -y rsync
rsync -av  repo/src/* $output/src/
cp $0 $output/bash/run.bash

python3 repo/src/preprocess/COCOSearch18/feature_extractor.py --dataset_path "$1/$2/COCO/TP"

# TORCH_DISTRIBUTED_DEBUG=DETAIL
TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --config_file repo/src/config.yaml --main_process_port 29600 repo/src/train_explanation_alignment.py --project_dir runs/${MODEL_NAME} \
  --project_name ExplanationScanpath --checkpoint_every 1 --checkpoint_every_rl 1 --epochs 1 --start_rl_epoch 8  --batch 32 --test_batch 48 --dataset_dir "$1/$2/" \
   --search_dataset $2 --use_responses
