accelerate launch --config_file repo/src/config.yaml --main_process_port 29700 repo/src/test_explanation_alignment.py --split test --test_batch 32 --dataset_dir "$1/$2/" --model_name "$3/"
