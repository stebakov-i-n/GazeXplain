from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torchvision.transforms as T
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import PIL
import os
from os.path import join, isdir, isfile
import numpy as np
import argparse
import json


def image_data(dataset_path, device='cuda:0', overwrite=False):
    resize_dim = (384 * 2, 512 * 2)
    src_path = join(dataset_path, 'images/')
    target_path = join(dataset_path, '../image_features/')

    images_all = []

    with open(join(join(dataset_path, 'fixations/', 'coco_search18_fixations_TP_validation.json')), "r") as f:
        validation_fix = json.load(f)

    with open(join(join(dataset_path, 'fixations/', 'coco_search18_fixations_TP_test.json')), "r") as f:
        test_fix = json.load(f)

    with open(join(join(dataset_path, 'fixations/', 'coco_search18_fixations_TP_train.json')), "r") as f:
        train_fix = json.load(f)

    with open(join(join(dataset_path, 'processed/', 'Qwen3_30B_PROMPT_L.json')), "r") as f:
        resps = json.load(f)

    images_all += [i["name"] for i in test_fix]
    images_all += [i["name"] for i in validation_fix]
    images_all += [i["name"] for i in train_fix]

    model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    folders = [i for i in os.listdir(src_path) if isdir(join(src_path, i))]

    for folder in folders:
        if not (os.path.exists(target_path) and os.path.isdir(target_path)):
            os.mkdir(target_path)
        files = [i for i in os.listdir(join(src_path, folder)) if
                 isfile(join(src_path, folder, i)) and i.endswith('.jpg')]
        for f in files:
            if f in images_all:
                if overwrite == False and os.path.exists(join(target_path, f.replace('jpg', 'pth'))):
                    continue
                image = PIL.Image.open(join(src_path, folder, f)).resize(resize_dim)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {"type": "text", "text": resps[f + folder]},
                        ],
                    }
                ]

                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    padding="max_length",
                    max_length=1200,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )

                inputs = inputs.to(model.device)

                with torch.no_grad():
                    # Получаем скрытые состояния модели
                    outputs = model(
                        **inputs,
                        max_new_tokens=512,
                        output_hidden_states=True,
                        return_dict=True
                    )

                    # Последние скрытые состояния
                    last_hidden_states = outputs.hidden_states[-1][0].squeeze().detach().cpu()

                torch.save(last_hidden_states, join(target_path, f.replace('jpg', 'pth').replace('.pth', '_vlm.pth')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Gazeformer Feature Extractor Utils', add_help=False)
    parser.add_argument('--dataset_path', default='/repo/GazeXplain/GazeSearch/COCO/TP', type=str)
    parser.add_argument('--cuda', default=0, type=int)
    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.cuda))
    image_data(dataset_path=args.dataset_path, device=device, overwrite=True)
