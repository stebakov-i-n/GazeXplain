"""Evaluation"""
from __future__ import print_function

import json
import os
import time
import torch
import numpy as np
from torch import Tensor

from collections import OrderedDict

import scipy.stats
import sys
import copy


import tempfile
from json import encoder

from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from lib.dataset.dataset import UnifiedScanpath
from lib.evaluation.evaluator import Evaluator

from lib.models.models import Transformer
from lib.models.gazeformer_explanation_alignment import gazeformer

from accelerate.utils import tqdm

encoder.FLOAT_REPR = lambda o: format(o, '.3f')



def get_prediction(accelerator, model, data_loader, opt):
    """
    Get prediction
    """
    # Initialize the Evaluator
    evaluator = Evaluator(opt)

    blip_tokenizer = BertTokenizerFast.from_pretrained("Salesforce/blip-image-captioning-base")

    repeat_num = opt.eval_repeat_num
    prediction_scanpaths = []
    gt_scanpaths = []
    image_sizes = []
    evaluation_scores = []
    idx_batch = []
    dataset_idxes = []
    all_generated_ids = []
    with tqdm(total=len(data_loader)) as pbar:
        for i_batch, batch in enumerate(data_loader):
            batch = {k: v if not torch.is_tensor(v) else v.to(accelerator.device) for k, v in batch.items()}

            with torch.no_grad():
                prediction, scanpath_prediction, generated_ids, sampling_prediction, action_masks, duration_masks = model(batch, opt.eval_repeat_num)

            # by default we repeat once for inference
            scanpath_prediction = scanpath_prediction[:, 0]
            generated_ids = generated_ids[:, :, 0]
            if i_batch != len(data_loader) - 1:
                # not the last iter, do not need to drop, the evaluation can be done in different GPU

                # transform the torch type prediction as json
                predictions = evaluator.transform(scanpath_prediction)
                targets = evaluator.transform(batch["gt_fixation"])


                # evaluation on the prediction and ground truth
                scores = evaluator.measure(targets, predictions, batch)

                # gather for metric
                scanpath_prediction = accelerator.gather_for_metrics(scanpath_prediction)
                gt_fixation = accelerator.gather_for_metrics(batch["gt_fixation"])
                image_size = accelerator.gather_for_metrics(batch["image_size"])
                scores = accelerator.gather_for_metrics(scores)
                idx = accelerator.gather_for_metrics(batch["idx"])
                dataset_idx = accelerator.gather_for_metrics(batch["dataset_idx"])
                generated_ids = accelerator.gather_for_metrics(generated_ids)

                if accelerator.is_main_process:
                    prediction_scanpaths.extend(scanpath_prediction)
                    gt_scanpaths.extend(gt_fixation)
                    image_sizes.extend(image_size)
                    evaluation_scores.extend(scores)
                    idx_batch.extend(idx)
                    dataset_idxes.extend(dataset_idx)
                    all_generated_ids.extend(generated_ids)


            else:
                # the last iter, need to drop, the evaluation cannot be done in different GPU
                # gather for metric
                scanpath_prediction = accelerator.gather_for_metrics(scanpath_prediction)
                gt_fixation = accelerator.gather_for_metrics(batch["gt_fixation"])
                image_size = accelerator.gather_for_metrics(batch["image_size"])
                idx = accelerator.gather_for_metrics(batch["idx"])
                dataset_idx = accelerator.gather_for_metrics(batch["dataset_idx"])
                generated_ids = accelerator.gather_for_metrics(generated_ids)
                fixation_info = [data_loader.dataset.fixations[_] for _ in idx]
                batch["image_size"] = image_size
                batch["dataset_idx"] = dataset_idx
                batch["fixation_info"] = fixation_info


                # transform the torch type prediction as json
                predictions = evaluator.transform(scanpath_prediction)
                targets = evaluator.transform(gt_fixation)

                # evaluation on the prediction and ground truth
                scores = evaluator.measure(targets, predictions, batch)

                if accelerator.is_main_process:
                    prediction_scanpaths.extend(scanpath_prediction)
                    gt_scanpaths.extend(gt_fixation)
                    image_sizes.extend(image_size)
                    evaluation_scores.extend(scores)
                    idx_batch.extend(idx)
                    dataset_idxes.extend(dataset_idx)
                    all_generated_ids.extend(generated_ids)

            # accelerator.print(scanpath_prediction.shape)
            # accelerator.print(gt_fixation.shape)
            # accelerator.print(image_size.shape)
            # accelerator.print(len(scores))
            # accelerator.print(len(fixation_info))
            # accelerator.print(len(idx_batch))

            pbar.update()


    # transform the gather scanpath prediction to JSON format file
    if accelerator.is_main_process:
        scanpath_dataset = ["AiR-D", "OSIE", "COCO-TP", "COCO-TA", "COCO-FV"]
        json_prediction_scanpaths = evaluator.transform(prediction_scanpaths)
        idx_list = torch.stack(idx_batch).cpu().numpy().tolist()
        json_gt_scanpaths = data_loader.dataset.fixations
        json_targets = evaluator.transform(gt_scanpaths)

        # collect for saliency evaluation
        prediction_fixation_dict = {}
        gt_fixation_dict = {}
        for iter in range(len(json_gt_scanpaths)):
            cur_dataset = scanpath_dataset[dataset_idxes[iter]]
            if cur_dataset == "AiR-D":
                key = "{}-{}".format(cur_dataset, json_gt_scanpaths[iter]["question_id"])
            elif cur_dataset == "OSIE":
                key = "{}-{}".format(cur_dataset, json_gt_scanpaths[iter]["name"])
            elif cur_dataset == "COCO-TP":
                key = "{}-{}-{}".format(cur_dataset, json_gt_scanpaths[iter]["task"], json_gt_scanpaths[iter]["name"])
            elif cur_dataset == "COCO-TA":
                key = "{}-{}-{}".format(cur_dataset, json_gt_scanpaths[iter]["task"], json_gt_scanpaths[iter]["name"])
            else:
                raise "Invalid dataset"
            prediction_fixation_dict.setdefault(key, []).append(json_prediction_scanpaths[iter])
            gt_fixation_dict.setdefault(key, []).append(json_targets[iter])


        # collect for caption explanation
        explanation_res = {}
        for iter in range(len(json_gt_scanpaths)):
            cur_dataset = scanpath_dataset[dataset_idxes[iter]]
            if cur_dataset == "AiR-D":
                key = "{}-{}".format(json_gt_scanpaths[iter]["question_id"], json_gt_scanpaths[iter]["subject"])
            elif cur_dataset == "OSIE":
                key = "{}-{}".format(json_gt_scanpaths[iter]["name"][:-4], json_gt_scanpaths[iter]["subject"])
            elif cur_dataset == "COCO-TP":
                key = "TP-{}-{}-{}".format(json_gt_scanpaths[iter]["task"], json_gt_scanpaths[iter]["name"][:-4], json_gt_scanpaths[iter]["subject"])
            elif cur_dataset == "COCO-TA":
                key = "TA-{}-{}-{}".format(json_gt_scanpaths[iter]["task"], json_gt_scanpaths[iter]["name"][:-4], json_gt_scanpaths[iter]["subject"])
            else:
                raise "Invalid dataset"


            # explanation_res.setdefault(key, []).append(
            #     {
            #         "caption": " ".join(blip_tokenizer.batch_decode(all_generated_ids[iter], skip_special_tokens=True)).strip(),
            #     }
            # )

            tmp_caption = blip_tokenizer.batch_decode(all_generated_ids[iter], skip_special_tokens=True)
            # tmp_caption = ["there is " + _ if len(_) > 0 else _ for _ in tmp_caption]
            explanation_res.setdefault(key, []).append(
                {
                    "caption": " ".join(tmp_caption).strip(),
                }
            )

        all_saliency_evals = evaluator.eval_saliency(prediction_fixation_dict, gt_fixation_dict)

        # evaluation on captioning
        gts = data_loader.dataset.explanation_gts
        res = explanation_res
        evaluator.explanation_evaluation(gts, res)



        for idx in idx_list:
            json_prediction_scanpath = json_prediction_scanpaths[idx]
            json_gt_scanpath = json_gt_scanpaths[idx]
            json_prediction_scanpath["dataset"] = json_gt_scanpath["dataset"]
            json_prediction_scanpath["evaluation_scores"] = evaluation_scores[idx]

            cur_dataset = scanpath_dataset[dataset_idxes[idx]]
            if cur_dataset == "AiR-D":
                key = "{}-{}".format(json_gt_scanpaths[idx]["question_id"],
                                     json_gt_scanpaths[idx]["subject"])
            elif cur_dataset == "OSIE":
                key = "{}-{}".format(json_gt_scanpaths[idx]["name"][:-4],
                                     json_gt_scanpaths[idx]["subject"])
            elif cur_dataset == "COCO-TP":
                key = "TP-{}-{}-{}".format(json_gt_scanpaths[idx]["task"],
                                           json_gt_scanpaths[idx]["name"][:-4],
                                           json_gt_scanpaths[idx]["subject"])
            elif cur_dataset == "COCO-TA":
                key = "TA-{}-{}-{}".format(json_gt_scanpaths[idx]["task"],
                                           json_gt_scanpaths[idx]["name"][:-4],
                                           json_gt_scanpaths[idx]["subject"])
            else:
                raise "Invalid dataset"

            json_prediction_scanpath["explanation"] = explanation_res[key]
            json_prediction_scanpath["gt_explanation"] = gts[key]

            # log the caption evaluation
            evaluation_scores[idx]['Bleu_4'] = evaluator.scanpath_eval.evalImgs[idx]['Bleu_4']
            evaluation_scores[idx]['METEOR'] = evaluator.scanpath_eval.evalImgs[idx]['METEOR']
            evaluation_scores[idx]['ROUGE_L'] = evaluator.scanpath_eval.evalImgs[idx]['ROUGE_L']
            evaluation_scores[idx]['CIDEr'] = evaluator.scanpath_eval.evalImgs[idx]['CIDEr']
            evaluation_scores[idx]['CIDEr-R'] = evaluator.scanpath_eval.evalImgs[idx]['CIDEr-R']


        dataset_evaluation_scores_dict = {}
        for iter, evaluation_score in enumerate(evaluation_scores):
            dataset_evaluation_scores_dict.setdefault(scanpath_dataset[dataset_idxes[iter]], []).append(
                evaluation_score)

        dataset_gather_scores_dict = {}
        for dataset_name, evaluation_score in dataset_evaluation_scores_dict.items():
            if len(evaluation_score) > 0:
                if dataset_name in ["AiR-D", "OSIE"]:
                    saliency_eval = np.array( [v for k, v in all_saliency_evals.items() if dataset_name in k])
                    gather_score = {
                        "scanmatch_score": np.array([_["scanmatch_score"] for _ in evaluation_score]),
                        "multimatch_score": np.array([_["multimatch_score"] for _ in evaluation_score]),
                        "sed_score": np.array([_["sed_score"] for _ in evaluation_score]),
                        "stde_score": np.array([_["stde_score"] for _ in evaluation_score]),
                        "SS_score": np.array([_["SS_score"] for _ in evaluation_score]),
                        "CC": saliency_eval[:, 0],
                        "AUC": saliency_eval[:, 1],
                        "NSS": saliency_eval[:, 2],
                        "sAUC": saliency_eval[:, 3],
                        "KLD": saliency_eval[:, 4],
                        "SIM": saliency_eval[:, 5],
                        "Bleu_4": np.array([_["Bleu_4"] for _ in evaluation_score]),
                        "METEOR": np.array([_["METEOR"] for _ in evaluation_score]),
                        "ROUGE_L": np.array([_["ROUGE_L"] for _ in evaluation_score]),
                        "CIDEr": np.array([_["CIDEr"] for _ in evaluation_score]),
                        "CIDEr-R": np.array([_["CIDEr-R"] for _ in evaluation_score]),
                    }
                else:
                    saliency_eval = np.array([v for k, v in all_saliency_evals.items() if dataset_name in k])
                    gather_score = {
                        "scanmatch_score": np.array([_["scanmatch_score"] for _ in evaluation_score]),
                        "multimatch_score": np.array([_["multimatch_score"] for _ in evaluation_score]),
                        "sed_score": np.array([_["sed_score"] for _ in evaluation_score]),
                        "stde_score": np.array([_["stde_score"] for _ in evaluation_score]),
                        "SS_score": np.array([_["SS_score"] for _ in evaluation_score]),
                        "SSS_score": np.array([_["SSS_score"] for _ in evaluation_score if not np.isnan(sum(_["SSS_score"]))]),
                        "CC": saliency_eval[:, 0],
                        "AUC": saliency_eval[:, 1],
                        "NSS": saliency_eval[:, 2],
                        "sAUC": saliency_eval[:, 3],
                        "KLD": saliency_eval[:, 4],
                        "SIM": saliency_eval[:, 5],
                        "Bleu_4": np.array([_["Bleu_4"] for _ in evaluation_score]),
                        "METEOR": np.array([_["METEOR"] for _ in evaluation_score]),
                        "ROUGE_L": np.array([_["ROUGE_L"] for _ in evaluation_score]),
                        "CIDEr": np.array([_["CIDEr"] for _ in evaluation_score]),
                        "CIDEr-R": np.array([_["CIDEr-R"] for _ in evaluation_score]),
                    }
                dataset_gather_scores_dict[dataset_name] = gather_score

        dataset_cur_metrics_dict = {}
        for dataset_name, gather_score in dataset_gather_scores_dict.items():
            if dataset_name in ["AiR-D", "OSIE"]:
                cur_metric = {
                    "metrics/SM without Dur": gather_score["scanmatch_score"][:, 0].mean(),
                    "metrics/SM with Dur": gather_score["scanmatch_score"][:, 1].mean(),
                    "metrics/MM Vector": gather_score["multimatch_score"][:, 0].mean(),
                    "metrics/MM Direction": gather_score["multimatch_score"][:, 1].mean(),
                    "metrics/MM Length": gather_score["multimatch_score"][:, 2].mean(),
                    "metrics/MM Position": gather_score["multimatch_score"][:, 3].mean(),
                    "metrics/MM Duration": gather_score["multimatch_score"][:, 4].mean(),
                    "metrics/MM": gather_score["multimatch_score"].mean(),
                    "metrics/SED": gather_score["sed_score"].mean(),
                    "metrics/STDE": gather_score["stde_score"].mean(),
                    "metrics/SS without Dur": gather_score["SS_score"][:, 0].mean(),
                    "metrics/SS with Dur": gather_score["SS_score"][:, 1].mean(),
                    "metrics/CC": gather_score["CC"].mean(),
                    "metrics/AUC": gather_score["AUC"].mean(),
                    "metrics/NSS": gather_score["NSS"].mean(),
                    "metrics/sAUC": gather_score["sAUC"].mean(),
                    "metrics/KLD": gather_score["KLD"].mean(),
                    "metrics/SIM": gather_score["SIM"].mean(),
                    "metrics/Bleu_4": gather_score["Bleu_4"].mean(),
                    "metrics/METEOR": gather_score["METEOR"].mean(),
                    "metrics/ROUGE_L": gather_score["ROUGE_L"].mean(),
                    "metrics/CIDEr": gather_score["CIDEr"].mean(),
                    "metrics/CIDEr-R": gather_score["CIDEr-R"].mean(),
                }
            else:
                cur_metric = {
                    "metrics/SM without Dur": gather_score["scanmatch_score"][:, 0].mean(),
                    "metrics/SM with Dur": gather_score["scanmatch_score"][:, 1].mean(),
                    "metrics/MM Vector": gather_score["multimatch_score"][:, 0].mean(),
                    "metrics/MM Direction": gather_score["multimatch_score"][:, 1].mean(),
                    "metrics/MM Length": gather_score["multimatch_score"][:, 2].mean(),
                    "metrics/MM Position": gather_score["multimatch_score"][:, 3].mean(),
                    "metrics/MM Duration": gather_score["multimatch_score"][:, 4].mean(),
                    "metrics/MM": gather_score["multimatch_score"].mean(),
                    "metrics/SED": gather_score["sed_score"].mean(),
                    "metrics/STDE": gather_score["stde_score"].mean(),
                    "metrics/SS without Dur": gather_score["SS_score"][:, 0].mean(),
                    "metrics/SS with Dur": gather_score["SS_score"][:, 1].mean(),
                    "metrics/SSS without Dur": gather_score["SSS_score"][:, 0].mean(),
                    "metrics/SSS with Dur": gather_score["SSS_score"][:, 1].mean(),
                    "metrics/CC": gather_score["CC"].mean(),
                    "metrics/AUC": gather_score["AUC"].mean(),
                    "metrics/NSS": gather_score["NSS"].mean(),
                    "metrics/sAUC": gather_score["sAUC"].mean(),
                    "metrics/KLD": gather_score["KLD"].mean(),
                    "metrics/SIM": gather_score["SIM"].mean(),
                    "metrics/Bleu_4": gather_score["Bleu_4"].mean(),
                    "metrics/METEOR": gather_score["METEOR"].mean(),
                    "metrics/ROUGE_L": gather_score["ROUGE_L"].mean(),
                    "metrics/CIDEr": gather_score["CIDEr"].mean(),
                    "metrics/CIDEr-R": gather_score["CIDEr-R"].mean(),
                }
            dataset_cur_metrics_dict[dataset_name] = cur_metric

        # for all the dataset
        saliency_eval = np.array(list(all_saliency_evals.values()))
        gather_scores = {
            "scanmatch_score": np.array([_["scanmatch_score"] for _ in evaluation_scores]),
            "multimatch_score": np.array([_["multimatch_score"] for _ in evaluation_scores]),
            "sed_score": np.array([_["sed_score"] for _ in evaluation_scores]),
            "stde_score": np.array([_["stde_score"] for _ in evaluation_scores]),
            "SS_score": np.array([_["SS_score"] for _ in evaluation_scores]),
            "SSS_score": np.array([_["SSS_score"] for _ in evaluation_scores if "SSS_score" in _ and not np.isnan(sum(_["SSS_score"]))]),
            "CC": saliency_eval[:, 0],
            "AUC": saliency_eval[:, 1],
            "NSS": saliency_eval[:, 2],
            "sAUC": saliency_eval[:, 3],
            "KLD": saliency_eval[:, 4],
            "SIM": saliency_eval[:, 5],
            "Bleu_4": np.array([_["Bleu_4"] for _ in evaluation_scores]),
            "METEOR": np.array([_["METEOR"] for _ in evaluation_scores]),
            "ROUGE_L": np.array([_["ROUGE_L"] for _ in evaluation_scores]),
            "CIDEr": np.array([_["CIDEr"] for _ in evaluation_scores]),
            "CIDEr-R": np.array([_["CIDEr-R"] for _ in evaluation_scores]),
        }

        cur_metrics = {
            "metrics/SM without Dur": gather_scores["scanmatch_score"][:, 0].mean(),
            "metrics/SM with Dur": gather_scores["scanmatch_score"][:, 1].mean(),
            "metrics/MM Vector": gather_scores["multimatch_score"][:, 0].mean(),
            "metrics/MM Direction": gather_scores["multimatch_score"][:, 1].mean(),
            "metrics/MM Length": gather_scores["multimatch_score"][:, 2].mean(),
            "metrics/MM Position": gather_scores["multimatch_score"][:, 3].mean(),
            "metrics/MM Duration": gather_scores["multimatch_score"][:, 4].mean(),
            "metrics/MM": gather_scores["multimatch_score"].mean(),
            "metrics/SED": gather_scores["sed_score"].mean(),
            "metrics/STDE": gather_scores["stde_score"].mean(),
            "metrics/SS without Dur": gather_scores["SS_score"][:, 0].mean(),
            "metrics/SS with Dur": gather_scores["SS_score"][:, 1].mean(),
            "metrics/SSS without Dur": gather_scores["SSS_score"][:, 0].mean() if gather_scores["SSS_score"].shape[0] != 0 else np.nan,
            "metrics/SSS with Dur": gather_scores["SSS_score"][:, 1].mean() if gather_scores["SSS_score"].shape[0] != 0 else np.nan,
            "metrics/CC": gather_scores["CC"].mean(),
            "metrics/AUC": gather_scores["AUC"].mean(),
            "metrics/NSS": gather_scores["NSS"].mean(),
            "metrics/sAUC": gather_scores["sAUC"].mean(),
            "metrics/KLD": gather_scores["KLD"].mean(),
            "metrics/SIM": gather_scores["SIM"].mean(),
            "metrics/Bleu_4": gather_scores["Bleu_4"].mean(),
            "metrics/METEOR": gather_scores["METEOR"].mean(),
            "metrics/ROUGE_L": gather_scores["ROUGE_L"].mean(),
            "metrics/CIDEr": gather_scores["CIDEr"].mean(),
            "metrics/CIDEr-R": gather_scores["CIDEr-R"].mean(),
        }

        dataset_cur_metrics_dict["all"] = cur_metrics

        return json_prediction_scanpaths, dataset_cur_metrics_dict

    else:
        return None, None

def eval(accelerator, model_path, opt, split='dev', save_path=None):
    """
    Evaluate a trained model on either dev or test.
    """
    eval_dataset = UnifiedScanpath(split=split, opt=opt)

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=4,
        collate_fn=eval_dataset.collate_func,
        drop_last=False
    )

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    transformer = Transformer(args=opt)
    model = gazeformer(transformer=transformer, args=opt)

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)


    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)


    # Load cbest heckpoint .
    # Infer iteration number through file name (it's hacky but very simple), so don't rename
    # saved checkpoints if you intend to continue training.
    accelerator.print(f"Load from best checkpoint: {model_path}")
    accelerator.load_state(model_path, strict=False)

    model.eval()

    predict_results, dataset_cur_metrics_dict = get_prediction(accelerator, model, eval_dataloader, opt)

    if save_path and accelerator.is_main_process:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, "metric.json"), "w") as f:
            json.dump(dataset_cur_metrics_dict, f, indent=2)
        with open(os.path.join(save_path, "predictions.json"), "w") as f:
            json.dump(predict_results, f, indent=2)

    if accelerator.is_main_process:
        for dataset, cur_metrics in dataset_cur_metrics_dict.items():
            accelerator.print("-" * 40)
            accelerator.print("{:30}".format(dataset))
            for key, value in cur_metrics.items():
                accelerator.print("{:30}: {:.3f}".format(key, value))
        accelerator.print("-" * 40)







