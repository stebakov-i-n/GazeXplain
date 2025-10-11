import math
import multiprocessing

import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

import os
import json

import numpy as np
import scipy.stats

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import LoggerType
from accelerate.utils import tqdm
from accelerate import DistributedDataParallelKwargs
from transformers import AutoTokenizer, BertTokenizerFast

from lib.dataset.dataset import UnifiedScanpath
from lib.evaluation.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from lib.models.loss import CrossEntropyLoss, MLPLogNormalDistribution, LogAction, LogDuration, AlignmentLoss
from lib.scst.cider.cider import Cider
from lib.scst.ciderR.ciderR import CiderR
from lib.scst.tokenizer import tokenizer
from opts import parse_opt

from lib.utils.checkpointing import CheckpointManager
from lib.utils.recording import RecordManager
from lib.evaluation.evaluator import Evaluator

import lib.models.models
import lib.models.gazeformer_explanation_alignment

from lib.models.sample.sampling import Sampling