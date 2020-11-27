# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from model_service.pytorch_model_service import PTServingBaseService

import time
from metric.metrics_manager import MetricsManager
import log
from io import BytesIO
import base64
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000000000
logger = log.getLogger(__name__)

import cv2, sys

sys.path.append('.')
from mmseg.apis import inference_segmentor, init_segmentor


class ImageClassificationService(PTServingBaseService):
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path

        config_file = '/home/mind/model/deeplabv3plus512x512.py'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = init_segmentor(config_file, self.model_path, device=self.device)

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                # img = cv2.imread(file_content, 1)
                preprocessed_data[k] = img
        return preprocessed_data

    def _inference(self, data):
        img = data["input_img"]
        data = img
        target_l = 1024
        x, y, c = data.shape
        label = np.zeros((x, y))
        x_num = (x // target_l + 1) if x % target_l else x // target_l
        y_num = (y // target_l + 1) if y % target_l else y // target_l
        for i in range(x_num):
            for j in range(y_num):
                x_s, x_e = i * target_l, (i + 1) * target_l
                y_s, y_e = j * target_l, (j + 1) * target_l
                img = data[x_s:x_e, y_s:y_e, :]
                if np.mean(img)==0:
                    out_l=np.ones((img.shape[0],img.shape[1]))
                else:
                    out_l = inference_segmentor(self.model, img)[0]
                label[x_s:x_e, y_s:y_e] = out_l.astype(np.int8)
        _label = label.astype(np.int8).tolist()
        _len, __len = len(_label), len(_label[0])
        o_stack = []
        for _ in _label:
            out_s = {"s": [], "e": []}
            j = 0
            while j < __len:
                if _[j] == 0:
                    out_s["s"].append(str(j))
                    while j < __len and _[j] == 0: j += 1
                    out_s["e"].append(str(j))
                j += 1
            o_stack.append(out_s)
        result = {"result": o_stack}
        return result

    def _postprocess(self, data):
        return data

    def inference(self, data):
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')
        if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)
        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000
        logger.info('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)
        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
        if self.model_name + '_LatencyInference' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)
        # Update overall latency metric
        if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)
        logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        data['latency_time'] = pre_time_in_ms + infer_in_ms + post_time_in_ms
        return data