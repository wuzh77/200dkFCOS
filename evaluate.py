#!/usr/bin/env python
# coding=utf-8
# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
import stat
import argparse
import json
from sys import argv
import cv2
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import numpy as np


def loadtxtmethod(filename):
    data = np.loadtxt(filename, dtype=np.float64, delimiter=' ')
    return data


def run_coco_eval(coco_gt_obj, image_id_list, dt_file_path):
    annotation_type = 'bbox'
    print('Running test for {} results.'.format(annotation_type))
    coco_dt = coco_gt_obj.loadRes(dt_file_path)
    coco_eval = COCOeval(coco_gt_obj, coco_dt, annotation_type)
    coco_eval.params.imgIds = image_id_list
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    pre_file = argv[1]  # 预测的结果
    detect_file = argv[2]  # 存放评估结果
    FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    MODES = stat.S_IWUSR | stat.S_IRUSR
    ANNOTATIONFILE = './dataset/annotations/instances_val2017.json'
    coco_gt = COCO(ANNOTATIONFILE)
    image_ids = coco_gt.getImgIds()
    coco_result = []
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    for image_idx, image_id in enumerate(image_ids):
        image_info = coco_gt.loadImgs(image_id)[0]
        pre_txt = image_info['file_name']
        pre_image_txt = pre_file + '/' + pre_txt.replace('.jpg', '.txt')
        print(pre_image_txt)
        pre_info = loadtxtmethod(pre_image_txt)  # (X_0,Y_0,W,H) score category
        pre_info.tolist()

        if pre_info.size > 0:
            if (pre_info.size > 6):
                for item in pre_info:
                    image_result = {
                        'image_id': image_id,
                        'category_id': cats[int(item[5])]['id'],
                        'score': float(item[4]),
                        'bbox': [item[0], item[1], item[2], item[3]]
                    }
                    coco_result.append(image_result)
            else:
                image_result = {
                    'image_id': image_id,
                    'category_id': cats[int(item[5])]['id'],
                    'score': float(pre_info[4]),
                    'bbox':
                    [pre_info[0], pre_info[1], pre_info[2], pre_info[3]]
                }
                coco_result.append(image_result)
        else:
            image_result = {
                'image_id': image_id,
                'category_id': 1,
                'score': float(0),
                'bbox': [0, 0, 0, 0]
            }
            coco_result.append(image_result)
    if os.path.exists(detect_file):
        os.remove(detect_file)
    with os.fdopen(os.open(detect_file, FLAGS, MODES), 'w') as f:
        json.dump(coco_result, f, indent=4)
    run_coco_eval(coco_gt, image_ids, detect_file)
