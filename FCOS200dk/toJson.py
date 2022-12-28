import numpy as np
from sys import argv
import argparse
import sys


def loadtxtmethod(filename):
    data = np.loadtxt(filename, dtype=np.float64, delimiter=' ')
    return data


if __name__ == '__main__':
    pre_file = argv[1]  # 预测的结果
    detect_file = argv[2]   # 存放评估结果
    pre_info = loadtxtmethod(pre_file)
    result = []
    if pre_info.size > 0:
        if (pre_info.size > 6):
            for item in pre_info:
                image_result = {
                    'image_id': image_id,
                    'category_id': int(item[5])+1,
                    'score': float(item[4]),
                    'bbox': [int(item[0]), int(item[1]), int(item[2]), int(item[3])]
                }
                coco_result.append(image_result)
        else:
            image_result = {
                'image_id': image_id,
                'category_id': int(pre_info[5]) + 1,
                'score': float(pre_info[4]),
                'bbox': [int(pre_info[0]), int(pre_info[1]), int(pre_info[2]), int(pre_info[3])]
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
    with open(detect_file, 'w') as f:
        json.dump(coco_result, f, indent=4)
