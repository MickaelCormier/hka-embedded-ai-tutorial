import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from mmpose.apis import init_model, inference_topdown

config = '/mmlab/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py'
checkpoint = '/mmlab/mmpose/models/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth'
image_path = '/mmlab/mmdeploy/demo/resources/human-pose.jpg'


model = init_model(config, checkpoint, device='cuda') # 'cpu'

human_bbox = [[0, 0, 218, 346]]

num_runs = 50

# Run inference
execution_times = []

print('Start inference.')

for _ in range(num_runs):
    start_time = time.time()
    pose_results = inference_topdown(model, image_path, human_bbox, bbox_format="xywh")
    end_time = time.time()

    execution_time = (end_time - start_time) * 1000
    execution_times.append(execution_time)

print(pose_results)

# Remove outliers
Q1 = np.percentile(execution_times, 25)
Q3 = np.percentile(execution_times, 75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

no_outliers = [time for time in execution_times if time >= lower_bound and time <= upper_bound]

avg_time = np.mean(execution_times)
max_time = np.max(execution_times)
min_time = np.min(execution_times)
std_dev = np.std(execution_times)

print(f'With outliers: avg time: {avg_time} ms, max time: {max_time} ms, min time: {min_time} ms, std dev: {std_dev} ms')

avg_time = np.mean(no_outliers)
max_time = np.max(no_outliers)
min_time = np.min(no_outliers)
std_dev = np.std(no_outliers)

print(f'Without outliers: avg time: {avg_time} ms, max time: {max_time} ms, min time: {min_time} ms, std dev: {std_dev} ms')
