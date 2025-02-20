from mmdet.apis import init_detector, inference_detector
from mmyolo.registry import VISUALIZERS
import mmcv
import os

config_file = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'
checkpoint_file = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'
image_path = 'demo.jpg'
output_path = 'vis_output.jpg'
score_thr = 0.75

model = init_detector(config_file, checkpoint_file, device='cuda:0')

# init visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

# Run inference
result = inference_detector(model, image_path)

print(result)

img = mmcv.imread(image_path)
img = mmcv.imconvert(img, 'bgr', 'rgb')

visualizer.add_datasample(
    os.path.basename(image_path),
    img,
    data_sample=result,
    draw_gt=False,
    show=False,
    wait_time=0,
    out_file=output_path,
    pred_score_thr=score_thr)
