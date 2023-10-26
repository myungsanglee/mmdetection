# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import cv2
import mmcv
import torch

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--rtsp', type=str, help='rtsp address')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    device = torch.device(args.device)
    model = init_detector(args.config, args.checkpoint, device=device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    if args.rtsp is not None:
        camera = cv2.VideoCapture(args.rtsp)
    else:
        camera = cv2.VideoCapture(args.camera_id)
    
    # camera = cv2.VideoCapture('/home/plx/datasets/fall_detection/test_videos/outside1.mp4')
    # camera = cv2.VideoCapture(0)
    
    print(f'Camera FPS: {int(round(camera.get(cv2.CAP_PROP_FPS)))}')
    print(f'Camera Width: {int(round(camera.get(cv2.CAP_PROP_FRAME_WIDTH)))}')
    print(f'Camera Height: {int(round(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))}')
    
    classes = model.dataset_meta['classes']
    palette = model.dataset_meta['palette']

    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        t0 = time.time()
        
        ret_val, img = camera.read()
        if not ret_val:
            print('can not read frame from camera')
            break
        
        t1 = time.time()
        result = inference_detector(model, img)
        inference = (time.time() - t1) * 1000
        
        pred_instances = result.pred_instances
        # print(pred_instances)
        pred_instances = pred_instances[pred_instances.scores > args.score_thr]
        # print(pred_instances)
        pred_instances = pred_instances[pred_instances.labels == 0]
        # print(pred_instances)
        
        if 'bboxes' in pred_instances and pred_instances.bboxes.sum() > 0:
            bboxes = pred_instances.bboxes.detach().cpu().numpy()
            labels = pred_instances.labels.detach().cpu().numpy()
            scores = pred_instances.scores.detach().cpu().numpy()

            for label, bbox, score in zip(labels, bboxes, scores):
                x1, y1, x2, y2 = [int(round(x)) for x in bbox]
                
                img = cv2.rectangle(img, (x1, y1), (x2, y2), color=palette[label], thickness=1)
                img = cv2.putText(img, 
                                f'{classes[label]}, {score:.2f}',
                                (x1, y1 + 20),
                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=1,
                                color=palette[label])
                
                
        
        # break
        
        fps = 1000 / ((time.time() - t0) * 1000)
        print(f'FPS: {fps:.2f}, Inference time: {inference:.2f}ms')
        
        cv2.imshow('result', img)

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

if __name__ == '__main__':
    main()
