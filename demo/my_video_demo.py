# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import cv2
import mmcv
import torch

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection my video demo')
    parser.add_argument('video', help='video file')
    parser.add_argument('config', help='vonfig file')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
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

    cap = cv2.VideoCapture(args.video)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('result', 640, 480)
    
    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        t0 = time.time()
        
        ret_val, img = cap.read()
        if not ret_val:
            break
        
        t1 = time.time()
        result = inference_detector(model, img)
        inference = (time.time() - t1) * 1000

        img = mmcv.imconvert(img, 'bgr', 'rgb')
        visualizer.add_datasample(
            name='result',
            image=img,
            data_sample=result,
            draw_gt=False,
            pred_score_thr=args.score_thr,
            show=False)

        img = visualizer.get_image()
        img = mmcv.imconvert(img, 'bgr', 'rgb')

        fps = 1000 / ((time.time() - t0) * 1000)
        print(f'FPS: {fps:.2f}, Inference time: {inference:.2f}ms')
        
        cv2.imshow('result', img)

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break


if __name__ == '__main__':
    main()
