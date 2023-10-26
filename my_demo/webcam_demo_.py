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

    # if args.rtsp is not None:
    #     camera = cv2.VideoCapture(args.rtsp)
    # else:
    #     camera = cv2.VideoCapture(args.camera_id)
    
    # camera = cv2.VideoCapture('/home/plx/datasets/fall_detection/test_videos/outside1.mp4')
    # camera = cv2.VideoCapture(0)
    
    camera = cv2.VideoCapture('/mnt/plx/datasets/fall_detection/test_videos/test.avi')
    
    fps_total = 0
    ms_total = 0
    tmp_num = 0
    count_num = 1000

    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        tmp_num += 1
        
        t0 = time.time()
        
        ret_val, img = camera.read()
        
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
        
        if tmp_num > 100:
            fps_total += fps
            ms_total += inference
            
            if tmp_num == (count_num + 100):
                break
        
        cv2.imshow('result', img)

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

    print(f'FPS avg: {fps_total/count_num:.2f}, Inference avg: {ms_total/count_num:.2f}')

if __name__ == '__main__':
    main()
