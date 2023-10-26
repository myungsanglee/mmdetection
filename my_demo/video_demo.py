# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import cv2
import torch

from mmdet.apis import inference_detector, init_detector


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('video', help='video file')
    parser.add_argument('config', help='vonfig file')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file path (mp4)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise OSError('Can not open video')
    
    print(f'Camera FPS: {int(round(cap.get(cv2.CAP_PROP_FPS)))}')
    print(f'Camera Width: {int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))}')
    print(f'Camera Height: {int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))}')
    
    video_writer = None
    if args.out:
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
    
    classes = model.dataset_meta['classes']
    palette = model.dataset_meta['palette']
    
    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('result', 640, 480)
    
    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        t0 = time.time()
        
        ret_val, frame = cap.read()
        if not ret_val:
            print('can not read frame from video')
            break
        
        t1 = time.time()
        result = inference_detector(model, frame)
        inference = (time.time() - t1) * 1000

        # get pred result
        pred_instances = result.pred_instances
        pred_instances = pred_instances[pred_instances.scores > args.score_thr]
        # pred_instances = pred_instances[pred_instances.labels == 0]
        
        # draw bboxes
        if 'bboxes' in pred_instances and pred_instances.bboxes.sum() > 0:
            bboxes = pred_instances.bboxes.detach().cpu().numpy()
            labels = pred_instances.labels.detach().cpu().numpy()
            scores = pred_instances.scores.detach().cpu().numpy()

            for label, bbox, score in zip(labels, bboxes, scores):
                x1, y1, x2, y2 = [int(round(x)) for x in bbox]
                
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color=palette[label], thickness=1)
                frame = cv2.putText(frame, 
                                f'{classes[label]}, {score:.2f}',
                                (x1, y1 + 20),
                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=1,
                                color=palette[label])


        fps = 1000 / ((time.time() - t0) * 1000)
        print(f'FPS: {fps:.2f}, Inference time: {inference:.2f}ms')
        
        if args.out:
            video_writer.write(frame)
            
        cv2.imshow('result', frame)

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

    if video_writer:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
