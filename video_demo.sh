python demo/video_demo.py \
/home/plx/datasets/fall_detection/test_videos/outside3.mp4 \
configs/plx_fall_detection/rtmdet_l_1xb16-100e_fall-det.py \
work_dirs/rtmdet_l_1xb16-100e_fall-det/best_coco_bbox_mAP_epoch_70.pth \
--device 'cuda:0' \
--score-thr 0.5 \
--out '/home/plx/datasets/fall_detection/test_videos/outside3_thr-0.5_result.mp4'