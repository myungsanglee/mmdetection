# python my_demo/video_demo.py \
# /mnt/plx/datasets/fall_detection/test_videos/test.avi \
# configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
# checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth \
# --device 'cuda:0' \
# --score-thr 0.5


python my_demo/video_demo.py \
/mnt/plx/datasets/fall_detection/test_videos/test.avi \
configs/plx_fall_detection/rtmdet_l_1xb16-100e_fall-det.py \
work_dirs/rtmdet_l_1xb16-100e_fall-det/best_coco_bbox_mAP_epoch_70.pth \
--device 'cuda:0' \
--score-thr 0.5