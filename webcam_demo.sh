# python3 my_demo/webcam_demo.py \
# configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
# checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth \
# --device 'cuda:0' \
# --camera-id 0 \
# --rtsp 'rtsp://192.168.4.79/profile4/media.smp' \
# --score-thr 0.5


python3 my_demo/webcam_demo.py \
configs/plx_fall_detection/rtmdet_l_1xb16-100e_fall-det.py \
work_dirs/rtmdet_l_1xb16-100e_fall-det/best_coco_bbox_mAP_epoch_70.pth \
--device 'cuda:0' \
--camera-id 0 \
--rtsp 'rtsp://192.168.4.79/profile4/media.smp' \
--score-thr 0.5
