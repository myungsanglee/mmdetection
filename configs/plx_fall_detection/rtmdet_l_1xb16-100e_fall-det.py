_base_ = '../rtmdet/rtmdet_l_8xb32-300e_coco.py'

num_classes = 3
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    bbox_head=dict(num_classes=num_classes),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300),
)

work_dir = './work_dirs/v2'
dataset_type = 'CocoDataset'
data_root = '/mnt/plx/datasets/fall_detection/coco_format/v1/'
train_annotations = 'annotations/train_v1.json'
val_annotations = 'annotations/valid_v1.json'
metainfo = {
    'classes':
    ('normal_person', 'fallen_person', 'fire'),
    # palette is a list of color tuples, which is used for visualization.
    'palette':
    [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
}
batch_size = 16
num_workers = 16
img_scale = (640, 640)

vis_backends = [
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='fall-detection',
            name='v2'
        )
    )
]
visualizer = dict(type='Visualizer', vis_backends=vis_backends)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomResize',
        scale=img_scale*2,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=img_scale,
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=img_scale,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file=train_annotations,
        data_prefix=dict(img='train/')
    )
)

val_dataloader = dict(
    batch_size=batch_size, 
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file=val_annotations,
        data_prefix=dict(img='valid/')
    )
)
test_dataloader = val_dataloader

max_epochs = 100
stage2_num_epochs = 10
base_lr = 0.004
interval = 5
# eta = base_lr * (batch_size * n_gpu / 16)**0.5

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)]
)

val_evaluator = dict(ann_file=data_root + val_annotations)
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    _delete_=True,
    # type='OptimWrapper',
    type='AmpOptimWrapper', # for Mixed Precision Training
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)
)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=interval,
        max_keep_ckpts=3,  # only keep latest 3 checkpoints
        save_best='coco/bbox_mAP' # 'loss'
    ),
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP', # 'loss
        patience=10,
        min_delta=0.005
    )
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49
    ),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2
    )
]

# We can use the pre-trained model to obtain higher performance
load_from = 'checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'
