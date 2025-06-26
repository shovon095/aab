# === PASTE THIS ENTIRE BLOCK INTO: work_dirs/cfinet_sodad/cfinet_r50_1x_soda.py ===

# 1. Inherit from the base configurations
_base_ = [
    '../../configs/_base_/models/faster_rcnn_r50_fpn.py',
    '../../configs/_base_/schedules/schedule_1x.py', 
    '../../configs/_base_/default_runtime.py'
]

# 2. Set custom model parameters for CFINet
find_unused_parameters=True
rpn_weight = 0.9
model = dict(
    type='FasterRCNN',
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    rpn_head=dict(
        _delete_=True,
        type='CRPNHead',
        num_stages=2,
        stages=[
            dict(
                type='StageRefineRPNHead',
                in_channels=256,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[2],
                    ratios=[1.0],
                    strides=[4, 8, 16, 32]),
                refine_reg_factor=200.0,
                refine_cfg=dict(type='dilation', dilation=3),
                refined_feature=True,
                sampling=False,
                with_cls=False,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.1, 0.1, 0.5, 0.5)),
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=10.0 * rpn_weight)),
            dict(
                type='StageRefineRPNHead',
                in_channels=256,
                feat_channels=256,
                refine_cfg=dict(type='offset'),
                refined_feature=True,
                sampling=True,
                with_cls=True,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.05, 0.05, 0.1, 0.1)),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0 * rpn_weight),
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=10.0 * rpn_weight))]),
    roi_head=dict(
        type='FIRoIHead',
        num_gpus=1,
        temperature=0.6,
        contrast_loss_weights=0.50,
        num_con_queue=256,
        con_sampler_cfg=dict(
            num=128,
            pos_fraction=[0.5, 0.25, 0.125]),
        con_queue_dir="./work_dirs/roi_feats/cfinet",
        ins_quality_assess_cfg=dict(
            cls_score=0.05,
            hq_score=0.65,
            lq_score=0.25,
            hq_pro_counts_thr=8),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=5, # <<< PROBLEM C FIXED: Changed from 9 to 5
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        rpn=[
            dict(
                assigner=dict(
                    type='DynamicAssigner',
                    low_quality_iou_thr=0.2,
                    base_pos_iou_thr=0.25,
                    neg_iou_thr=0.15),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ],
        rpn_proposal=dict(max_per_img=300, nms=dict(iou_threshold=0.8)),
        rcnn=dict(
            assigner=dict(
                pos_iou_thr=0.50, neg_iou_thr=0.50, min_pos_iou=0.50),
            sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5))),
    test_cfg=dict(
        rpn=dict(max_per_img=300, nms=dict(iou_threshold=0.5)),
        rcnn=dict(score_thr=0.05))
)

# 3. Define the dataset paths and settings explicitly
dataset_type = 'CocoDataset'
data_root = '/home/shouvon/CFINet/SODA-D-1024/' # Use the full path to your data

# PROBLEM A & B FIXED: Override the entire data dictionary
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # Point to your exact annotation file
        ann_file=data_root + 'Annotations/train.json',
        # Point to your exact image folder
        img_prefix=data_root + 'Images/train/'),
    # Disable val and test loaders since data is missing
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'Annotations/train.json', # Using train data as placeholder
        img_prefix=data_root + 'Images/train/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'Annotations/train.json', # Using train data as placeholder
        img_prefix=data_root + 'Images/train/'))

# 4. Define optimizer and learning rate schedule
fp16 = dict(loss_scale='dynamic')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12

# Disable evaluation during training to avoid crashing on missing val set
evaluation = dict(interval=1000, metric='bbox') # Set interval high to effectively disable
log_config = dict(interval=50)
