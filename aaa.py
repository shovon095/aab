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
        rpn=dict(max_per_img=300, nms=dict(iou_threshold=0.1)),
        rcnn=dict(score_thr=0.005))
)

