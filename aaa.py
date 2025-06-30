 demo/de>     demo/demo.jpg \
>     configs/cfinet/faster_rcnn_r50_fpn_cfinet_1x.py \
>     work_dirs/faster_rcnn_r50_fpn_cfinet_1x/epoch_10.pth \
>     --device cuda \
>     --out-file my_visualizations/final_working_result.jpg
/home/shouvon/CFINet/mmdet/models/losses/iou_loss.py:266: UserWarning: DeprecationWarning: Setting "linear=True" in IOULoss is deprecated, please use "mode=`linear`" instead.
  warnings.warn('DeprecationWarning: Setting "linear=True" in '
/home/shouvon/CFINet/mmdet/models/dense_heads/anchor_head.py:116: UserWarning: DeprecationWarning: `num_anchors` is deprecated, for consistency or also use `num_base_priors` instead
  warnings.warn('DeprecationWarning: `num_anchors` is deprecated, '
/home/shouvon/CFINet/mmdet/models/dense_heads/anchor_head.py:123: UserWarning: DeprecationWarning: anchor_generator is deprecated, please use "prior_generator" instead
  warnings.warn('DeprecationWarning: anchor_generator is deprecated, '
load checkpoint from local path: work_dirs/faster_rcnn_r50_fpn_cfinet_1x/epoch_10.pth
/home/shouvon/CFINet/mmdet/datasets/utils.py:66: UserWarning: "ImageToTensor" pipeline is replaced by "DefaultFormatBundle" for batch inference. It is recommended to manually replace it in the test data pipeline in your config file.
  warnings.warn(
/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/torch/functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1634272068694/work/aten/src/ATen/native/TensorShape.cpp:2157.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
