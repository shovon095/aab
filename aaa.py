load checkpoint from local path: work_dirs/faster_rcnn_r50_fpn_cfinet_1x/epoch_10.pth
load checkpoint from local path: work_dirs/faster_rcnn_r50_fpn_cfinet_1x/epoch_10.pth
load checkpoint from local path: work_dirs/faster_rcnn_r50_fpn_cfinet_1x/epoch_10.pth
load checkpoint from local path: work_dirs/faster_rcnn_r50_fpn_cfinet_1x/epoch_10.pth
The model and loaded state dict do not match exactly

size mismatch for roi_head.bbox_head.fc_cls.weight: copying a param with shape torch.Size([10, 1024]) from checkpoint, the shape in current model is torch.Size([6, 1024]).
size mismatch for roi_head.bbox_head.fc_cls.bias: copying a param with shape torch.Size([10]) from checkpoint, the shape in current model is torch.Size([6]).
size mismatch for roi_head.bbox_head.fc_reg.weight: copying a param with shape torch.Size([36, 1024]) from checkpoint, the shape in current model is torch.Size([20, 1024]).
size mismatch for roi_head.bbox_head.fc_reg.bias: copying a param with shape torch.Size([36]) from checkpoint, the shape in current model is torch.Size([20]).
[                                                  ] 0/2, elapsed: 0s, ETA:/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/torch/functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1634272068694/work/aten/src/ATen/native/TensorShape.cpp:2157.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/torch/functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1634272068694/work/aten/src/ATen/native/TensorShape.cpp:2157.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/torch/functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1634272068694/work/aten/src/ATen/native/TensorShape.cpp:2157.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/torch/functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1634272068694/work/aten/src/ATen/native/TensorShape.cpp:2157.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 4/2, 1.4 task/s, elapsed: 3s, ETA:     0sTraceback (most recent call last):
  File "tools/test.py", line 286, in <module>
    main()
  File "tools/test.py", line 278, in main
    metric = dataset.evaluate(outputs, **eval_kwargs)
  File "/home/shouvon/CFINet/mmdet/datasets/coco.py", line 641, in evaluate
    result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
  File "/home/shouvon/CFINet/mmdet/datasets/coco.py", line 383, in format_results
    result_files = self.results2json(results, jsonfile_prefix)
  File "/home/shouvon/CFINet/mmdet/datasets/coco.py", line 315, in results2json
    json_results = self._det2json(results)
  File "/home/shouvon/CFINet/mmdet/datasets/coco.py", line 252, in _det2json
    data['category_id'] = self.cat_ids[label]
IndexError: list index out of range
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 19442) of binary: /home/shouvon/miniconda3/envs/cfinet/bin/python
Traceback (most recent call last):
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/torch/distributed/launch.py", line 193, in <module>
    main()
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/torch/distributed/launch.py", line 189, in main
    launch(args)
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/torch/distributed/launch.py", line 174, in launch
    run(args)
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/torch/distributed/run.py", line 710, in run
    elastic_launch(
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 131, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 259, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
tools/test.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-06-30_01:09:41
  host      : dxs4-DGX-Station
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 19442)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
