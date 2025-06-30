    os.rmdir(entry.name, dir_fd=topfd)
FileNotFoundError: [Errno 2] No such file or directory: '0'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/utils/registry.py", line 52, in build_from_cfg
    return obj_cls(**args)
  File "/home/shouvon/CFINet/mmdet/models/detectors/faster_rcnn.py", line 19, in __init__
    super(FasterRCNN, self).__init__(
  File "/home/shouvon/CFINet/mmdet/models/detectors/two_stage.py", line 50, in __init__
    self.roi_head = build_head(roi_head)
  File "/home/shouvon/CFINet/mmdet/models/builder.py", line 40, in build_head
    return HEADS.build(cfg)
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/utils/registry.py", line 215, in build
    return self.build_func(*args, **kwargs, registry=self)
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/cnn/builder.py", line 27, in build_model_from_cfg
    return build_from_cfg(cfg, registry, default_args)
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')
FileNotFoundError: FIRoIHead: [Errno 2] No such file or directory: '0'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "tools/test.py", line 286, in <module>
    main()
  File "tools/test.py", line 222, in main
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
  File "/home/shouvon/CFINet/mmdet/models/builder.py", line 58, in build_detector
    return DETECTORS.build(
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/utils/registry.py", line 215, in build
    return self.build_func(*args, **kwargs, registry=self)
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/cnn/builder.py", line 27, in build_model_from_cfg
    return build_from_cfg(cfg, registry, default_args)
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')
FileNotFoundError: FasterRCNN: FIRoIHead: [Errno 2] No such file or directory: '0'
load checkpoint from local path: work_dirs/faster_rcnn_r50_fpn_cfinet_1x/epoch_10.pth
load checkpoint from local path: work_dirs/faster_rcnn_r50_fpn_cfinet_1x/epoch_10.pth
load checkpoint from local path: work_dirs/faster_rcnn_r50_fpn_cfinet_1x/epoch_10.pth
[                                                  ] 0/2, elapsed: 0s, ETA:WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 20183 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 20185 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 20186 closing signal SIGTERM
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 1 (pid: 20184) of binary: /home/shouvon/miniconda3/envs/cfinet/bin/python
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
  time      : 2025-06-30_01:34:42
  host      : dxs4-DGX-Station
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 20184)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
