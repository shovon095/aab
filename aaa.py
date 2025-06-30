AttributeError: 'ConfigDict' object has no attribute 'val'
Traceback (most recent call last):
  File "tools/test.py", line 286, in <module>
    main()
  File "tools/test.py", line 147, in main
    cfg = compat_cfg(cfg)
  File "/home/shouvon/CFINet/mmdet/utils/compat_config.py", line 17, in compat_cfg
    cfg = compat_loader_args(cfg)
  File "/home/shouvon/CFINet/mmdet/utils/compat_config.py", line 98, in compat_loader_args
    if 'samples_per_gpu' in cfg.data.val:
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/utils/config.py", line 49, in __getattr__
    raise ex
AttributeError: 'ConfigDict' object has no attribute 'val'
Traceback (most recent call last):
  File "tools/test.py", line 286, in <module>
    main()
  File "tools/test.py", line 147, in main
    cfg = compat_cfg(cfg)
  File "/home/shouvon/CFINet/mmdet/utils/compat_config.py", line 17, in compat_cfg
    cfg = compat_loader_args(cfg)
  File "/home/shouvon/CFINet/mmdet/utils/compat_config.py", line 98, in compat_loader_args
    if 'samples_per_gpu' in cfg.data.val:
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/utils/config.py", line 49, in __getattr__
    raise ex
AttributeError: 'ConfigDict' object has no attribute 'val'
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 20049) of binary: /home/shouvon/miniconda3/envs/cfinet/bin/python
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
[1]:
  time      : 2025-06-30_01:21:08
  host      : dxs4-DGX-Station
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 20050)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2025-06-30_01:21:08
  host      : dxs4-DGX-Station
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 20051)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[3]:
  time      : 2025-06-30_01:21:08
  host      : dxs4-DGX-Station
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 20052)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-06-30_01:21:08
  host      : dxs4-DGX-Station
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 20049)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
