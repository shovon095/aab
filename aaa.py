  File "tools/train.py", line 221, in main
    return obj_cls(**args)
    TypeErrordatasets = [build_dataset(cfg.data.train)]: __init__() missing 1 required positional argument: 'pipeline'


During handling of the above exception, another exception occurred:

  File "/home/shouvon/CFINet/mmdet/datasets/builder.py", line 82, in build_dataset
Traceback (most recent call last):
  File "tools/train.py", line 247, in <module>
    dataset = build_from_cfg(cfg, DATASETS, default_args)
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    main()
  File "tools/train.py", line 221, in main
    raise type(e)(f'{obj_cls.__name__}: {e}')
TypeError: CocoDataset: __init__() missing 1 required positional argument: 'pipeline'
    datasets = [build_dataset(cfg.data.train)]
  File "/home/shouvon/CFINet/mmdet/datasets/builder.py", line 82, in build_dataset
    dataset = build_from_cfg(cfg, DATASETS, default_args)
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')
TypeError: CocoDataset: __init__() missing 1 required positional argument: 'pipeline'
Traceback (most recent call last):
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/utils/registry.py", line 52, in build_from_cfg
    return obj_cls(**args)
TypeError: __init__() missing 1 required positional argument: 'pipeline'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "tools/train.py", line 247, in <module>
    main()
  File "tools/train.py", line 221, in main
    datasets = [build_dataset(cfg.data.train)]
  File "/home/shouvon/CFINet/mmdet/datasets/builder.py", line 82, in build_dataset
    dataset = build_from_cfg(cfg, DATASETS, default_args)
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')
TypeError: CocoDataset: __init__() missing 1 required positional argument: 'pipeline'
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 23290) of binary: /home/shouvon/miniconda3/envs/cfinet/bin/python
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
tools/train.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2025-06-26_01:25:26
  host      : dxs4-DGX-Station
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 23291)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2025-06-26_01:25:26
  host      : dxs4-DGX-Station
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 23292)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[3]:
  time      : 2025-06-26_01:25:26
  host      : dxs4-DGX-Station
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 23293)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-06-26_01:25:26
  host      : dxs4-DGX-Station
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 23290)
  error_file: <N/A>
