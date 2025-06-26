  File "tools/train.py", line 236, in main
    main()
  File "tools/train.py", line 236, in main
    train_detector(
  File "/home/shouvon/CFINet/mmdet/apis/train.py", line 246, in train_detector
    train_detector(
  File "/home/shouvon/CFINet/mmdet/apis/train.py", line 246, in train_detector
    runner.run(data_loaders, cfg.workflow)
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 127, in run
    runner.run(data_loaders, cfg.workflow)
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 127, in run
    epoch_runner(data_loaders[i], **kwargs)
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 45, in train
        epoch_runner(data_loaders[i], **kwargs)self.call_hook('before_train_epoch')

  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 45, in train
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/base_runner.py", line 309, in call_hook
    self.call_hook('before_train_epoch')
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/base_runner.py", line 309, in call_hook
    getattr(hook, fn_name)(self)
  File "/home/shouvon/CFINet/mmdet/datasets/utils.py", line 158, in before_train_epoch
    getattr(hook, fn_name)(self)
  File "/home/shouvon/CFINet/mmdet/datasets/utils.py", line 158, in before_train_epoch
    self._check_head(runner)
  File "/home/shouvon/CFINet/mmdet/datasets/utils.py", line 144, in _check_head
    self._check_head(runner)
  File "/home/shouvon/CFINet/mmdet/datasets/utils.py", line 144, in _check_head
    assert module.num_classes == len(dataset.CLASSES), \
AssertionError: The `num_classes` (9) in FIRoIHead of MMDistributedDataParallel does not matches the length of `CLASSES` 80) in CocoDataset
Traceback (most recent call last):
    assert module.num_classes == len(dataset.CLASSES), \  File "tools/train.py", line 247, in <module>

AssertionError: The `num_classes` (9) in FIRoIHead of MMDistributedDataParallel does not matches the length of `CLASSES` 80) in CocoDataset
    main()
  File "tools/train.py", line 236, in main
    train_detector(
  File "/home/shouvon/CFINet/mmdet/apis/train.py", line 246, in train_detector
    runner.run(data_loaders, cfg.workflow)
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 127, in run
    epoch_runner(data_loaders[i], **kwargs)
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 45, in train
    self.call_hook('before_train_epoch')
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/base_runner.py", line 309, in call_hook
    getattr(hook, fn_name)(self)
  File "/home/shouvon/CFINet/mmdet/datasets/utils.py", line 158, in before_train_epoch
    self._check_head(runner)
  File "/home/shouvon/CFINet/mmdet/datasets/utils.py", line 144, in _check_head
    assert module.num_classes == len(dataset.CLASSES), \
AssertionError: The `num_classes` (9) in FIRoIHead of MMDistributedDataParallel does not matches the length of `CLASSES` 80) in CocoDataset
Traceback (most recent call last):
  File "tools/train.py", line 247, in <module>
    main()
  File "tools/train.py", line 236, in main
    train_detector(
  File "/home/shouvon/CFINet/mmdet/apis/train.py", line 246, in train_detector
    runner.run(data_loaders, cfg.workflow)
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 127, in run
    epoch_runner(data_loaders[i], **kwargs)
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 45, in train
    self.call_hook('before_train_epoch')
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/base_runner.py", line 309, in call_hook
    getattr(hook, fn_name)(self)
  File "/home/shouvon/CFINet/mmdet/datasets/utils.py", line 158, in before_train_epoch
    self._check_head(runner)
  File "/home/shouvon/CFINet/mmdet/datasets/utils.py", line 144, in _check_head
    assert module.num_classes == len(dataset.CLASSES), \
AssertionError: The `num_classes` (9) in FIRoIHead of MMDistributedDataParallel does not matches the length of `CLASSES` 80) in CocoDataset
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 23459) of binary: /home/shouvon/miniconda3/envs/cfinet/bin/python
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
  time      : 2025-06-26_01:30:56
  host      : dxs4-DGX-Station
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 23460)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2025-06-26_01:30:56
  host      : dxs4-DGX-Station
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 23461)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[3]:
  time      : 2025-06-26_01:30:56
  host      : dxs4-DGX-Station
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 23462)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-06-26_01:30:56
  host      : dxs4-DGX-Station
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 23459)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
