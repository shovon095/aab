after_run:
(VERY_LOW    ) TextLoggerHook
 --------------------
2025-06-26 01:37:46,825 - mmdet - INFO - workflow: [('train', 1)], max: 12 epochs
2025-06-26 01:37:46,827 - mmdet - INFO - Checkpoints will be saved to /home/shouvon/CFINet/work_dirs/cfinet_r50_1x_soda by HardDiskBackend.
Traceback (most recent call last):
Traceback (most recent call last):
Traceback (most recent call last):
  File "tools/train.py", line 247, in <module>
  File "tools/train.py", line 247, in <module>
  File "tools/train.py", line 247, in <module>
            main()main()main()


  File "tools/train.py", line 236, in main
  File "tools/train.py", line 236, in main
  File "tools/train.py", line 236, in main
        train_detector(train_detector(

  File "/home/shouvon/CFINet/mmdet/apis/train.py", line 246, in train_detector
train_detector(  File "/home/shouvon/CFINet/mmdet/apis/train.py", line 246, in train_detector

  File "/home/shouvon/CFINet/mmdet/apis/train.py", line 246, in train_detector
        runner.run(data_loaders, cfg.workflow)runner.run(data_loaders, cfg.workflow)

      File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 127, in run
runner.run(data_loaders, cfg.workflow)  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 127, in run

  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 127, in run
        epoch_runner(data_loaders[i], **kwargs)epoch_runner(data_loaders[i], **kwargs)

      File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 45, in train
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 45, in train
epoch_runner(data_loaders[i], **kwargs)
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 45, in train
        self.call_hook('before_train_epoch')self.call_hook('before_train_epoch')

      File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/base_runner.py", line 309, in call_hook
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/base_runner.py", line 309, in call_hook
self.call_hook('before_train_epoch')
  File "/home/shouvon/miniconda3/envs/cfinet/lib/python3.8/site-packages/mmcv/runner/base_runner.py", line 309, in call_hook
        getattr(hook, fn_name)(self)getattr(hook, fn_name)(self)

      File "/home/shouvon/CFINet/mmdet/datasets/utils.py", line 158, in before_train_epoch
getattr(hook, fn_name)(self)  File "/home/shouvon/CFINet/mmdet/datasets/utils.py", line 158, in before_train_epoch

  File "/home/shouvon/CFINet/mmdet/datasets/utils.py", line 158, in before_train_epoch
        self._check_head(runner)self._check_head(runner)

      File "/home/shouvon/CFINet/mmdet/datasets/utils.py", line 144, in _check_head
self._check_head(runner)  File "/home/shouvon/CFINet/mmdet/datasets/utils.py", line 144, in _check_head

  File "/home/shouvon/CFINet/mmdet/datasets/utils.py", line 144, in _check_head
    assert module.num_classes == len(dataset.CLASSES), \
assert module.num_classes == len(dataset.CLASSES), \
AssertionErrorassert module.num_classes == len(dataset.CLASSES), \:
AssertionErrorThe `num_classes` (9) in FIRoIHead of MMDistributedDataParallel does not matches the length of `CLASSES` 80) in CocoDataset:
AssertionErrorThe `num_classes` (9) in FIRoIHead of MMDistributedDataParallel does not matches the length of `CLASSES` 80) in CocoDataset:
The `num_classes` (9) in FIRoIHead of MMDistributedDataParallel does not matches the length of `CLASSES` 80) in CocoDataset
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
