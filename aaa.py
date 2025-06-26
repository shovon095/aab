nch(
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
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
