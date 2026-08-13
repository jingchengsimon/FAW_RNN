# Remote Experiment Job History

该文档由 `python -m experiments.monitoring.job_registry rebuild` 从 `jobs/*.json`
生成。记录默认永久保留；只有人类明确确认后才能删除。
它只服务于实验定位和检测，不是实验协议或项目方法定义。

| Experiment ID | Status | Host | Scheduler / run IDs | Units | Remote root | Description |
|---|---|---|---|---:|---|---|
| `atari5-l3-eps500k-lrpertask-20m-lstm-gru-gawf-s1-2` | running | `sjc-remote` | sjc_5task_l3_20m_seed1_20260812, sjc_5task_l3_20m_seed2_20260812, sjc_5task_l3_20m_seed1_20260812 | 6 | `/G/MIMOlab/Codes/aim3_gawf_rnn` | Five-task full18 L3 formal 20M run: LSTM/GRU/GaWF, two seeds, per-task 500K replay, per-task 1M LR decay, and fixed 500K global epsilon decay. |

单个 job 的精确日志、结果路径、完成条件和备注位于对应的
`jobs/<id>.json`。
