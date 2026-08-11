# Atari experiments

Atari A2C 使用 `python run_task.py atari-a2c`；DQN/DRQN 使用
`python run_task.py atari-dqn`。本目录保存任务定义，
包括 `atari_ssm_param_match.py`。Slurm 和双 GPU wrappers 仍按执行环境放在
`amarel/`。

Pong 名称必须明确写为 `pong_fs1_stack1` 或 `pong_fs4_stack1`。正式运行记录 commit hash、
frame skip、frame stack、feedback mode 和结果 suffix，不依赖旧 Atari worktree 路径。

严格单任务 Pong（6-action）和 Breakout（4-action）的 checkpoint 视频使用
`utils/analysis/evaluate_atari_dqn_video.py` 在 `render_mode=rgb_array` 下执行 greedy evaluation，
并由 OpenCV 直接编码 MP4，避免依赖 Gymnasium 的可选 MoviePy 录制组件。正式视频同时保存
metadata JSON，注明训练 seed、evaluation seed、逐 episode return、被选中的最佳 episode、
frame protocol 和 checkpoint。

## Five-task full-18 pilot

固定 five-task 协议使用 Pong、Breakout、Assault、Seaquest 和 Skiing，所有任务共享
ALE canonical 18-action output，模型不接收 task ID。Collection 在 episode boundary 按累计
environment steps 选择最少的 task；replay batch 的非整除 remainder 在 task 间轮转。
训练须等每个 task 至少收集 20k valid environment steps 后才开始 update，scheduler
task counts/cursor 随 checkpoint 恢复。
Skiing 的 ALE legal action set 只有 9 个动作，因此 unsupported fire variants 映射到对应的
legal non-fire movement，standalone FIRE 映射到 NOOP；模型仍保持 18-dim output，且不使用
task-specific action mask。

历史 global-step decay 的 Amarel 提交入口是
`experiments/rl/atari/amarel/submit_atari_5task_18action_l3_pilot.sh`。当前正式 five-task
protocol 使用 `submit_atari_5task_18action_l3_lrpertask_pilot.sh`：每个 task 的 LR 在其自身
达到 1M environment steps 后才 decay；该 5M global-step pilot 因而不会触发 decay。它不设
smoke gate，提交 5-model × 3-seed × 5M-step array，三个较慢的 GaWF seeds 优先，最大并发为 5。
每个 task 使用独立的 0.5M mmap replay（每个 unit 总容量 2.5M transitions）；任务完成后清理
replay。Amarel 请求为单 GPU、16 CPUs、64G memory、30 小时 walltime，且不固定 GPU 型号，
以提高 backfill 机会。structured results 和后续 figures 分别写入
`results/data/rl/atari/5task_18action/{parameter_match,smoke,pilot,figs}/`；此 protocol 的 pilot
结果根为 `pilot/per_task_buf500k/`，不得写入或覆盖 `multitask_18action`。

10M formal GRU/LSTM 扩展使用
`amarel/submit_atari_5task_18action_formal_10m.sh`。它固定五个 task、full18、fs4/stack4、
five 个独立 1M mmap partitions、10M global steps（每 task 2M）、GRU L3/h458 和当前验证的
LSTM L3/h373，各三个 seeds。它先运行 500-step structural/recovery smoke：checkpoint 必须包含
五个 replay partitions，受控 `SIGUSR1` 后自动 requeue/resume；仅 smoke 成功才通过
`afterok` 释放 six-unit formal array。成功的 smoke result/artifact leaves 由独立 compute
cleanup job 精确删除；正式 run 成功时只由训练入口删除本 unit 的 replay。
失败 smoke leaf 必须保留；重提时使用唯一的 `--smoke-attempt <tag>`，仅隔离 smoke 路径，
不改变六个 formal result suffix。

## SJC two-task L3 GRU comparison

The SJC comparison launcher `experiments/remote/run_sjc_atari_multitask_l3_gru.sh` evaluates a
GRU L3 h458, seed 42 protocol using Pong plus Breakout (4M global steps) or a corresponding
single-task control (2M steps). All use full18, `fs4/stack4`, per-task 1M mmap replay and a
per-task 1M LR decay; the two-task run additionally uses `transition_balanced` collection and
`task_balanced` replay. Its smoke is fixed at 25k steps and writes to a separate `_smoke` result
leaf. These results belong under `results/data/rl/atari/multitask_18action/`, never the fixed
five-task namespace.
