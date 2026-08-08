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
`experiments/rl/atari/amarel/submit_atari_5task_18action_l3_pilot.sh`。当前 per-task LR decay
protocol 使用 `submit_atari_5task_18action_l3_lrpertask_pilot.sh`：shared optimizer 仅在每个
task 都达到 1M environment steps 后才 decay，因此 5M global-step pilot 中 LR 不下降。它先
提交 5-model smoke，再以 `afterok` 提交 5-model × 3-seed × 5M-step pilot；三个较慢的 GaWF
seeds 会优先与五个非-GaWF units 并行，array 最大并发仍为 8。
所有 structured results 和后续 figures 分别写入
`results/data/rl/atari/5task_18action/{parameter_match,smoke,pilot,figs}/`，不得写入或覆盖
`multitask_18action`。
