# MiniGrid experiments

MiniGrid DQN/DRQN、通用 PPO 和 paper-protocol PPO 分别由 `run_task.py minigrid-dqn`、
`run_task.py minigrid-ppo` 与 `run_task.py minigrid-ppo-paper` 启动。共享 recurrent core 位于
`utils/training/recurrent_cores/`，任务专用 encoder/head 位于
`utils/training/minigrid/`。

Amarel paper PPO 由 `amarel/submit_minigrid_ppo_paper.sh` 提交；本地双 GPU
smoke 使用 `experiments/local/run_minigrid_ppo_paper_smoke_2gpu.sh`。远端输出必须显式
使用 `AIM3_RESULTS_PATH`，不依赖旧 MiniGrid worktree。
