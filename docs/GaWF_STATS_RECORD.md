# GaWF 10-seed 统计记录

更新日期：2026-08-19

## 口径与范围

本记录逐项对应 `GaWF_stats_request.md`。数据优先从 Amarel / sjc-remote 的已登记实验精确
路径读取；远端数值文件只在 SSH 会话中读取，未同步到本地。本地 `results/save/` 仅保留当前
正式图的交付文件，`results/save_data/` 仅用于本地已有 curated data 的核对。

除非另有说明，数值格式为 **10 个 training seeds 的均值 ± seed-level SEM**，其中
SEM 为 $s/\sqrt{10}$。accuracy 与 variance fraction 使用百分数。
门中位数先在每个 seed 内求得，再报告十个 seed median 的均值 ± SEM。除 §6.15 的指定
interaction test 外，不报告 p value。

数据审计发现：

- Fig1 test accuracy 现使用六个模型各 10 seeds 的 reset-excluded CSV：每个 32-frame rollout
  的 `t=0` 都从 accuracy 分母中剔除（每个 seed 保留 55,769/57,568 frames）。
- Fig1 target-switch recovery 使用独立的
  `40h-float32-jointswitch-balanced-10digit-unique` test protocol；每个 32-frame rollout 的
  `t=0` 同样剔除。Supplementary 1 保持其原有 512-frame rollout protocol，并剔除其每个
  512-frame window 的 `t=0`。每个 offset 的阴影是 training-seed mean 的 SEM，不是 pooled
  switch-event uncertainty。
- 本地缺少 Fig4 activation 和 formal Fig7 的 compact summaries；本记录直接只读查询
  sjc-remote 已登记的 10-seed result paths，没有同步这些数值文件。
- Amarel 的 `multiseed_1_10/seed01`–`seed10` 含完整 GaWF synapse-level unified
  decomposition；顶层 manifest 只是较早的单-checkpoint 结果，不能代表该 multiseed 子树。
- `results/save_data/supple2/gate_context_specificity/` 是单 trajectory 结果，不能用于本文
  10-seed 数字。
- §5.9 使用本次在 sjc-remote 完成的 10-seed × 9-sector compact analysis；training seed 是
  inference unit，未使用 pooled connection-level 显著性检验。
- Fig6 的正式 encoder maps 以 sjc-remote 的
  `fig6_encoder_patterns_resetexcluded_gawf_10seed/` 为唯一数据源：Sector/Digit 各有 10 个
  reset-excluded compact outputs；`results/save/` 的正式 Sector PDF 已由此重绘。早期
  `fig6_encoder_tuning_gawf_10seed/` 路径不再是任何正式图或统计的数据源，故不作为缺失项追踪。
- `results/save/` 的正式 Fig3 分布图及 Fig6 maps 不显示跨-seed uncertainty，因而不需要改变
  图形误差条；Fig3 histograms 与 Fig6 encoder maps 均使用 reset-excluded 数据。

### 正式图的 reset / window 口径总表

本表只覆盖 `results/save/` 的当前正式交付物；**全部**已排除每个 recurrent window 的 `t=0`。
`results/save/archive/` 仅保留文件历史，不是正式结果或数据源。所有 32/512 标记都是 rollout
window，而非总 frame 数。

| 图 | reset-excluded 新结果 | window | 说明 |
|---|---|---:|---|
| Fig1（test accuracy、target-switch recovery） | 是 | 32 | test 与 recovery 都逐 window 排除 `t=0`。 |
| Fig2（feedback-shuffle ablation） | 是 | 512 | baseline / shuffle-sector / shuffle-digit 同口径。 |
| Fig3（主 gate/weight distribution PDF） | 是 | 32 | `fig3_gate_distribution` 在写入每个 seed 的 histogram 前已通过 `feedback != 0` 排除 reset；PDF 从这些 reset-excluded histograms 汇总。 |
| Fig4（six-model activation） | 是 | 32 | reset-excluded 60 model-seed compact summaries。 |
| Fig4（core objects：input/recurrent gate + activations） | 是 | 32 | 本次重算 raw synapse gate 后更新，含 10-seed points。 |
| Fig4（shuffle activation / gate ANOVA） | 是 | 512 | §4.12 / §2 shuffle-ablation protocol。 |
| Fig5（unit-gate marginalization） | 是 | 32 | GaWF 是 raw synapse 的 destination-unit projection。 |
| Fig6（encoder pattern maps） | 是 | 32 | Sector/Digit 各 10 seeds 均在 equal-n selection 前排除 1,799 个 `t=0`，各保留 55,769 frames；正式 Sector PDF 已更新。 |
| Fig6（sequential gate / sign maps） | 是 | 32 | 以 `feedback != 0` 排除 reset frame。 |
| Fig7（recurrent gate sign gaps） | 是 | 32 | 使用 reset-excluded r3 compact caches。 |
| Supplementary 1（feedback ablation recovery） | 是 | 512 | ablation 系列统一用 512-frame window。 |
| Supplementary 2（input-gate sign/magnitude） | 是 | 32 | 九个 sectors 的 reset-excluded compact analysis。 |
| Supplementary 3（recurrent-gate sign/magnitude） | 是 | 32 | Fig7 同一 reset-excluded compact caches。 |
| Supplementary 4（net recurrent-current decomposition） | 是 | 32 | §6.17 的 unit-level record：Digit 与 Sector 各自 equal-n，逐帧实际 `g·W·h(t−1)` 后按目的地单元数归一。Fig8 的正式柱图另用 §6.18 的 per-connection 口径。 |

## §1 — 行为表现

| ID | 需要的统计值 | 具体值（10-seed） |
|---|---|---|
| 1.2 | GaWF 的 test accuracy：digit、sector 两个读出 | **每个 32-frame rollout 剔除 t=0 后**：Digit **86.6169% ± 0.1480%**；Sector **93.2852% ± 0.1238%**；各 n=10。 |
| 1.3 | 五个 baseline 各自的 test accuracy，两个读出 | **每个 32-frame rollout 剔除 t=0 后**：RNN：Digit **80.8603% ± 0.1732%**，Sector **91.3516% ± 0.0858%**。LSTM：**80.4103% ± 0.2266%**，**90.9369% ± 0.0637%**。GRU：**79.4637% ± 0.1705%**，**90.9471% ± 0.0524%**。S5：**75.1772% ± 0.3548%**，**89.3149% ± 0.1764%**。Mamba：**83.0981% ± 0.1669%**，**92.5997% ± 0.0749%**。每项 n=10，顺序均为 Digit、Sector。 |
| 1.4 | 全文统一的 seed 数 | 本记录所有已交付统计统一为 **n=10 training seeds（seeds 1–10）**；不满足 n=10 的现有结果不报数字。 |
| 1.6 | 目标切换处准确率掉幅：GaWF 与各 baseline，两个读出 | 两张 recovery 图现均为 **n=10 training seeds**；Fig1 的每个 32-frame window 与 Supplementary 1 的每个 512-frame window 均排除 `t=0`，每个 offset 显示 seed mean ± SEM。尚未把“掉幅”压缩成单一 scalar，因为请求未指定使用 pre1、pre-window mean 或其他切换前基准。 |
| 1.7 | 恢复到切换前水平所需帧数 | 10-seed curves 已完成，但“恢复”的 threshold 与连续帧判据仍未定义，因此不擅自报告 recovery-frame scalar。 |
| 1.8 | switch window 横轴范围；绝对值或归一化 | Fig1 与 Supplementary 1 均为完整 **pre10–pre1、post1–post10**；纵轴为**绝对 accuracy (%)**，不是归一化值；每个点为 n=10 seed mean，阴影为 SEM。 |

## §2 — 反馈 shuffle 消融

正式的 baseline、shuffle-sector 与 shuffle-digit 均采用 §4.12 的相同 512-frame recurrent
rollout protocol；每个 condition、每个 seed 均排除每个 window 的 `t=0`，保留 57,232 frames。
shuffle 实现为：先按原始顺序计算真实 feedback，再在每个 512-frame window 内、对每个 sample
独立 permute 时间索引，仅替换指定的 feedback slice。这样三根柱子可作严格的同 protocol 对照。

| ID | 需要的统计值 | 具体值（10-seed） |
|---|---|---|
| 2.2 | Digit 读出：baseline / shuffle-sector / shuffle-digit | Baseline **89.7893% ± 0.1450%**；shuffle-sector **54.3366% ± 0.9315%**；shuffle-digit **73.5005% ± 0.5073%**；各 n=10。 |
| 2.3 | Sector 读出：同三个条件 | Baseline **94.2476% ± 0.1197%**；shuffle-sector **63.1848% ± 0.7705%**；shuffle-digit **91.7661% ± 0.1708%**；各 n=10。 |
| 2.4 | 可选：切换后恢复曲线形状量化 | **尚未分析。** 每个 seed 的曲线已保存，但没有已定义并保存的 shape scalar；按要求不新增量化。 |

## §3 — 门值分布

每个 seed 的 gate histogram 使用 400 个等宽 bins（范围 [0, 1]）。median 来自每个 seed
metadata 中保存的精确值；[0.1, 0.9] 比例由已保存的 per-seed histogram bins 直接汇总。
§3.5 则在 sjc-remote 直接读取十个 seed 的原始 Figure 3 feedback trajectories 和 `U/V`，按
原始 eager float32 公式重建 gate 并流式计数，没有把 trajectory 或新增数值文件同步到本地。
0.5 点质量判据为 $|g-0.5|<10^{-6}$；不接收反馈的目标单元判据为
$\max_r|U[j,r]|<10^{-6}$。

| ID | 需要的统计值 | 具体值（10-seed） |
|---|---|---|
| 3.1 | 输入门中位数 | seed median 的均值 **0.0088 ± 0.0012**，n=10。 |
| 3.2 | 循环门中位数 | seed median 的均值 **0.5108 ± 0.0094**，n=10。 |
| 3.5a | gate 值恰好为 0.5 的占比 | 按 $|g-0.5|<10^{-6}$：输入门 **3.125026% ± 0.000001%**；循环门 **3.125063% ± 0.000002%**；各 n=10。每个 seed 恰有 1,799/57,568 = **3.125%** 的 reset frames。reset 外对应占比仅为输入门 **0.0000257% ± 0.0000007%**、循环门 **0.0000633% ± 0.0000021%**。 |
| 3.5b | 0.5 点质量的跨 context 方差 | 对 reset 点质量按全部 90 个 observed digit-sector labels 分组，input 与 recurrent 的 synapse-level context-mean variance 分位数均为 **min/q25/median/q75/max = 0/0/0/0/0**；十个 seed 中每个分位数的均值均为 **0 ± 0**。因为 reset feedback 为零，所有 gate 均严格为 `sigmoid(0)=0.5`。这不是一批跨完整 trajectory 恒为 0.5 的固定 synapses；$10^{-6}$ 判据包含的极少量 reset 外近 0.5 值不属于该零方差点质量。 |
| 3.5c | 剔除 0.5 点质量后，剩余 gate 落在 [0.1, 0.9] 的比例 | 输入门 **14.8352% ± 0.3929%**；循环门 **33.3198% ± 0.9836%**；各 n=10。计算式为 `(middle_count-half_count)/(total_count-half_count)`。 |
| 3.5d | 0.5 点质量是否按行成块；不接收反馈的 hidden units | **否。** 点质量覆盖 reset frame 的全部行，而不是固定 destination rows。按 $\max_r|U[j,r]|<10^{-6}$，输入门与循环门共享的 256 个 destination units 中不接收反馈者为 **0.0 ± 0.0 / 256**，n=10；十个 seed 均为 0。各 seed 的最小 row-wise $\max_r|U[j,r]|$ 为 0.1399–0.2125，远离阈值。 |
| 3.5e | 两端 gate 值占比 | **剔除每条 sequence 的 t=0 reset frame 后**，由每个 seed 的 400-bin histogram 汇总，区间为 $[0,0.1)$ / $[0.9,1]$。输入门分别为 **64.0288% ± 0.5675%** / **21.2005% ± 0.5741%**；循环门分别为 **32.0196% ± 0.8129%** / **34.8294% ± 0.7393%**；各 n=10。 |
| 3.6 | 门值落在 [0.1, 0.9] 的比例 | **剔除每条 sequence 的 t=0 reset frame 后**：输入门 **14.8352% ± 0.3929%**；循环门 **33.3199% ± 0.9836%**；各 n=10。该口径等同于 3.5c 所报告的非 reset gate 中间区间比例。 |

**§3.5 解释：** 原先提出的两种解释并不完备。0.5 spike 的主因是每条 sequence 首帧的
zero-feedback reset，而不是 `U[j]≈0` 的死通路；剔除该 spike 后，循环门仍有约三分之一的值
位于中间区间。因此现有结果不支持把 recurrent gate 概括为普遍 binary；至少需要把输入门与
循环门的表述分开，并把 recurrent gate 描述为保留显著 graded mass。

## §4 — 方差分解

4.1–4.3 使用 sjc-remote 已完成的 60 个 model-seed units Fig4 activation ANOVA。每个
training seed 先对
20 个 fixed balanced draws 取均值，再跨 10 seeds 计算 SEM；分母是 balanced condition means
的总方差，Sector、Digit、Interaction 合计 100%，不含 trial-level residual。

4.8–4.11 使用 Fig5 已保存的 10-seed `unit_gate_context_variance_multiseed.json`。分母口径是
**balanced 9-sector × 10-digit condition means 的总方差**；Sector、Digit、Interaction 三项
归一化后合计 100%，**不含 trial-level residual**。GaWF 门先对每个 destination unit 的所有
incoming synapse raw sigmoid gates 做 arithmetic mean，再在 unit level 分解；LSTM/GRU 本身
为 unit gate。该口径不能替代 GaWF synapse-level 口径。

| ID | 需要的统计值 | 具体值（10-seed） |
|---|---|---|
| 4.1 | 六模型 encoder activation：sector / digit / interaction | **剔除 t=0 reset frame 后**。GaWF：**51.2719% ± 0.5870% / 6.7013% ± 0.0720% / 42.0267% ± 0.5170%**。RNN：**63.5789% ± 0.6671% / 5.2921% ± 0.1024% / 31.1290% ± 0.5652%**。LSTM：**65.8523% ± 0.6737% / 4.9673% ± 0.1094% / 29.1804% ± 0.5655%**。GRU：**55.1824% ± 0.7133% / 6.2990% ± 0.0937% / 38.5186% ± 0.6202%**。S5：**59.7009% ± 0.3988% / 5.9241% ± 0.0664% / 34.3749% ± 0.3410%**。Mamba：**64.2181% ± 0.4190% / 5.0755% ± 0.0564% / 30.7064% ± 0.3649%**。每项 n=10，顺序均为 Sector / Digit / Interaction。 |
| 4.2 | 六模型 hidden activation：sector / digit / interaction | **剔除 t=0 reset frame 后**。GaWF：**31.3420% ± 0.3851% / 52.4565% ± 0.4553% / 16.2015% ± 0.2271%**。RNN：**39.4247% ± 0.2474% / 41.9165% ± 0.2957% / 18.6588% ± 0.1605%**。LSTM：**41.6517% ± 0.2713% / 36.9830% ± 0.2365% / 21.3653% ± 0.2851%**。GRU：**39.0142% ± 0.3533% / 46.2347% ± 0.3884% / 14.7511% ± 0.2568%**。S5：**47.0554% ± 0.6575% / 28.7683% ± 0.7687% / 24.1763% ± 0.2371%**。Mamba：**40.1968% ± 0.2300% / 43.5827% ± 0.2141% / 16.2205% ± 0.1384%**。每项 n=10，顺序均为 Sector / Digit / Interaction。 |
| 4.3 | GaWF encoder 与其余模型的 sector、interaction | **剔除 t=0 reset frame 后**。Sector / Interaction：GaWF **51.2719% ± 0.5870% / 42.0267% ± 0.5170%**；RNN **63.5789% ± 0.6671% / 31.1290% ± 0.5652%**；LSTM **65.8523% ± 0.6737% / 29.1804% ± 0.5655%**；GRU **55.1824% ± 0.7133% / 38.5186% ± 0.6202%**；S5 **59.7009% ± 0.3988% / 34.3749% ± 0.3410%**；Mamba **64.2181% ± 0.4190% / 30.7064% ± 0.3649%**；各 n=10。 |
| 4.4 | 输入门（突触级）sector / digit / interaction | **剔除 t=0 reset frame 后**：Sector **76.7069% ± 0.5348%**；Digit **15.9906% ± 0.4413%**；Interaction **7.3025% ± 0.1266%**。各 seed 在 55,769 frames 上做相同的 20-draw balanced ANOVA，n=10。 |
| 4.5 | 循环门（突触级）sector / digit / interaction | **剔除 t=0 reset frame 后**：Sector **23.2512% ± 0.5253%**；Digit **71.7333% ± 0.4868%**；Interaction **5.0155% ± 0.1135%**。同 4.4 的 raw-synapse、20-draw、n=10 口径。 |
| 4.8 | GaWF 两个门 destination-unit 投影后的三分量 | **剔除 t=0 reset frame 后**：Input gate：Sector **88.2487% ± 3.1062%**，Digit **9.7954% ± 3.0684%**，Interaction **1.9558% ± 0.0739%**。Recurrent gate：**11.3422% ± 1.6160%**，**86.5034% ± 1.6762%**，**2.1543% ± 0.0777%**。各 n=10。 |
| 4.9 | GRU reset / update 门（unit 级）三分量 | **剔除 t=0 reset frame 后**：Reset：Sector **56.9358% ± 0.3889%**，Digit **22.7363% ± 0.5218%**，Interaction **20.3279% ± 0.3687%**。Update：**51.7696% ± 0.8600%**，**37.5716% ± 0.8743%**，**10.6587% ± 0.2951%**。各 n=10。 |
| 4.10 | LSTM input / forget / output 门（unit 级）三分量 | **剔除 t=0 reset frame 后**：Input：Sector **36.5091% ± 0.7504%**，Digit **52.3285% ± 0.7399%**，Interaction **11.1624% ± 0.2600%**。Forget：**62.6161% ± 0.8204%**，Digit **24.2261% ± 0.7064%**，Interaction **13.1578% ± 0.2224%**。Output：**65.8904% ± 0.7740%**，**23.8193% ± 0.6304%**，**10.2903% ± 0.2858%**。各 n=10。 |
| 4.11 | 七个门的 interaction 项，同一投影层级 | **剔除 t=0 reset frame 后**：GaWF input **1.9558% ± 0.0739%**；GaWF recurrent **2.1543% ± 0.0777%**；GRU reset **20.3279% ± 0.3687%**；GRU update **10.6587% ± 0.2951%**；LSTM input **11.1624% ± 0.2600%**；forget **13.1578% ± 0.2224%**；output **10.2903% ± 0.2858%**；各 n=10。 |

### 4.12 — §2 shuffle 条件下的 activation / gate ANOVA

复用 §2 的 `40h-uint8`、512-frame rollout 和逐 sample feedback permutation 顺序，在未
shuffle 的 ground-truth label 上做分解；每个条件、每个 seed 排除首个 reset frame 后有
57,232 frames，以相同的 20 个 fixed balanced draws 汇总。以下均为 **10-seed mean ± SEM**。
`between_condition_var` 是未归一化的 balanced condition-mean total variance；三项百分比以该
variance 为分母并合计 100%，`trial-level residual` 则以 trial-level total variance 为分母。

| feedback 条件 | Digit accuracy | Sector accuracy |
|---|---:|---:|
| Baseline | 89.7893% ± 0.1450% | 94.2476% ± 0.1197% |
| Shuffle digit | 73.5005% ± 0.5073% | 91.7661% ± 0.1708% |
| Shuffle sector | 54.3366% ± 0.9315% | 63.1848% ± 0.7705% |

| 对象 | feedback 条件 | Sector / Digit / Interaction | 未归一化 `between_condition_var` | trial-level residual |
|---|---|---|---:|---:|
| Encoder activation | Baseline | 51.3103% ± 0.5921% / 6.6934% ± 0.0735% / 41.9963% ± 0.5205% | 2.5761 ± 0.0940 | 85.6702% ± 0.1280% |
| Encoder activation | Shuffle digit | 51.3103% ± 0.5921% / 6.6934% ± 0.0735% / 41.9963% ± 0.5205% | 2.5761 ± 0.0940 | 85.6702% ± 0.1280% |
| Encoder activation | Shuffle sector | 51.3103% ± 0.5921% / 6.6934% ± 0.0735% / 41.9963% ± 0.5205% | 2.5761 ± 0.0940 | 85.6702% ± 0.1280% |
| Hidden activation | Baseline | 30.2743% ± 0.3828% / 53.7094% ± 0.4698% / 16.0163% ± 0.2339% | 2.9942 ± 0.0844 | 38.3718% ± 0.3466% |
| Hidden activation | Shuffle digit | 50.3564% ± 1.0297% / 32.7871% ± 1.0481% / 16.8565% ± 0.1359% | 1.3303 ± 0.0635 | 64.0272% ± 0.8468% |
| Hidden activation | Shuffle sector | 40.1362% ± 0.5727% / 41.4261% ± 0.4246% / 18.4377% ± 0.2677% | 0.4635 ± 0.0134 | 81.8317% ± 0.3688% |
| Input gate | Baseline | 75.5101% ± 0.5488% / 16.8494% ± 0.4603% / 7.6406% ± 0.1240% | 15,923.7783 ± 97.6968 | 30.3288% ± 0.1325% |
| Input gate | Shuffle digit | 96.5238% ± 0.0701% / 1.3060% ± 0.0393% / 2.1702% ± 0.0337% | 10,441.2873 ± 139.1854 | 55.1829% ± 0.5231% |
| Input gate | Shuffle sector | 25.2028% ± 0.9480% / 54.9499% ± 1.3343% / 19.8473% ± 0.4096% | 881.2302 ± 32.7129 | 95.9226% ± 0.1254% |
| Recurrent gate | Baseline | 22.1828% ± 0.4987% / 72.7907% ± 0.4667% / 5.0266% ± 0.1147% | 4,264.4144 ± 70.5233 | 30.8503% ± 0.1737% |
| Recurrent gate | Shuffle digit | 81.2008% ± 0.3816% / 12.0795% ± 0.2815% / 6.7197% ± 0.1143% | 923.2748 ± 21.8691 | 85.2064% ± 0.3056% |
| Recurrent gate | Shuffle sector | 4.2203% ± 0.2240% / 87.7793% ± 0.4019% / 8.0004% ± 0.2118% | 794.2656 ± 27.2721 | 83.2358% ± 0.4186% |

一致性检查显示 encoder 在三种 condition 下逐 seed 完全相同，符合 shuffle 只改 feedback 的
实现。Hidden 的未归一化 condition-mean variance 从 baseline 的 2.9942 降至 shuffle-digit 的
1.3303 和 shuffle-sector 的 0.4635，同时 residual 升至 64.0272% 和 81.8317%；因此三项
归一化百分比的变化不能单独解释为 factor signal 的重新分配，而是伴随 substantial total signal
collapse。

### 4.13 — 带 residual 图的统一 trial-level 四分量

本小节对应 `Fig4_core_objects_aggregate_1x4_10seed_with_residual`、
`Fig4_shuffle_activation_anova_1x3_10seed`、
`Fig4_activation_anova_1x2_6model_10seed_with_residual` 与
`Fig5_unit_gate_marginalization_1x3_with_residual`。**不替代**上文任何以
condition-mean total variance 为分母的 Sector / Digit / Interaction 结果。为使每组四柱可加和
为 100%，每个 training seed 先按
\(\eta^2_f(\mathrm{trial})=\eta^2_f(\mathrm{condition\ mean})
\,[1-\eta^2_{\mathrm{residual}}(\mathrm{trial})]\) 转换三个 factor；Residual 保持
\(SS_{\mathrm{residual}}/SS_{\mathrm{total,trial}}\)。下列均为转换后再跨 10 seeds 的 mean ±
SEM，顺序均为 Sector / Digit / Interaction / Residual。

计算的逐 seed 输入是
`Fig4_shuffle_activation_anova_long_10seed.csv` 中同一 `object × condition × seed` 的
`sector_pct`、`digit_pct`、`interaction_pct` 与 `residual_frac`（均已先对该 seed 的 20 draws
取均值），而不是跨 seed 的列均值。对每个 factor 使用
(e_{f,s}=eta^2_{f,s}(1-r_s))，再报告 (\mathrm{mean}_s(e_{f,s})\pm\mathrm{SEM}_s(e_{f,s}))。

**Fig4 core（标准 32-frame protocol；reset-excluded）**

| 对象 | trial-level 四分量（%） |
|---|---|
| Input gate | **51.0500 ± 0.4009 / 10.6400 ± 0.2883 / 4.8593 ± 0.0814 / 33.4508 ± 0.1309** |
| Recurrent gate | **14.9963 ± 0.3500 / 46.2569 ± 0.2971 / 3.2340 ± 0.0715 / 35.5128 ± 0.1373** |
| Encoder activation | **7.3495 ± 0.1306 / 0.9597 ± 0.0092 / 6.0181 ± 0.0640 / 85.6727 ± 0.1279** |
| Hidden activation | **18.0744 ± 0.2189 / 30.2601 ± 0.3587 / 9.3449 ± 0.1430 / 42.3206 ± 0.3556** |

**Fig4 shuffle hidden activation（reset-excluded）**

| feedback 条件 | trial-level 四分量（%） |
|---|---|
| Baseline | **18.6537 ± 0.2258 / 33.1041 ± 0.3847 / 9.8704 ± 0.1531 / 38.3718 ± 0.3466** |
| Shuffle digit | **18.0564 ± 0.3000 / 11.8569 ± 0.6721 / 6.0594 ± 0.1282 / 64.0272 ± 0.8468** |
| Shuffle sector | **7.2949 ± 0.1901 / 7.5272 ± 0.1754 / 3.3462 ± 0.0671 / 81.8317 ± 0.3688** |

**Fig4 six-model activation（reset-excluded）**

| 对象 | 模型 | trial-level 四分量（%） |
|---|---|---|
| Input activation | GaWF | **7.3495 ± 0.1306 / 0.9597 ± 0.0092 / 6.0181 ± 0.0640 / 85.6727 ± 0.1279** |
| Input activation | RNN | **8.6564 ± 0.1992 / 0.7183 ± 0.0078 / 4.2258 ± 0.0416 / 86.3995 ± 0.1797** |
| Input activation | LSTM | **8.9365 ± 0.2598 / 0.6709 ± 0.0089 / 3.9423 ± 0.0445 / 86.4504 ± 0.2677** |
| Input activation | GRU | **6.2625 ± 0.2098 / 0.7120 ± 0.0120 / 4.3528 ± 0.0716 / 88.6727 ± 0.2661** |
| Input activation | S5 | **7.2094 ± 0.1849 / 0.7138 ± 0.0107 / 4.1431 ± 0.0695 / 87.9337 ± 0.2511** |
| Input activation | Mamba | **9.3009 ± 0.1466 / 0.7341 ± 0.0042 / 4.4412 ± 0.0297 / 85.5239 ± 0.1452** |
| Hidden activation | GaWF | **18.0744 ± 0.2189 / 30.2601 ± 0.3587 / 9.3449 ± 0.1430 / 42.3206 ± 0.3556** |
| Hidden activation | RNN | **19.7689 ± 0.1459 / 21.0204 ± 0.1960 / 9.3569 ± 0.0974 / 49.8538 ± 0.2666** |
| Hidden activation | LSTM | **26.5929 ± 0.1979 / 23.6142 ± 0.2020 / 13.6374 ± 0.1581 / 36.1555 ± 0.1831** |
| Hidden activation | GRU | **22.4459 ± 0.1548 / 26.6082 ± 0.2710 / 8.4930 ± 0.1737 / 42.4529 ± 0.2610** |
| Hidden activation | S5 | **16.9490 ± 0.2118 / 10.3688 ± 0.2956 / 8.7117 ± 0.1056 / 63.9705 ± 0.1813** |
| Hidden activation | Mamba | **23.9547 ± 0.1310 / 25.9744 ± 0.1607 / 9.6666 ± 0.0849 / 40.4043 ± 0.1410** |

**Fig5 unit gates（reset-excluded；GaWF 为 destination-unit projection）**

| 模型 | Gate | trial-level 四分量（%） |
|---|---|---|
| GaWF | Input | **54.7013 ± 1.9496 / 6.1058 ± 1.9180 / 1.2101 ± 0.0392 / 37.9827 ± 0.7102** |
| GaWF | Recurrent | **7.6242 ± 1.0641 / 58.3621 ± 1.2990 / 1.4509 ± 0.0465 / 32.5628 ± 0.3322** |
| LSTM | Input | **16.7857 ± 0.5232 / 24.0071 ± 0.4078 / 5.1136 ± 0.0905 / 54.0935 ± 0.6888** |
| LSTM | Forget | **19.1374 ± 0.3950 / 7.3878 ± 0.1853 / 4.0148 ± 0.0524 / 69.4600 ± 0.2825** |
| LSTM | Output | **36.6962 ± 0.5795 / 13.2659 ± 0.3764 / 5.7204 ± 0.1302 / 44.3175 ± 0.4563** |
| GRU | Reset | **20.3446 ± 0.3185 / 8.1163 ± 0.1856 / 7.2584 ± 0.1389 / 64.2807 ± 0.3945** |
| GRU | Update | **20.1538 ± 0.4549 / 14.6287 ± 0.4201 / 4.1393 ± 0.0908 / 61.0782 ± 0.5526** |

## §5 — 输入门的空间组织与符号盲性

本节只做定性分析，不增加过多定量统计检验。仅保留检验 sign-blindness 所必需的 5.9
seed-level descriptive overlap gaps、slopes 与 levels；不报告 pooled connection-level p values。

| ID | 需要的统计值 | 具体值（10-seed） |
|---|---|---|
| 5.9a | matching sources overlap gap，九个 sectors 汇总 | **剔除 reset frame 后**：W+ mean Δg **+0.2756 ± 0.0053**；W− **+0.2891 ± 0.0052**；overlap gap W+−W− **−0.0135 ± 0.0018**；n=10。 |
| 5.9b | other sources overlap gap | **剔除 reset frame 后**：W+ mean Δg **−0.0345 ± 0.0007**；W− **−0.0361 ± 0.0007**；overlap gap **+0.0017 ± 0.0002**；n=10。 |
| 5.9c | 分符号 slope 与 SEM，matching / other 分开 | 在各 seed 自己的 shared-\|W\| band 内拟合 OLS `Δg ~ \|W\|`，**剔除 reset frame 后**。Matching：W+ slope **−0.0429 ± 0.0060**，W− **+0.0073 ± 0.0027**。Other：W+ **+0.0054 ± 0.0008**，W− **−0.0009 ± 0.0003**；各 n=10。 |
| 5.9d | matching / other 的 Δg 分箱均值水平 | **剔除 reset frame 后**，跨各组全部 connections 与 9 sectors 的 per-seed overall Δg：Matching **+0.2825 ± 0.0052**；Other **−0.0353 ± 0.0006**；各 n=10。图中的曲线仍是 binned mean ± SEM，仅作定性展示。 |

## §6 — 循环门的符号依赖调制

6.3–6.6 使用 sjc-remote formal 10-seed `fig7_seed_level_summary.npz`。每个 seed 先在该
seed 自己的 positive/negative shared-|W| overlap band 内计算 unique-connection mean，gap 定义为
$\Delta g_{W>0}-\Delta g_{W<0}$，再跨 training seeds 计算 SEM。下表沿用 efferent
`src→dst` TT/TR/RT/RR 口径，不使用 afferent companion。

6.13 在每个 seed 内对 45 个 digit pairs 计算 normalized overlap
$|T_a\cap T_b|/\sqrt{|T_a||T_b|}$，然后先对 pairs 取均值、再以 10 个 training seeds 为
独立单位计算 SEM。其 independent-set chance baseline 在每个 seed 内固定 observed $|T_a|$、
$|T_b|$，从同一 FDR-eligible 且非 interaction-dominant pool $E$ 独立均匀抽取，故为
$\sqrt{|T_a||T_b|}/|E|$。6.15 直接使用 Figure 7 已保存的 group-specific shared-|W| overlap-band
seed cell means；在 sign-gap 上做 `group × variable` repeated-measures ANOVA，等价于原始
cell means 上的 `group × sign(W) × variable` interaction。6.11 在每个 seed、variable 和
TT/TR/RT/RR group 自己的 positive/negative shared-|W| overlap band 内分别拟合 W+、W−
的 OLS `Δg ~ |W|`，再跨 10 个 training seeds 计算 SEM；overall level 不限制 overlap band。

6.17 对每个 seed、Digit 或 Sector context 逐帧计算实际循环电流
$I_{group}(c)=\sum_{(i,j)\in group}\langle g_{ij}(t)W_{ij}h_j(t-1)\rangle_{t\in c}$，并以
$\Delta I(c)=I(c)-\operatorname{mean}_{c'}I(c')$ 定义 context delta。门的瞬时贡献为
$\Delta I^{gate}(c)=\sum\langle(g_{ij}(t)-\bar g_{ij})W_{ij}h_j(t-1)\rangle_{t\in c}$；其中
$\bar g_{ij}$ 是同一分析内全部条件均值的等权平均（Digit 为 10、Sector 为 9），实际
$h_j(t-1)$ 保持不变。TT/TR/RT/RR 都保留
diagonal、按 `sign(W)` 拆为 E/I，并除以该组目的地单元数。它是**瞬时分解**，不是冻结 gate 后的
完整反事实；每个 32-frame window 的 `t=0` 在 equal-n selection 前剔除。

6.18 使用同一逐帧定义，但改为 **per nonzero recurrent connection**：`W > 0` 与 `W < 0`
各自除以该 sign 的连接数；total 则以两个 sign 的连接数加权，
$\Delta I^{gate}_{\mathrm{total}}=(N_+\Delta I^{gate}_+ + N_-\Delta I^{gate}_-)/(N_+ + N_-)$，
**不是**两个 sign-specific 均值的直接相加。每个 seed 先跨 Digit（10）或 Sector（9）条件平均，
再跨 10 个 training seeds 计算 mean ± SEM。

| ID | 需要的统计值 | 具体值（10-seed） |
|---|---|---|
| 6.3–6.5 | 四组 × 两变量符号缺口完整表，delta 版 | **剔除 reset frame 后**。Digit：TT **+0.1097 ± 0.0058**；TR **−0.0702 ± 0.0036**；RT **−0.0143 ± 0.0026**；RR **+0.0071 ± 0.0006**。Sector：TT **−0.0130 ± 0.0033**；TR **−0.0132 ± 0.0016**；RT **−0.0097 ± 0.0006**；RR **+0.0035 ± 0.0002**。各 n=10。 |
| 6.6 | 同表 W>0、W<0 各自 Δg 水平 | **剔除 reset frame 后**。Digit：TT W+ **−0.2514 ± 0.0088**，W− **−0.3611 ± 0.0072**；TR **−0.1396 ± 0.0070 / −0.0695 ± 0.0041**；RT **−0.0142 ± 0.0020 / +0.0002 ± 0.0028**；RR **+0.0180 ± 0.0009 / +0.0109 ± 0.0004**。Sector：TT **−0.1020 ± 0.0037 / −0.0889 ± 0.0034**；TR **−0.0871 ± 0.0026 / −0.0739 ± 0.0031**；RT **+0.0063 ± 0.0006 / +0.0160 ± 0.0009**；RR **+0.0101 ± 0.0003 / +0.0066 ± 0.0003**。每对均为 W+ / W−，各 n=10。 |
| 6.11a | 分符号 slope 与 SEM，Digit/Sector × TT/TR/RT/RR 分开 | 在各 seed 自己的 shared-\|W\| overlap band 内拟合、**剔除 reset frame 后**。Digit：TT W+ **+0.1285 ± 0.0161**，W− **−0.0210 ± 0.0144**；TR **+0.0386 ± 0.0043 / +0.0634 ± 0.0031**；RT **+0.0352 ± 0.0032 / +0.0211 ± 0.0017**；RR **−0.0082 ± 0.0008 / −0.0099 ± 0.0006**。Sector：TT **+0.0042 ± 0.0042 / +0.0032 ± 0.0020**；TR **+0.0169 ± 0.0022 / +0.0097 ± 0.0007**；RT **−0.0076 ± 0.0020 / +0.0089 ± 0.0010**；RR **+0.0004 ± 0.0003 / −0.0022 ± 0.0002**。每对均为 W+ / W−，各 n=10。 |
| 6.11b | 八组的 overall Δg level，对齐 5.9d | **剔除 reset frame 后**，跨各组全部 connections 与 contexts 的 per-seed overall Δg（不限制 overlap band）：Digit TT **−0.3098 ± 0.0078**；TR **−0.0941 ± 0.0050**；RT **−0.0046 ± 0.0022**；RR **+0.0136 ± 0.0006**。Sector TT **−0.0952 ± 0.0032**；TR **−0.0790 ± 0.0028**；RT **+0.0124 ± 0.0008**；RR **+0.0079 ± 0.0003**。各 n=10。 |
| 6.12 | hidden size、实际 \|T\|、diagonal 处理 | **H=256**。T 是每个 seed 中 FDR-eligible 且非 interaction-dominant units 的 top 10%，用 `ceil(0.1 × eligible)`；seeds 1–10 的实际 \|T\| 为 **[25, 24, 24, 24, 24, 24, 24, 24, 23, 25]**，同一 seed 的所有 Digit/Sector contexts 数量相同。formal 分组**保留 diagonal `i=j`**，所以 TT 候选规模为 \|T\|²（分别为 625、576 或 529，之后仍应用 `weight != 0`）。 |
| 6.13 | 不同 digit 的 T 集合重叠度 | **剔除 reset frame 后**。每个 seed 的 10 个 digit `T` masks 来自同一 reset-excluded compact cache；45 个 digit pairs 的 normalized overlap 先在 seed 内取均值，再跨 seeds 汇总为 **12.14% ± 0.20%**（SEM，n=10）。固定各 seed 的 observed $|T|$、从其 FDR-eligible non-interaction pool 独立抽取的 chance baseline 为 **10.22% ± 0.04%**，故 observed − chance 为 **+1.93 ± 0.20 percentage points**。45 个 digit-pair 的跨-seed 均值范围为 **3.35%–21.51%**（最低 digit 0–1；最高 digit 4–6），保留了明显的 pair heterogeneity；所有 seed 内的 $|T|$ 分别固定为 23、24 或 25。 |
| 6.15 | `group × sign(W) × variable` interaction test | **剔除 reset frame 后**，在 10-seed shared-\|W\| sign-gap cells 上做 `group × variable` repeated-measures ANOVA（等价原始 cell means 的三因素 interaction）：**F(3, 27) = 479.4657，p = 1.62 × 10⁻²³**。 |
| 6.17a | 净循环贡献，Digit：`I`、`ΔI` 与 `ΔI^gate` | **剔除 reset frame 后**；下列 `I / ΔI^gate` 为每 seed 先跨十个 digits 平均、再作 10-seed mean ± SEM，顺序均为 E / I / total（每目的单元）。TT：`I` **+0.2270 ± 0.0150 / −0.1384 ± 0.0092 / +0.0885 ± 0.0114**；`ΔI^gate` **−0.1477 ± 0.0072 / +0.7385 ± 0.0330 / +0.5908 ± 0.0336**。TR：**+0.1418 ± 0.0051 / −1.1770 ± 0.0586 / −1.0352 ± 0.0543**；**−0.0739 ± 0.0033 / −0.0017 ± 0.0094 / −0.0756 ± 0.0094**。RT（R→T）：**+0.4781 ± 0.0213 / −0.9206 ± 0.0417 / −0.4424 ± 0.0273**；**−0.2545 ± 0.0121 / +1.7394 ± 0.0617 / +1.4850 ± 0.0572**。RR：**+0.5565 ± 0.0151 / −2.4798 ± 0.0699 / −1.9232 ± 0.0646**；**−0.1961 ± 0.0067 / +0.5801 ± 0.0172 / +0.3841 ± 0.0154**。`ΔI` 的跨 digit 平均按定义为 0；其十个 digit 的 10-seed mean 范围（E / I / total）是 TT **−0.0848–+0.2739 / −0.1376–+0.0610 / −0.0388–+0.1363**，TR **−0.0160–+0.0295 / −0.2591–+0.4644 / −0.2587–+0.4933**，RT **−0.1355–+0.4630 / −0.7879–+0.2758 / −0.3248–+0.1695**，RR **−0.0666–+0.0476 / −0.3636–+0.0993 / −0.4303–+0.1278**；各 digit 的 mean ± SEM 及所有 seed-level 值见同一 long CSV 与 Supple4。T→T 的 I 正贡献（减少负电流）大于 E 负贡献，且十个 digits 的总 `ΔI^gate` 均为正（**+0.0769–+0.8819**，最小 `mean−SEM=+0.0439`），故支持瞬时 **disinhibition**。R→T 亦不可忽略：总 `ΔI^gate` 十个 digits 均为正（**+1.0523–+1.8893**，最小 `mean−SEM=+0.9391`）。 |
| 6.17b | 净循环贡献，Sector：`I`、`ΔI` 与 `ΔI^gate` | **剔除 reset frame 后**；每 seed 先跨九个 sectors 平均、再作 10-seed mean ± SEM，顺序均为 E / I / total（每目的单元）。TT：`I` **+0.2570 ± 0.0205 / −0.2478 ± 0.0158 / +0.0092 ± 0.0269**；`ΔI^gate` **−0.1183 ± 0.0056 / +0.1675 ± 0.0055 / +0.0491 ± 0.0078**。TR：**+0.1204 ± 0.0042 / −0.4924 ± 0.0211 / −0.3720 ± 0.0193**；**−0.0654 ± 0.0024 / +0.1944 ± 0.0067 / +0.1290 ± 0.0052**。RT（R→T）：**+0.5769 ± 0.0209 / −2.4396 ± 0.0986 / −1.8627 ± 0.0950**；**−0.2290 ± 0.0106 / +0.5085 ± 0.0299 / +0.2794 ± 0.0213**。RR：**+0.5603 ± 0.0148 / −3.0418 ± 0.0980 / −2.4815 ± 0.0919**；**−0.1958 ± 0.0064 / +0.5430 ± 0.0187 / +0.3472 ± 0.0165**。`ΔI` 的跨 sector 平均按定义为 0；九个 sector 的 10-seed mean 范围（E / I / total）是 TT **−0.1096–+0.3676 / −0.0934–+0.0621 / −0.1846–+0.3999**，TR **−0.0169–+0.0445 / −0.1190–+0.1131 / −0.1283–+0.1576**，RT **−0.0781–+0.1978 / −0.5872–+0.3352 / −0.3894–+0.2625**，RR **−0.0805–+0.0221 / −0.0941–+0.1772 / −0.0964–+0.1395**。Sector TT 的 E 项始终为负、I 项始终为正，但总 `ΔI^gate` 依 sector 变号（**−0.0350–+0.0912**；最小 `mean−SEM=−0.0554`），故其跨 sector 平均的正净贡献不能写成“每一 sector 均 disinhibition”。相反，RT 总 `ΔI^gate` 九个 sectors 均为正（**+0.0868–+0.3800**，最小 `mean−SEM=+0.0248`）。所有 condition-level mean ± SEM 与 seed-level 值见 sector long CSV 与 sector Supple4。 |
| 6.18 | Fig8 的 per-connection `ΔI^gate` 四个正文填空 | **剔除 reset frame 后**；每 seed 先跨条件平均、再做 10-seed mean ± SEM，单位均为 per nonzero recurrent connection。Digit TT：`W > 0` **−0.01339 ± 0.00063**（`[VAL-8a]`）；`W < 0` **+0.05628 ± 0.00260**（`[VAL-8b]`）；total **+0.02461 ± 0.00149**（`[VAL-8c]`）。Sector TT total **+0.00200 ± 0.00032**（`[VAL-8d]`）。方向与正文预期一致：两种条件下 W>0 均为负、W<0 均为正，total 为正；sector total 约为 digit total 的 **8.1%**。各 n=10。 |
| 6.16 | §6 10-seed 完成状态 | **reset-excluded 10-seed 完成：6.3–6.6、6.11、6.13、6.15、6.17 与 6.18（含 Fig7/Supple3、Fig8 与 Supple4 图）。** 6.9 已按用户要求删除，不计入完成状态。 |

## 附 — 已删除或降级主张的当前处理

| 位置 | 原主张 | request 中的现状 | 本记录处理 |
|---|---|---|---|
| §3 末 / 转场 T3 | context 改变哪些突触打开，但不改变打开多少 | 已删除 | 不恢复；本记录只报告 3.1、3.2、3.6。 |
| §3 第 2 段 | 27% 通过 / 73% 清零、norm ratio 0.55 | 已删除 | 不恢复；这些不是本次请求的保留统计。 |
| §5 第 4 段 | 两条曲线跨 \|W\| 平行 | 附条件保留，5.9c 已完成 | matching 与 other 中正负曲线 slopes 均不同方向，因此不支持“平行”这一无条件表述。 |
| §6 第 1 段 | 四组具体 delta 值 | 已完成 6.3–6.5 | 已由 sjc-remote formal 10-seed summary 补齐，见 6.3–6.5。 |

## 已使用的数据源

- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig1_reset_excluded_behavior_6model_10seed_v8/final/reset_excluded_test_accuracy_10seed.csv`
  （§1 formal test accuracy；每个 32-frame window 排除 `t=0`）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig1_target_switch_recovery_resetexcluded_6model_10seed_v4/`
  （Fig1 formal recovery；独立 joint-balanced test dataset、每个 32-frame window 排除 `t=0`）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/supple1_feedback_ablation_resetexcluded_10seed_v5/`
  （Supplementary 1 formal recovery；每个 512-frame window 排除 `t=0`）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig4_shuffle_activation_anova_10seed/final/Fig4_shuffle_activation_anova_long_10seed.csv`
  （§2 formal baseline / shuffle-digit / shuffle-sector accuracy；§4.12、512-frame、排除 `t=0`）
- `results/save_data/fig3/seed*/gawf_gate_distribution_meta.json`
- `results/save_data/fig3/seed*/gawf_gate_distribution_stats.npz`
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig3_gate_half_mass_10seed/fig3_gate_half_mass.json`
  （§3.5 十个 seed 的 0.5 点质量、reset 来源、点质量剔除后的中间占比与 `U` row audit）
- `results/save_data/fig5/unit_gate_context_variance_multiseed.json`
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig4_activation_anova_6model_10seed_residual_resetexcluded/gawf-seed*/activation_anova.npz`
  （§4.1–4.3、Fig4 core activation；32-frame、排除 `t=0`）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig4_gate_synapse_anova_resetexcluded_10seed/seed*/gate_synapse_anova.npz`
  （§4.4–4.5 raw-synapse gate；32-frame、排除 `t=0`、20-draw balanced ANOVA）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig6_encoder_patterns_resetexcluded_gawf_10seed/`
  （Fig6 encoder Sector/Digit patterns；32-frame、每 seed 排除 1,799 个 `t=0`，随后做 equal-n selection）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig7_recurrent_gate_reset_excluded_10seed_r3/final/fig7_seed_level_summary.npz`
  （§6.3–6.6 与 §6.15 的 reset-excluded seed-level sign gaps）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig7_recurrent_gate_resetexcluded_stats_r3/supple3_seed_level_sign_magnitude_stats.json`
  （§6.11 的 10-seed 分符号 slopes、SEM 与 overall delta levels）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig7_recurrent_gate_reset_excluded_10seed_r3/seed*/compact/recurrent_gate_condition_means.npz`
  （只读计算 tuned-mask counts、hidden size 与 6.13 observed digit-pair overlap）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig7_recurrent_gate_10seed/seed*/selectivity/part1_selectivity.npz`
  （6.13 的每 seed FDR-eligible、non-interaction-dominant pool，用于 conditional independent-set chance baseline）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig6_net_recurrent_current_resetexcluded_10seed_r2/seed*/net_recurrent_current.npz`
  与 `final/net_recurrent_current_10seed_long.csv`、`final/net_recurrent_current_10seed_summary.npz`
  （§6.17 的 10-seed × digit × group × sign `I`、`ΔI`、`ΔI^gate`；逐帧实际
  `g·W·h(t−1)`，reset-excluded、按目的地单元数归一）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/fig6_net_recurrent_current_sector_resetexcluded_10seed/seed*/net_recurrent_current.npz`
  与 `final/net_recurrent_current_10seed_long.csv`、`final/net_recurrent_current_10seed_summary.npz`
  （§6.17 的 10-seed × sector × group × sign `I`、`ΔI`、`ΔI^gate`；逐帧实际
  `g·W·h(t−1)`，reset-excluded、按目的地单元数归一）
- sjc-remote：
  `results/save_data/fig8/recurrent_current/connection/{digit,sector}/`
  `net_recurrent_current_connection_10seed_long.csv` 与
  `Fig8_recurrent_current_connection_caption_stats.json`
  （§6.18：从 §6.17 retained current means 转为 per nonzero recurrent connection；`W > 0` /
  `W < 0` 各按自身连接数平均，total 按全部非零连接数加权）
- sjc-remote：
  `/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/supple2_input_gate_sign_magnitude_9sector_10seed/`
  （10-seed × 9-sector §5.9 compact results 与 seed-level stats）

以上远端**数值文件**均只在 SSH 会话内读取，未复制、下载或同步到本地。本次已用本地已有
curated raw data 重绘 Fig2 与 Fig5；并用远端已有 10-seed 结构化结果重绘 Fig1、Supple1、
Fig4 activation、Fig4 core objects、Fig6 encoder maps、Fig7、Supple2、Supple3 与 Supple4。回传到
`results/save/` 的仅为正式图文件（Fig 为 PDF，Supplementary 为 PNG），未回传数值文件。

## 历史对照：未排除每个 window 的 t=0 的 accuracy（不作为正式结果）

以下数值保留以便和旧图/旧文字比对，**不得作为正文正式 accuracy 引用**。它们没有剔除每个
recurrent window 的 `t=0`；而同一帧的 gate 是 zero-feedback artifact 的 0 值，故它们与 gate
统计口径不一致。loss curves 未受这一替换影响。

| 原统计 | 未排除 t=0 的历史值（10-seed mean ± SEM） |
|---|---|
| 旧 §1.2 GaWF canonical test | Digit **85.8765% ± 0.1466%**；Sector **92.7826% ± 0.1199%**。 |
| 旧 §1.3 RNN / LSTM / GRU / S5 / Mamba canonical test | RNN：**80.1815% ± 0.1721% / 90.8102% ± 0.0870%**；LSTM：**79.7103% ± 0.2268% / 90.3591% ± 0.0637%**；GRU：**78.7389% ± 0.1708% / 90.3415% ± 0.0568%**；S5：**74.6588% ± 0.3558% / 88.7969% ± 0.1787%**；Mamba：**82.4159% ± 0.1635% / 92.0645% ± 0.0718%**；顺序均为 Digit / Sector。 |
| 旧 §2.2 shuffle Digit | Baseline **85.8765% ± 0.1466%**；shuffle-sector **54.1572% ± 0.8730%**；shuffle-digit **73.3829% ± 0.4910%**。 |
| 旧 §2.3 shuffle Sector | Baseline **92.7826% ± 0.1199%**；shuffle-sector **62.9858% ± 0.7528%**；shuffle-digit **91.7324% ± 0.1464%**。 |
