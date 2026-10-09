<!-- ==================== Layer 1 · 叙事报告 ====================
     本文件名：SVCO_20261008_narrative.md
     本文件在5层体系中的位置：第2个文件（总表见 SVCO_20261008_intelligence_brief.json 的 file_manifest 字段）
     其余4个文件：Layer0=SVCO_20261008_intelligence_brief.json(情报头文件（总表）) / Layer2=SVCO_20261008_clusters_tiered.csv(三层分级事件表) / Layer3=SVCO_20261008_events_material.csv(精选事件明细表) / Layer4=SVCO_20261008_machine_input.json(机器输入文件)
     本文件目录：执行摘要（模板化生成，引用direction_signal数值） / S1 时序操作地图（T1层事件按时间排序） / S2 冰山地图（单个冰山候选Top10+分区量） / S3 对倒分析（成交量对价格驱动效率） / S4 流动性真空异常地图（Liquidity Void事件明细）
     本文件内容概述：机构研报格式：执行摘要/时序操作地图/冰山地图/对倒分析/流动性真空异常地图。
-->

# SVCO · 2026-10-08 · 盘口情报叙事报告

> 📌 **一句话结论**：方向=**NEUTRAL·LONG_BIAS**（置信度0.37）。阶段打分块已从正式输出中移除，具体阶段由AI结合盘口证据自行判断。

---

## Section 1：时序操作地图（核心 · 最先读）

| 时间(ET) | 价位 | 行为类型 | 家族 | 量 | ret_300s(bps) |
|----------|------|---------|------|-----|-----|
| 11:40 | $8.0 | FRAG_ICE+ICE+RELOAD+RESERVE_B_LOW_1540_1540 | FRAG_ICE+ICE+RELOAD+RESERVE | 23326 | 162.5 |
| 11:40 | $8.0 | FRAG_ICE+RESERVE_A_LOW_1540_1540 | FRAG_ICE+RESERVE | 34344 | 168.86 |
| 15:55 | $8.01 | CANCEL_A_LOW_1955_2000 | CANCEL | 25195 | 75.09 |
| 15:55 | $7.98 | CANCEL_B_LOW_1955_2000 | CANCEL | 31042 | 75.09 |

---

## Section 2：冰山地图（单个冰山候选 · 高/中/低位区间）

> 冰山候选 = ICE(分批成交) + RESERVE(原生补单) + RELOAD(算法补单) + FRAG_ICE(碎片化冰山)，top10单位为单个候选事件/单条冰山链，不是微簇合并量；合计 **52笔 / 84329股**

| 区 | 笔数 | 冰山量 | 占比 |
|----|------|--------|------|
| HIGH | 0 | 0 | 0.0% |
| MID | 10 | 5877 | 7.0% |
| LOW | 42 | 78452 | 93.0% |

**前十大冰山（按单个候选事件量排序）**：

| # | candidate_id | 家族 | 方向 | 价位 | 量 | 区 | 时间 |
|---|------|------|------|-----|-----|----|------|
| 1 | 2214df2f8f85 | FRAG_ICE | A | $8.0 | 30931 | LOW | 11:40 |
| 2 | 2babe6591a77 | FRAG_ICE | B | $8.0 | 10183 | LOW | 11:40 |
| 3 | aaf02fe6c902 | FRAG_ICE | A | $8.0 | 5299 | LOW | 15:58 |
| 4 | f1eff575b9fe | FRAG_ICE | B | $8.02 | 4952 | LOW | 11:40 |
| 5 | 37a9c64c064f | FRAG_ICE | B | $8.0 | 4303 | LOW | 15:57 |
| 6 | 257a2cfb4b68 | ICE | B | $8.02 | 3200 | LOW | 11:40 |
| 7 | 6d254f750f26 | FRAG_ICE | A | $8.02 | 1708 | LOW | 11:40 |
| 8 | ed97d2975249 | FRAG_ICE | A | $8.0 | 1218 | LOW | 15:55 |
| 9 | a8b234369686 | FRAG_ICE | A | $7.96 | 1186 | LOW | 15:48 |
| 10 | 797c4d9a1012 | FRAG_ICE | B | $8.16 | 1048 | MID | 11:48 |

---

## Section 3：滚动分层谎骗（Rolling Layering）

> rolling_layer_spoof = 同侧贴盘大单撤销后，在相邻价位恰好±1 tick快速重挂，规模相近，chain_length≥3；合计 **0笔 / 0股**。

- 本日无 Rolling Layering 候选事件（count=0，非漏检，检测器已运行）。

---

## Section 4：对倒分析（对敲/换手）

> 对倒 = MATCHED（同价同量买卖500ms内配对），合计 **2笔 / 4718股**，主导区=LOW

| 区 | 笔数 | 对倒量 |
|----|------|--------|
| HIGH | 0 | 0 |
| MID | 0 | 0 |
| LOW | 2 | 4718 |

> ⚡ 主导区(LOW)对倒驱动效率=275.08bps/万股，对照全场方向单中位数=424.2bps/万股，洗量比=1.5x

---

## Section 5：流动性真空异常地图（Liquidity Void）

> **Liquidity Void**（流动性真空异常：日高/日低 P99 被动墙被连续主动单吃掉，价格位移≥5tick，且后续满足价差扩张确认），合计 **0笔 / 0股**

- 本日无 Liquidity Void 异常事件（count=0，非漏检，检测器已运行）。

<!-- 报告结束 · 华生叙事层 · 由generate_five_layer_report.py全自动生成 -->
