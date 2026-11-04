# Round 17: 过度防御问题 — 回声去重 (2026-11-03)

## GLM 判定

**核心论点**：问题不是"防御太多"，而是"同一防御重复太多次"（回声）。修法：去重（安全），不是删内容（危险）。

**架构核查（通过）**：两大承重发现（matched-FT CLIP 8/8 全胜、zero-shot ≤ majority baseline）在全文任何位置都以无限定直陈句出现。防御密度全部落在二线结论上——正确设计。

**四份外审证据**：无一抱怨过度防御。Round 14 反而要求更多披露。审稿人反感的是隐藏的局限和不受支持的主张，不是可见的限定。

## 回声计数

| 同一 caveat | 次数 |
|---|---|
| "matched-FT arm not replicated (Limitations, item 8)" | 6 次 + Limitation 8 本体 |
| "cannot resolve / within the test set's noise" (1.5pp) | ~6 次 |
| "descriptive stability, not powered significance" | 5 次 |
| "single-seed estimate" (C-selection) | 5 次 |
| K=1 平局 40.1/40.4 | 5 次 |

**规则**：每条 caveat 最多三次——定义处全文陈述、使用处指针、结论处一次。

## 手术清单（12 处）

| # | 位置 | 动作 |
|---|---|---|
| 1 | 番茄/玉米表注 | 删 "matched FT arm not replicated here" |
| 2 | Datasets 跨作物句 | 删第 2 次声明 |
| 3 | RQ3 answer | 第 4 次 "(Limitations, item 8)" 改纯指针 |
| 4 | RQ3 answer (ii) 巨句 | K=1 故事 3 遍→3 短句，删复述 |
| 5-6 | RQ2 (1)(2) | "cannot resolve" 第 2 次改 "the residual gap (above)" |
| 7 | RQ2 answer | "descriptive stability rather than powered significance" → "(descriptive; sec:cv)" |
| 8 | RQ2 (1) | "(single-seed; Methods)" 删括号 |
| 9 | Methods matched-budget | 删尾 "which we do not measure" |
| 10 | Intro finding (4) | 删 "reported as a characterization, not a new method" |
| 11 | 全文 | "baseline-tuning-dependent" 与 "governed by the baseline's tuning budget" 统一 |
| 12 | 拆段 | **Round 16 已完成，跳过** |

## 不可动清单 (armor)

10 条 Limitations、功效分析与 p_d bracket、pre-registration 声明、reporting convention、TPDD provenance 与 split-integrity 披露、feature-store unification、CoOp superseded protocol note、各表 provenance 脚注。

## 红线

从这一轮起方向反转：继续往"更少防御"方向走才会真正伤害论文。删任何 armor 条目 = 重新打开已关闭的异议。
