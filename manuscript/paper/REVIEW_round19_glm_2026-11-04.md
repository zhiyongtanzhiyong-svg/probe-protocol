# 第十九轮（外部）：修回稿终检 — GLM

> 判定：两个补实验都跑了、结果诚实、级联更新基本到位——但抓到 3 处过期数字（其中一处自相矛盾）、1 个新引入的数值错误、1 条 Minor 未做，以及整个修回包缺了最关键的交付物：response letter。约 1 小时可全部清零。

## 一、两个新实验的核验（通过，含交叉锁定）

**M3（C-sweep 15 seeds）**：新数字链自洽——DFN 增益 +2.0→+15.1、LAION +0.7→+8.5，逐 K 验算吻合（K=10 DFN 56.4→71.5 = +15.1 = 上界 ✓）；对 matched ResNet 的间隙全部落在 ±0.1pp 报告约定内。关键的处理是对的：单种子 +6.0pp 反转不复现，正文明写 "does not reproduce"、摘要降级为 "non-significant near-tie (no probe lead)"、Finding 1/Fig 1/结论/摘要五处同步。这是 M3 的完整兑付——审稿人要的二选一，你们选了更难的那条并且结果不利也没有藏。

**M2（ResNet 低 LR 扩展）**：15-config union（9+6）✓；K=1/2 不变、K=5/10 移到 lr=1e-4/300 epochs，68.1±4.1 / 76.7±3.8（+3.2/+2.1 算术 ✓）。而且两个实验形成了一个漂亮的互锁：C-selected LAION K=5 的名义显著领先（+3.2pp, p=0.032）恰好被扩展 ResNet 的 68.1% 中和成平局（68.2 vs 68.1）——诚实报告的典范。框架采用了审稿人的选项(i)（探索性、协议外），级联被正确限制。

**M4**：Appendix D 干净兑现，描述性报告 + "not comparable" 声明 + "no matched CLIP fine-tune on these crops" 的诚实注脚 ✓。
**Minor 1**：power 括界已改用同协议数字（86.2/87.7），重算后 5.8–12.2pp 验算通过 ✓。
**M5**：四处（摘要/结论/sec:field/Finding 5）全部软化 ✓。
**M1/Minor 5**：新标题 + Discussion 边界框架 + cover letter 协议可比性论证段 ✓。

## 二、必须修的（P0，全部是级联漏网）

| # | 位置 | 问题 | 修法 |
|---|---|---|---|
| 1 | Limitation (5) | 仍写 "The C-grid sweep is limited to K=1/2; the K=5/10 frozen-probe contrasts use default C=1"——与新实验直接矛盾（Methods 明写 15 seeds 全 K 扫描）。这正是我上轮预警的级联点，漏网 | 重写为现状："C-selected 探针对比按名义 p 值报告、在预注册 16-contrast 家族之外；K=5/10 家族成员仍为 default C=1" |
| 2 | Practical Implications 2 | "improves it by up to +10.1 pp at K=2" 是旧 seed-0 数字。新 15-seed 下 K=2 增益为 +8.4/+4.2，最大 few-shot 增益在 K=10（+15.1） | 改为 "up to +15.1 pp (K=10)" |
| 3 | Related Work 末句 | "95.7% for LoRA"——LoRA 在 Table 2 是 93.5%，95.7 在全文无任何出处。这是修回新引入的数字错误 | 改 93.5% |
| 4 | Minor 2 未做 | 田间数据来源声明缺失（采集者/地点/协议），而这直接连着审稿人问题 3。正文 "collection protocol" 指涉还被删了——因为本来就没有 | 一句话补齐：采集者（X.L. 所在的 Meizhou 烟草公司是天然解释——这同时回答了"为何田间贡献未体现在作者贡献里"）、地点、季节、许可；Author Contributions 给 X.L. 加 field data curation |

## 三、修回包的结构性缺口（P1）

**缺 point-by-point response letter**——这是修回提交的核心交付物，现在整个包里没有。骨架现成：M1–M5 逐条引用审稿人原文 + 指向修改位置；M2/M3 各附一个小表，给出重算后的 margins（审稿人问题 1 的直接答案）：扩展 ResNet 下 DFN K=5 +12.2→+9.0、LAION +10.9→+7.7；K=10 DFN +10.5→+8.4、LAION +4.7→+2.6——注明 LAION K=10 的 2.6pp 仍高于 15-seed 检测限（2.1pp）但已接近。Minor 4（统计搬家）用既有立场回应 + 指出 Evaluation 已分块。

**Cover letter 需要重写成修回信**：现在仍是首次投稿措辞（"pleased to submit for consideration"）、日期仍是 October 14, 2026（round 16 标记过，至今没改）、Declaration 4 还说 "anonymized repository" 而 DAS 已改为公开 GitHub 仓库——三处同步。另外信中 Finding 2 那句的臂切换仍然无声（frozen-probe 对 matched ResNet → 突然变成 fine-tuned CLIP 对 fine-tuned ResNet），建议加 "of both backbones" 类的显式标记。

## 四、次要（P2）

1. 摘要约 250 词边缘（压缩后 +15 词），复数一遍；超了就删 "(no probe lead)"（与 "near-tie" 重复）。
2. Limitation (10) "the K≥5 margins ... are correspondingly understated on the ResNet side" 措辞易读反（实际方向：ResNet 侧被低估 → 报告的 margin 是上界）。改为 "the ResNet side of the K≥5 margins is understated, so the reported margins are upper bounds"。
3. 仓库侧两个确认项：`n_seeds_ge_clip→n_seeds_above_clip` 改名落实；`probe_cgrid_15seeds_20261104.json` 数字抽查——特别是 DFN K=2 的 44.9 同时出现在旧 seed-0 与新 15-seed 均值里，若是巧合无妨，值得对一眼原始文件。

## 五、结论

这版修回的实质是好的：两个实验、四个诚实降级、一个干净的附录 D，审稿人六条 Major/Minor 里五条完整兑付。剩下的全部是收尾——三处过期数字、一条未做的 Minor、一封还没写的 response letter、一封需要换骨的 cover letter。半天内做完，这个修回包就是一个审稿人无法再要求更多的版本：他要的两个实验给了，结果不利的地方没有藏，连他自己数字里的一个小笔误（问题 3 的贡献归属）都替他答了。
