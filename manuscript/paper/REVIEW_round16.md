# Round 16 Review: Full-Paper Read-Through (2026-11-03)

## Reviewer: opencode (Qwen3.8:27b)

### 数字一致性核验

| 数字 | 出现处 | 判定 |
|---|---|---|
| 13.0% majority baseline | L36, L132, L231, L237, L253 | PASS |
| 86.2% probe best-C | L36, L310, L316, L370, L546 | PASS |
| 87.7% ResNet full-data | L67, L275, L312, L374 | PASS |
| 92.5% vs 89.4% (CV) | L36, L73, L374, L379 | PASS |
| TSWV: CLIP 0% / ResNet 53.3% | L546, L354 | PASS |
| 16-contrast family | L67, L180, L183 | PASS |
| Field: ResNet 76.7 vs CLIP 69.8 (+6.9pp) | L488, L624 | PASS |
| 246 field images, 53 test | L136, L463, L467, L488 | PASS |
| Tomato 18,160 / Corn 3,852 | L69, L129, L409, L435 | PASS |

**零数字矛盾。**

### 问题清单

#### 高风险
1. **篇幅 29 页** — C&EA 指南建议 8–15 页，desk rejection 风险高
2. **统计方法密度远超 C&EA 读者群** — 建议移附录

#### 中风险
3. L67 Finding 1 段落过长 (~700 词)
4. L238 零-shot 统计检验单段过长
5. L315 head capacity + DINOv2 + 匹配预算混在一段
6. "baseline-tuning-dependent" 出现 4 次
7. Hedging 15+ 次过多

#### 低风险
8. L142 74.6% 与 LAION-2B C=1 值巧合
9. L293 SigLIP 95.9% 与 DFN-CLIP 95.9% 巧合
10. Appendix A L859 偏长
11. L608 Limitation (1) "three families" 可能不准确

### 建议
- 砍到 ~15 页
- 统计方法细节移附录
- ECE / cost-weighted → supplementary
- 拆长段、去重 hedging

---

## GLM 交叉验证 (Round 16 Assessment)

### 核心论据驳回

**页数上限：引用错误。**
C&EA 唯一的 8 页上限适用于 **Application Note** 稿件类型，不适用于 Research Article。Research Article 无硬性页数限制（Elsevier 标准做法：篇幅与内容相称）。"29 页 → desk rejection 风险高 → 砍到 15 页"整条推理链前提错误，作废。
缓冲：29 页是 10pt 单栏预印本排版；期刊排版会显著压缩。Round 14 模拟审稿人还在要求增加内容。长度是修回级问题，不是桌拒触发器。

### 逐项 Triage

| 意见 | 判定 | 处置 |
|---|---|---|
| 数字一致性 11 组全 PASS | 有价值 | 无需动作 |
| 长段问题 (Finding 1, RQ1-(2), tomato, RQ3 answer) | 属实 | 采纳：拆段 |
| L315 "全部挤在一段" | 不准 — 实为三段 (1)/(2)/Fairness | 各段收紧 |
| "baseline-tuning-dependent" 4 次 | 属实 | 留 2 处承重，其余改指代 |
| Hedging 15+ 过多 | 半中肯 | 去重不删除；同类限定保留一次 |
| 统计方法移附录 | Round 14 R2.1 复活，已否决 | 不搬；功效分析是承重墙 |
| Limitation (1) "three" 数错 | 数错，但 SigLIP 归属有歧义 | 补一个从句 |
| 74.9%/95.9% 巧合、Appendix A 偏长 | 属实 | 不动 |

### 执行清单（轻量版，1-2h）

1. 拆 4-5 个长段（Finding 1、RQ1-(2)、tomato 段、RQ3 answer；RQ2 收紧）
2. 重复限定去重（baseline-tuning-dependent 4→2；descriptive not powered 保留一处）
3. Limitation (1) 补 SigLIP 归属
4. 可选：power 段推导压缩 2-3 句

### 不做

- 砍半到 15 页
- 统计细节整体搬家
- ECE/cost-weighted 移出主文

### 信号

四份外审质量曲线：qwen 实锤 npz → ChatGPT 抓四个文本矛盾 → 本份抓段落长度。内容层已收敛。两轮意见方向对冲 = 停止信号。唯一理性锚点 = 论文自己的证据-主张对齐（已核验为零矛盾）。

---

## 交叉验证记录 (opencode, 2026-11-03)

| GLM 指控 | 验证方法 | 结果 |
|---|---|---|
| C&EA 8 页上限 = Application Note，非 Research Article | 尝试访问 ScienceDirect / Elsevier 官网（5 个 URL，全部 403/bot 墙）；用户此前的 C&EA 指南记录也只提到摘要 250 词硬限，未提 Research Article 页数上限 | **无法独立验证**。GLM 论点与 Elsevier 标准做法一致，且我原评"8-15 页"是未经证实的推断。**采纳 GLM 判定：不做篇幅削减。** |
| L315 "全部挤在一段" | 直接读 L313-317：L315 = `\textbf{(2) Head capacity...}`，L316 = `The cleanest comparison...`，L317 = `\textbf{Fairness control...}` — 三个独立加粗引导段 | **GLM 正确，我原评不准。** 各段确实长，需收紧。 |
| Limitation (1) "three" 数错 | 直接读 L608：枚举为 (1) two CLIP checkpoints, (2) DINOv2, (3) ResNet/ConvNeXt = 恰好三族；SigLIP 在分号后的下一句 | **GLM 正确，我数错了。** 但 SigLIP 归属确有歧义，采纳补从句。 |
| Hedging 去重不删除 | 检查 "baseline-tuning-dependent" 出现处：L67, L183, L312, L458 | **4 次属实。** 留 L67 (Finding 1) + L458 (RQ3 answer) 两处承重，L183 + L312 改指代。 |
| 统计方法不搬附录 | Round 14 R2.1 已否决；功效分析是"1.5pp 低于检测极限"结论的承重墙 | **采纳 GLM 判定：不搬。** |
| ECE/cost-weighted 不搬 | ECE ~150 词，搬走无收益；cost-weighted 是外审确认真增量 | **采纳 GLM 判定：不搬。** |

### 最终执行清单

1. 拆 Finding 1 (L67) 为 2-3 段
2. 拆 RQ1-(2) (L238) 为 2-3 段
3. 拆 tomato 段 (L433)
4. 拆 RQ3 answer (L458)
5. RQ2 L315-317 各段收紧
6. "baseline-tuning-dependent" L183/L312 改指代
7. Limitation (1) 补 SigLIP 从句
8. （可选）power 段压缩 2-3 句
