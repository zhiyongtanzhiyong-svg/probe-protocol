# Paper2: CLIP vs ResNet — 实验规程

> 本文件规定本仓库的实验作业规程：任何改动都要按此执行。
> 与 TODO.md 并存——TODO.md 记录「做什么」，本文件规定「怎么做才不浪费算力与时间」。

## 规程 0（最高优先级）：实验代码先走读审核，再执行

**规则：任何会消耗 GPU 或运行 > 几分钟的实验代码（训练 / 评估 / 特征提取 / 数据重建），
写完必须先做逐行走读审核（line-by-line walkthrough），审核通过后才能执行。禁止「写完即跑、跑了再修」。**

背景教训（2026-09-27，`scripts/coop_official.py` 连续 3 个 bug 都靠「跑崩了才发现」，
每次崩溃都浪费数小时 GPU + 数小时重跑）：
- `views[sup]`：`sup` 是**图像**索引，却索引到了 **view 轴**（size 200）→ `IndexError`（index 211 out of bounds）。
- 验证用 `views[:, val]`：3-D `(200, |val|, 768)` 触发 batched matmul → 维度错 + K=2 崩溃 + **~200× 变慢**。
- 正确做法：训练用增广视图 `views[:, sup]`，**验证用干净 eval-transform 特征 `X[val]`**（官方 CoOp 即如此）。

### 走读审核 checklist（执行前逐条确认）
1. **张量维度 / 索引轴**：每个 `arr[idx]` / `tensor[...]`，确认 `idx` 到底索引哪一维、该维的 size 是多少；
   3-D 张量做 `@` 会 batched matmul，先想清 shape。
2. **缓存顺序一致性**：dataset 行序 == 特征缓存行序（`y = y_full[:len(y)]`、label 与 feats 对齐）。
3. **类型 / 精度 / 设备**：fp16 vs fp32、`.float()` 时机、`device`、`torch.tensor(np...)` 是否产生意外 dtype。
4. **边界 / 空值**：`None`、`null`（如 `healthy` 的 `kg_entity: null`）、空 batch、DataLoader collate。
5. **协议忠实性**：是否匹配目标配方（官方 CoOp：N_CTX、epochs、seeds、views、**验证用固定 transform**）；
   训练/验证/测试用的 transform 是否各就其位。
6. **结果落地**：输出 JSON 路径存在、关键断言（class 数、样本数）能拦住脏数据。

### 执行前 gate（审核通过后）
1. **小样本 smoke test**：先用 1 个 seed / K=1 / 1 个 checkpoint 跑通，确认「不崩 + 数值合理」，再放开全量。
2. **长任务（>30 min GPU）**：必须 `setsid` 脱离会话启动（本环境 bash 工具超时即杀后台进程；
   `nohup`/`disown` 不可靠，`setsid` 已验证可存活），并轮询日志 + `nvidia-smi` 监控。
3. **先缓存、后训练**：把贵的中间产物（特征、增广视图）先落盘缓存，崩溃重跑时能复用，避免重复烧算力。

### 反模式（禁止）
- ❌ 写完直接全量跑，崩了再回头改（浪费算力）。
- ❌ 用 `pkill -f "<pattern>"` 时，pattern 恰好出现在**当前命令自身**命令行里 → 杀掉自己 shell。
- ❌ 假设某个 bug 修完就好，不做 smoke test 直接放全量。

## 规程 1：报告的数字必须可由 `data/*.json` 复算

`data/*.json` 是唯一事实来源。论文与 README 中的每个数字都必须能由这些文件推出，
且 `scripts/check_tex_vs_data.py` 必须在每次提交前通过。

**同一比较只能有一个口径。** 历史教训：全文曾把 15 种子均值（87.7%）与单一种子
（seed 0，87.0%）混用于同一句话，使「0.7 pp」看起来像模型差异，而真实均值差是 1.5 pp。
提交前须确认：报告的差值、置信区间、显著性检验三者来自同一口径。

**两类数字不可混用：**
- *同产物口径*：同一次运行的探针与其对照方法（如 LP++ 增益 = LP++ 减同产物探针）。
- *跨产物口径*：不同特征缓存或不同抽样下的数字。必须显式标注产物来源。

## 规程 2：改动前先备份

本仓库无远端。重写历史、批量删除、迁移目录等不可逆操作前，先
`git bundle create /home/lizhuo/Backup/<name>_$(date +%Y%m%d_%H%M).bundle --all`。
