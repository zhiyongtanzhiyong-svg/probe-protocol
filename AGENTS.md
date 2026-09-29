# Paper2: CLIP vs ResNet — 项目纪律 (Agent Discipline)

> 本文件是本仓库所有 agent 会话必须遵守的纪律。与 TODO.md 并存：TODO.md 记录「做什么」，
> 本文件规定「怎么做事才不浪费算力与时间」。

## 纪律 0（最高优先级）：实验代码先走读审核，再执行

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
