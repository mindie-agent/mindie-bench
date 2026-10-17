# 原始方案差距分析与后续实现

本文保留 G/H/I 阶段的差距诊断。其后已落实的改动见
[实际验收](acceptance.md)：知识私域与真实 MCP、首轮工具就绪、shell 源码访问、
并发区间计时、首末编辑和试验分段、可执行 L2 挖掘、真实报错回放、
有版本反例的 stale 题，以及真实 DP2/HCCL 状态题。
本页后文的“现有”“本次”均指原分析阶段，不能用来判定后续实现状态。
各项收益以新实验的独立判分和实际成本为准。

截至 2026-10-17，系统已具备原生模型、真实硬件、外部判分和经验交付能力，
但现有实验不能回答原方案提出的全部“能多解决多少问题、减少多少总成本”问题。
主要障碍是题目与协议覆盖、对照设计及观测语义，不能靠增加相同题目的重复次数解决。
本文没有修改旧回执、补造缺失字段、重试 unknown，也没有把新的覆盖检查当作硬件验收。

审计依据是原始 `mindie_agent_benchmark_strategy.md`，SHA-256
`92a5a9d19155f63b1081e81b0b57ca94114d6639e0ae8b7061a064225f02b6e3`。
原文第 2、3、10/M1、11.5 节分别定义业务收益、分层评测及首期选题。
现有实现和实际结果见 [design](design.md)、[acceptance](acceptance.md)、
[L1/L2 证据规则](../tasks/L1-L2-acceptance.md) 与 [Ascend 任务入口](../tasks/ascend/README.md)。
原方案引用的外部研究及其中的预期提升数字不是本系统的观测结果，本文不重新认证这些引用。

## 为什么主要收益指标暂时回答不了

1. **题目存在完成率天花板。** 当前六个真实触发契约来自 RMSNorm、partial RoPE、
   reduce-sampler 三个独立上游缺陷族，均为 easy、`near_exact`、单卡函数级任务。
   G 和 H 的两种硬件各自 G0 eval 均为 9/9 通过；在这些有效样本上，完成率正向提升没有空间。
   仍可描述耗时、成本或负迁移，但不能从“同样全过”推导知识提高了解题能力。
   同一缺陷的 seed/held-out 参数变体、A2/A3 副本和重复运行是更多观测，不是更多独立缺陷。
2. **实验未覆盖产品的全部合法激活协议。** 产品原则 7 明确允许用户通过
   `/mindie-agent <问题>` 按需唤醒领域知识；远端执行默认就绪，已授权的 Stop 沉淀默认运行。
   自然任务里提供工具但没有显式入口，与显式产品路径是不同协议。
   自然前台 MCP query/explain 为零，只能说明所测协议没有这些事件，
   不能自动判为产品故障，也不能证明显式路径无效。
   单独一次显式 canary 成功则只证明这条有限路径可用。
3. **G0–G3 不是知识归因实验的四种条件。** G1 同时改变远端工具入口，G2 增加知识与 Skill，
   G3 还增加 seed/capture 和时间上的 corpus 更新。它们不能直接充当
   Baseline / Static / Dynamic / Oracle 的四路知识消融。模型、题目相同仍不足以隔离这些因素。
4. **已有字段与想回答的问题并不等价。** 命令正则命中是一次测试命令尝试，
   不证明设备开始执行、执行完成或发生了多少 kernel；目标文件的 Read/Edit 是源码访问，
   不证明已提出正确根因。工具耗时、缺失 usage 或父进程退出，也不能自动补出业务耗时与成本。
5. **对照与样本覆盖不完整。** 固定组别设备池没有随机化或交叉平衡；
   unknown、infra 和被阻断的 held-out 必须保留。只挑 mutually valid 的成功子集，
   可以形成有限描述，但会改变样本选择，不能据此声称完整矩阵或一般因果收益。

前台阅读也不只发生在 MCP：成功原生 `Read`/`cat` 返回正文，并与自身 seed
新包的精确 revision/block、实际返回 bytes、文件及 body SHA 对上，可以证明正文交付和读取。
单有路径、grep 命中或配置引用不够。独立 consumer 读回不是前台使用。
原生直接读没有排名列表，不进入 MCP Recall/MRR 分母；任何一种读取本身都不证明语义适用或收益。

## 原始目标到当前缺口

| 原方案目标 | 已实现或已有的有限证据 | 仍缺什么；下一步的可验收产物 |
| --- | --- | --- |
| L0 组件微基准 | 确定性测试、真实原生/远端生命周期 canary、工具契约检查 | 用独立 operation ID 测冷/热启动、query/explain、Stop 前台与取消延迟，保留原始时间、版本、样本量和分布。测试通过数不是延迟基准。 |
| L1 路由泛化、日常零误扰 | 修正 fixture 后的四个实际阴性样本，以及一次单独显式 warm canary | 冻结正样本、相近负样本、无关请求和歧义标签；分别定义 Skill、knowledge、remote、Stop 事件。原计划百例级覆盖不能由 N=4 的零事件替代。 |
| 检索质量 | query/explain 与独立来源的 slice scorer；精确新包交付可核对 | 采集真实查询的有序结果、完整 block 和独立 relevance judgement；相关集合不全就不报一般 Recall。零自然查询时排名指标未测，不是 0 分。 |
| 四级知识复用 | schema 支持多个梯度，现有三族任务全部 `near_exact` | 新增独立源证据的 analogical/compositional/stale 条件；no-hit 单列。每种条件固定实际 corpus、gold 适用性及 stale 当前反例。 |
| 原首期 12 题、五业务域 | 六个触发契约、三个缺陷族，函数级真实 NPU source-fail/gold-pass | 不是原定 12 题：还缺图编译、真实多卡/HCCL、全模型精度、Profiling/吞吐和跨文件特性迁移等覆盖，尤其原计划 3 个多卡任务。不得用 TP=1 identity 代替 collective。 |
| L2 低成本工程孪生 | 两族四个分别证明的真实 CPU twin；一次 G0 原生 CPU smoke | 没有完整 CPU G0–G3 对照及更广独立历史任务集。CPU tensor 正确不证明 NPU、AscendC、HCCL、模型服务；重放反馈也不代替真实 source 执行。 |
| 完成率净增、重复可靠性 | 外部 reward、四态终态、每任务重复及 matched-pair 比较 | 先选择非天花板且业务重要的独立任务，再估计样本量。`(c/n)^k` 是插件估计，不是已观测连续成功或未来保证。 |
| 端到端总成本 | 前台/小模型 usage、交付时间和 G3 seed-inclusive 成本字段 | 必需 seed 成本不得丢；unknown 或 usage 缺失不等于零。USD 价格未配置仍是 null。实际对照需设备平衡及完整计划账本。 |
| 首次根因判定时间 | typed source access 的 localization proxy，缺观测时有覆盖保护 | 增加实际诊断文本、时间戳、根因 gold 和独立判定。打开目标文件不能代替语义根因判定。G0 shell 访问不可观测不等于没有定位。 |
| NPU 试错减少 | 命令尝试计数、实际外部 verifier case、设备前后状态 | 建 attempt ID→准入→设备进程开始→完成→退出/错误类别链，区分工程复现、外部 verifier 与 kernel。先不把正则计数更名为真实硬件次数。 |
| 知识的路由/内容归因 | 自然未查询与单独显式 canary 可分别观察 | 对同一适用任务冻结 Baseline、Static、Dynamic-natural、Dynamic-explicit、Oracle。显式是合法产品路径；自然路径作额外假设，不能强迫每题检索。 |
| 系统性实机验收 | H：90 planned、88 actual、2 blocked；I：独立四格原生 RoPE canary 和两硬件各两格零模型 RPC 边界检查 | H 不全过；I 没有 G0/G1/G2 对照、仅一族一重复，不能替换 H、完成首期 12 题或证明提效。旧 node 153 unknown 不因 node 156 成功而闭合。 |

原文第 2.2 节的梯度 3 是 **compositional**，第 11.5 节却把梯度 3 用作
**no-hit**。实现已将这两种问题分开；下一轮计划必须显式列出，而非默选一种声称全覆盖。
第 11.5 节的数量是 3 个 near-exact、5 个类比迁移、2 个 no-hit、2 个 stale，
共 **12 个任务**、9 单卡与 3 多卡；原方案没有要求“12 个独立缺陷族”。

## 可执行覆盖契约

纯函数 [coverage.py](../src/mindie_bench/evaluation/coverage.py) 接收：

```python
assess_coverage(
    tasks,  # Sequence[TaskSpec]，调用者负责实际加载/校验
    groups=["G0", "G1", "G2", "G3"],
    repetitions=3,
    targets=default_targets(),  # 可为明示的较窄问题定制 MetricTarget
    evidence={"task_completion": MetricEvidence(...)},
).to_dict()
```

它不读文件、不运行任务、不联网、不写回回执，也不推断缺失结果。
显式计划的 expected/actual/blocked 账本由独立 accounting 负责，不能用
`任务数 × 组数 × 重复数` 猜出实际矩阵或补造未启动 eval。
`execution_readiness="not_checked"` 明确表示本函数不核验解释器、凭据、image、模型或设备是否可运行。
没有 evidence 的调用是缺口预检，不是一次已完成的评测。

每项目标返回状态、具体原因、样本单位、observed/valid/unknown 数量、
声明及实际观测的任务/上游 fix/domain/gradient 覆盖、缺失字段样本数与缺失协议：

| 状态 | 精确含义 |
| --- | --- |
| `ready` | 明示目标的覆盖先决条件已满足；可能测得零收益或负收益，不代表显著性、硬件全验收或正向宣传结论。 |
| `not_measured` | 无有效实测样本，或必要语义字段/协议未实测完整。例如未做 Oracle、没有根因 gold、缺失成本。 |
| `insufficient_sample` | 有相关观测，但题数、独立性、领域、梯度、实际每任务重复不足，存在 invalid/unknown 覆盖缺口，或完成率已在天花板。 |
| `not_applicable` | 所选实验不适用这个问题，例如 CPU-only 任务的 NPU 指标，或只选 G3 却要求 G0/G2 对照。不能当作通过。 |

`MetricEvidence` 中 `field_samples` 表示该前提已**实际核实**的有效样本数，
不是字段存在次数。例如 `source_identity` 必须是实际 bytes/hash 核对通过，
`matched_controls` 必须有真实对照记录，`randomized_or_counterbalanced_devices`
必须有实际执行的分配记录。一个 report 写“已修”不能获得这些计数。
缺失或失败不能填满计数；未知新字段也不使旧回执变成已观测。
`observed_task_ids` 只列有效样本对应任务，family 以源仓库+fix commit 去重，
不以硬件、cluster 标签、重复次数或目录数造独立性。
该去重只是保守、可审计的源证据键，不自动认证不同 commit 是独立语义缺陷；
复杂跨提交缺陷仍需人工题源审查。

`task_repetitions` 记录**实际每任务**的 repetition 编号；计划为 3 不能代替已观测 3。
`protocol_samples` 分别记录真实运行过的自然、显式、Oracle 等协议。
数据归一化者必须按 sample unit 去重并绑定私有源证据；本模块不把可伪造计数当作来源认证。
对于 completion lift，baseline 的成功数与分母必须来自同一 matched 有效样本，
全成功时不能辨认正向完成率收益；这不阻止另报描述性完成率或衡量成本。

默认目标覆盖十四个问题。L0 30 次 operation 是明示、可调整的最低采集策略，
不是原方案的既有数据或统计功效结论；原方案的 12 题/五域、百例路由和三重复
也是覆盖目标，不能代替功效分析。实际因果结论还需明确效应量、置信区间、
任务/缺陷族相关性与多重比较；不能把多个 query 或同族参数变体当独立解题样本。

## 改进顺序与停止条件

先用零模型审计保留所有旧 raw/hash 与四态终态，输出真实计划账本和上述 coverage；
补齐可从已保存实际结果可靠提取的 usage、源身份和工具业务结果，无法补取的标 unknown/null。
这能改善解释能力，不会修复原来未验收的源码/硬件或消除外部 unknown。

随后为新的独立实验冻结激活协议、知识条件、模型与预算，先 source-fail/gold-pass，
再以有限试运行确认难度和业务相关性；若基线全过，就调整选题或把问题改为成本，
不能为了制造收益强制查询、隐藏基线成功或扩大重试次数。正式实验前再确定
随机/平衡设备安排、足够独立任务量与对照；保留 G1/G2 隔离知识效果、G0/G1 隔离工具效果，
G3 使用自身 seed→capture→held-out 的谱系并计入全部必需 seed 成本。

最后才在覆盖就绪且有明确预算的协议上收集新增实测；任何 unknown 保持原样，
仅做只读对账，不自动重放。公开结果只引用实际验收范围；
本地 Git 与独立 consumer 交付不冒充公开 PR/merge/Release/HF 发布。

## 本次实现的九原则审查

本次新增范围是一个只读覆盖函数、其测试与本差距说明，不改变模型/runner/工具路由。
依据原则修订 `240248b18996904490cca1e1ecceb8d9ecd83e03`：

1. 总成本：运行前暴露无法回答的问题，减少在天花板或缺对照实验上重复花费。
2. 封闭边界：只核配置与已提交的观测先决条件，不替 Agent 诊断开放业务问题。
3. 认知负担：复用 TaskSpec，提供默认目标和 JSON 输出；不新增产品入口或用户仪式。
4. 单一职责：coverage 不维护矩阵账本、不调度、不修改 scorer 或回执。
5. Skill 地图：文档只说明证据需求，不强制 Agent 按固定路径读知识。
6. 历史参考性：读取或命中不能证明适用，stale/no-hit 条件独立缺口可见。
7. 按需介入：显式知识入口合法；自然未触发不自动判故障；不补发模型任务。
8. 成熟机制：使用现有 TaskSpec 与标准库数据结构，复用既有真实证明和原生证据。
9. 错误可见：非法计数/未知任务明确抛错；缺字段不置零通过；unknown 不重试或改写。

相关自动验证见 [test_evaluation_coverage.py](../tests/test_evaluation_coverage.py)；
这些测试验证覆盖判断本身，不构成新的业务任务、模型或硬件验收。
