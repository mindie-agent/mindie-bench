# mindie-bench 0.4.3：行为级任务判分器、策略选择实测证据记账与维护补丁发布流水线

[English](https://github.com/mindie-agent/mindie-bench/blob/v0.4.3/docs/releases/mindie-bench-0.4.3.en.md)

`mindie-bench 0.4.3` 将六个回归评测任务的 Verifier 升级为行为级执行校验，消除了策略选择评测中合成续跑结果的默认回退并保留超时/异常单 cell 的已测成本，同时新增了面向 `0.1.x` 维护补丁的 `release-kind=patch` 产品发布校验路径与组件仓内社区 Release Notes 发布支持。

---

## 一、核心修复与工程改进

### 1. 六个任务 Verifier 升级为真实行为执行校验（PR `#25`）

- 针对此前可被无功能或常量修改通过的六个评测任务判分器：
  - `dspark-mrv2-drafter-crash-zh-vague`
  - `k3-idle-dp-gdn-state-zh-vague`
  - `l3-dspark-mrv2-drafter-crash-zh-vague`
  - `mtp-draft-slot-mapping-capacity-zh-vague`
  - `sfa-mtp-seq-lens-eval-zh-vague`
  - `sfa-mtp-seq-lens-seed-zh-vague`
- 升级后的 Verifier 直接加载并执行被测工作区源码的真实行为逻辑，同时验证：
  - 已知缺陷基线必败（`before_reward=0`）；
  - 金标修复补丁必过（`after_reward=1`）；
  - 判分过程零大模型调用（`model_calls=0`）；
  - 注入无功能或常量返回值的负对照必败。

### 2. 策略选择续跑隔离与部分完成证据保留（`strategy_selection.py`、`knowledge_tiered_support.py`、`comparator.py`、`metrics.py`，PR `#24` / `#26`）

- **移除默认合成续跑回退（PR `#24`）**：默认策略选择器不再读取私有评判答案来合成续跑结果与伪造成本；当未提供真实续跑执行器时，策略选择与任务续跑严格分离（标记为 `not_executed` 或 `continuation_unverified`，未实测的续跑耗时与设备成本保持 `None`）。
- **实测调用记账与超时/异常部分证据保留（PR `#26`）**：
  - 确定性策略选择如实记为 `0` 次模型调用，并将预留预算与实际观测到的模型调用分别记账；
  - 当单 cell 续跑发生超时（`unknown`）或基础设施异常（`infra`）时，完整保留已完成的选择器判定结果、已实际消耗的调用次数以及实测超时耗时。

### 3. 维护补丁发布流水线 `release-kind=patch` 与仓内 Release Notes 接入（`.github/workflows/publish-product.yml`、`collect_product_benchmarks.py`、`publish-release.yml`，PR `#27`）

- **新增 `release-kind=patch` 维护补丁发布路径**：
  - `.github/workflows/publish-product.yml` 与 `.github/scripts/collect_product_benchmarks.py` 新增显式的 `release-kind=patch` 模式，仅允许真正的维护补丁版本（`PATCH > 0` 且禁止传入历史 `value-suite-path`）使用；特性里程碑版本（`0.Y.0`）仍强制执行完整的 `milestone` 产品价值验收门禁。
  - 在 `release-kind=patch` 下，流水线绝不把历史 `0.1.0` 的 109 轮价值报告重标为 `0.1.1` 新实验，但继续强制校验全部组件的正式 GitHub Release、不可变 Tag SHA、源码包版本、各仓 `mindie-release-benchmark/2` 报告与 SHA256 校验和，并强制核对 `channels/0.1.json` 与 `benchmark-manifest.json` 的组件集合、版本号与提交 SHA 完全一致（随发布资产输出 `release-channel.json` 并在 `product-release-provenance.json` 中记录 `channel_sha256`）。
- **组件 Release 直接采用仓内审校的社区版本说明**：
  - `.github/workflows/publish-release.yml` 新增 `release-notes-path` 输入，支持各组件正式 GitHub Release 直接采用仓内审校定稿的社区版本说明（同时打包为 `release-notes.md` 并写入 `SHA256SUMS`），不再默认以原始基准表格或自动生成的提交列表充当 Release Notes。

---

## 二、验证事实与历史数据边界

- **Verifier Soundness 与负对照验证**：六个修复后的任务 Verifier 全部通过 `before_reward=0, after_reward=1, model_calls=0` 校验及无功能负对照测试（**`7 passed`**）。
- **发布流水线与收集器验证**：`mindie-bench #27`（合并提交 `484b80621aaf9293c559786d0ef2174c79a0dbb4`）已通过跨平台 CI 并合入主干；本地真实执行 Git Tag 前置检查与产物收集器的回归测试 **`30 passed`（`18.12s`）**。
- **历史 `0.1.0` 数据边界说明**：历史 `0.1.0` 的 109 个 Trial 尚未在修复后的行为级 Verifier 下重跑或重判。因此，修复 Verifier 本身不代表历史 Trial 自动生效；在完成重判前，`0.1.0` 的历史量化收益结论继续暂停作为当前产品价值验收依据。
