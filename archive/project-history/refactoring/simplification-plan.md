# WWW2027 项目适度简化方案

状态：用户已批准实施；完成情况见 [验收报告](../../../docs/simplification.md)。原方案如下。适用于当前单仓库研究原型，不预设用户已选择 Research Track 或 Demo Track。

## 1. 建议

保留已验证的复现基础，减少日常操作和历史维护负担。目标是将 13 个可见一级目录收敛为 9 个；不推倒现有 Python 包，不为了少几个文件把不同研究模块强行合并。

上一版方案对迁移过程和长期追踪投入较多，迁移审计也变成了日常检查的一部分。对当前投稿阶段，这部分可以退到档案中。此次应先确定一次稳定边界，再集中精力完成研究问题、基线比较、消融实验与演示。

[WWW2027 Research Track](https://www2027.thewebconf.org/research-track-papers/) 将技术贡献、执行质量和可复现性列为评估因素；[Demo Track](https://www2027.thewebconf.org/demos/) 要求已实现和测试的系统，并说明现场展示方式。两份指南没有规定仓库目录模板。此方案属于针对本项目的工程判断，不是会议强制要求。

## 2. 目标结构

```text
WeightLoss/
├── README.md                 # 安装、验证、复现、演示入口
├── pyproject.toml            # 可安装包与依赖组
├── Makefile                  # 常用命令
├── .env.example
├── src/weightloss/           # 保留现有核心模块及 CLI
├── apps/web/                 # 保留现有网页源码，避免无意义改名
├── configs/                  # 药物目录、默认流程与实验配置
├── data/
│   ├── raw/                  # 冻结采集快照
│   ├── external/             # 术语、有效嵌入、注释种子、说明书
│   ├── interim/              # 抽取结果与 checkpoint
│   ├── processed/            # 标准化数据
│   └── indexes/              # 可重建检索索引
├── results/<run_id>/
│   ├── manifest.json         # 本次实验的索引与来源记录
│   ├── metrics.json          # 评估指标（有实际评估时产生）
│   ├── figures/              # 论文图表（需要时创建）
│   └── demo/                 # 生成的网页与数据副本
├── tests/                    # 保留 unit/integration/frontend/fixtures
├── requirements/             # 两类已验证环境，保留平台限制说明
├── docs/                     # 当前研究范围、复现说明、参考文献
└── archive/                  # 历史实验、笔记本、旧结果、迁移材料
```

`.git/`、`.github/`、`.venv/`、`.venv-research/` 和 `.cache/` 是版本管理、CI、本地环境或缓存，不纳入可见源码职责目录计数。目录里的可选结果文件只在真实运行产生后创建。

## 3. 具体调整

| 当前内容 | 建议调整 | 保留条件 |
| --- | --- | --- |
| `experiments/legacy/` | `archive/experiments/` | 保留原脚本与来源说明；新实验配置统一进 configs |
| `notebooks/legacy/` | `archive/notebooks/` | 现有两本都是历史探索；未来确有活跃笔记再按需设置目录 |
| `results/legacy/` | `archive/results/` | 历史版本、报告与哈希成组保存 |
| `requirements/legacy/` | `archive/environments/` | 现用 requirements 只保留当前环境与说明 |
| `docs/refactoring/`、`docs/history/` | `archive/project-history/` 下各自子目录 | 保留日期、迁移映射与验证记录；更新所有相对链接 |
| `artifacts/embeddings/legacy/embedded_ae.csv` 及来源记录 | `data/external/terminology/` | 仍是标准化输入；按字节保留，不误删为缓存 |
| `artifacts/indexes/` | `data/indexes/<run_id>/` | 保留数据/模型指纹，生成的索引继续忽略 |
| `artifacts/web/`、`artifacts/web-runs/` | `results/<run_id>/demo/` | 仍由同一 processed 版本构建；源码与生成目录隔离 |
| 两个 `scripts/audit_*.py` | 作为一次性迁移工具归档 | 迁移时执行原审计；日常 CI 保留数据、功能与链接检查 |
| `src/weightloss/embeddings/embedding_pi_*.py` | 确认无维护消费者后归入 archive/experiments | 当前 CLI 无对应命令；不能连带移除基线所需的术语嵌入功能 |
| 根 `TODO.md` | 可并入 `docs/roadmap.md` | 保留未决状态，不将方案变为既定研究决策 |

删除空的 experiments、notebooks、artifacts、scripts 一级目录。保留 src、apps、configs、data、results、tests、docs、requirements、archive 九个目录。

## 4. 减少操作负担

### 日常入口

优先向使用者展示四个命令：

- `make install`：安装轻量可运行环境。
- `make test`：必要的数据校验与自动化测试。
- `make reproduce`：新增聚合入口，从固定小样例跑到可检查的结果。
- `make demo`：新增聚合入口，构建并启动现有演示。

实施时新增这两个别名。已有 CLI 的 collect、extract、standardize、build-index、import-graph 等底层命令保留，放到高级操作说明。聚合命令仍调用同一套函数，不增加第二套脚本。

### 环境

将用户可见的依赖选择收敛为 demo / research / dev 三组。保留轻量环境和研究环境的两份锁；不为了少一个目录把 Torch、FAISS、PDF 处理全部装进演示环境。无当前消费者的说明书实验依赖退出主研究锁，原环境留档。重新解析后必须重做安装与导入检查，不能把修改过的依赖表称作已验证。

### 实验记录

保留 run ID、冻结、失败恢复和并发写保护。在论文实验或会修改数据的运行中记录代码版本/源码摘要、输入哈希、配置、prompt/schema、模型与术语版本、参数、输出哈希和覆盖情况。

环境清单按 run/环境指纹保存一次；同一环境中的阶段记录引用它。每阶段保留输入输出、状态与失败记录。普通网页重新构建只保留当前 build manifest，无需反复生成完整科研操作档案；正式论文/演示快照仍引用完整研究 run。旧操作日志归档，不能覆盖真实历史。

## 5. 保留的研究质量保障

- 数据四层语义、唯一选定的 processed 版本及网页生成副本。
- 历史注释种子的运行依赖；pending/failed 与无副作用的区别。
- 索引数据/模型指纹、图导入保护、冻结基线。
- 当前 27 项回归、5 项模拟适配检查和 3,963 项前端断言覆盖。
- 配置、锁定环境、单一安装包与 CLI，避免工作目录依赖和导入副作用。
- TableRAG/GraphRAG 及当前标准化方法。本次整理不决定删掉某一种检索方法。

测试不为了目录数量而删除或重新实现。旧的“根目录必须恰好 13 个目录”检查应退出默认 CI；文件找不到、资源不一致和结果变化仍由功能检查发现。目录迁移本身保留一次明确验收。

## 6. 实施顺序与验收

1. **集中历史材料。** 按清单归档并校验字节；修复文档链接。先不动研究模块。
2. **合并生成产物边界。** 更新配置、new-run 默认路径、网页构建、测试读取与忽略规则。处理所有现存运行配置；冻结历史 manifest 保留原始内容，由迁移表解析路径。
3. **精简入口和日志。** 新增两个 Make 聚合入口，收敛依赖组及环境记录，归档一次性审计工具。
4. **统一文档与验证。** README 聚焦上手，docs 保留当前说明，archive 保留历史；验证真实目录与最终图一致。

验收要求：

- 可见一级职责目录为九个，无过渡兼容目录。
- 所有原始数据、选定抽取/标准化结果、有效嵌入和被移动历史文件字节不变。
- 单元、集成、前端和模拟适配检查通过；小样例复现结果保持一致。
- 新建 run、冻结、checkpoint 恢复、索引拒绝旧版本与网页 HTTP 资源加载仍成立。
- 全部文档链接及命令路径同步，不改变药物范围或研究方法，不重跑在线采集与模型。

原提案轮仅提交方案；后续用户已批准完整实施。正式论文结果和拟展示的后端能力仍需各自的实验或服务验证，目录简化不会替代它们。
