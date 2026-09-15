# 项目架构评估与重构计划

日期：2026-09-14。状态：用户已批准并已实施。以下保留批准时的方案；实际完成情况与验证边界见 [实施报告](refactoring/implementation.md)。

## 1. 结论与判断依据

当前是有文档、数据保护和测试的研究原型，但还没有形成完整的可复现研究软件结构。适合继续探索；若作为长期协作项目或论文复现仓库，需要补齐环境、数据版本、实验记录和执行入口。

“学术界标准架构”不是唯一固定模板，`code_*` 命名本身也不是错误。重点是能否明确回答：哪个输入、哪个版本的代码和模型、使用什么参数、产生了哪个结果，以及别人如何重新运行。

参考 [Wilson et al., 2017, Good enough practices in scientific computing](https://doi.org/10.1371/journal.pcbi.1005510) 关于项目组织、版本控制和协作的建议，以及 [Cookiecutter Data Science 官方实践](https://cookiecutter-data-science.drivendata.org/opinions/) 对不可变原始数据、处理阶段、环境和实验记录的建议。以下目录是针对本项目的设计，不是这两份资料要求的强制规范。

## 2. 当前结构的优点与缺口

| 维度 | 本项目证据 | 评估与动作 |
| --- | --- | --- |
| 功能分工 | `code_scraping/`、`code_extraction/`、`code_standardization/` 等 | 已有合理分工，保留边界并收进统一包 |
| 文档与历史 | README、TODO、`docs/history/`，区分当前范围与未决提案 | 较好；继续保留历史语境 |
| 数据一致性 | manifest、校验和、索引指纹、空图导入保护 | 较好；作为重构回归条件 |
| Python 包与路径 | `drug_catalog.py:4` 以自身位置定义 ROOT；多处修改 `sys.path`；TableRAG 使用工作目录相对路径 | 安装、移动及跨目录运行脆弱；集中配置路径并增加统一命令入口 |
| 导入行为 | `schema_extraction.py:18` 导入时读 CSV；`standardization.py:21` 起读 CSV 并初始化模型 | 导入与执行耦合；改成显式加载函数，核心模块导入不应读取研究数据或启动模型 |
| 数据生命周期 | `data_standardized/` 混放当前结果、top10、UMLS v2–v6、比较表和报告 | 当前、实验、报告、缓存边界不明确 |
| 历史数据依赖 | `refresh_data.py:77` 使用 `data_backup/pre_2026_refresh/` | 该部分是注释复用输入，不是可以随意迁走的闲置备份 |
| 网站数据 | `publish_website()` 复制到两处，测试检查一致性 | 已受控制的发布副本；应成为明确的构建产物而非多个权威数据入口 |
| 环境 | 多份 requirements；网站环境混合大量研究依赖，部分没有固定版本；README 说明未验证跨平台 | 建立核心依赖、可选功能组及验证过的锁定环境 |
| 实验追踪 | 有版本结果、模型名称和历史报告，未见统一 run manifest | 缺少贯通数据、代码、参数、模型与输出的记录 |
| 测试与自动化 | 2 个测试文件；没有已跟踪的 CI 工作流或统一流程文件 | 有基础回归，但尚不能证明完整模型与数据库链路可复现 |
| 论文交付 | 未见统一结果索引、CITATION.cff 或软件 LICENSE | 准备公开复现时补齐；作者、软件许可及数据许可须分别确认 |

网站的 `code_website/chatbot.py` 实际是到 `code_chatbot.chatbot` 的兼容导入，不应误判为第二份独立 chatbot 实现。

## 3. 建议目标结构

原则：维持单仓库，用一个可安装的研究包组织可复用代码；界面、数据、实验配置和结果分开。只创建实际需要的子目录。

```text
WeightLoss/
├── README.md
├── TODO.md
├── pyproject.toml             # 包定义、核心依赖与可选功能组
├── requirements/             # 经验证的锁定环境；记录 Python/平台
├── Makefile                  # 本地校验、数据处理、构建的快捷入口
├── .env.example
├── configs/
│   ├── drugs.json
│   └── pipeline.json         # 明确选择输入快照、注释种子及输出位置
├── src/weightloss/
│   ├── __init__.py
│   ├── cli.py
│   ├── settings.py           # 显式配置数据根目录，避免依赖 cwd
│   ├── catalog.py
│   ├── provenance.py
│   ├── ingestion/            # 当前 WebMD；Reddit 实现后再加入
│   ├── extraction/           # 可复用逻辑、schema 与版本化 prompt
│   ├── standardization/      # 当前基线与可复用标准化逻辑
│   ├── embeddings/
│   ├── retrieval/            # 保留现有 TableRAG / GraphRAG
│   ├── pipeline/             # refresh、validate、构建数据副本
│   └── evaluation/           # 可复用评估逻辑
├── apps/web/                 # Flask、模板和前端 JS
├── notebooks/                # 探索笔记；历史笔记标注状态
├── experiments/              # 实验配置、README、运行方式；不混入结果
├── data/
│   ├── README.md             # 字典、来源、版本与获取方式
│   ├── raw/webmd/<snapshot>/ # 不可变的采集快照及 collection manifest
│   ├── external/             # 术语表、说明书、annotation_seeds
│   ├── interim/<run_id>/     # 抽取结果与可恢复的中间状态
│   └── processed/<run_id>/   # 标准化结果与 dataset manifest
├── artifacts/
│   ├── embeddings/           # 带输入及模型指纹的派生产物
│   ├── indexes/              # 带数据和模型指纹的检索索引
│   └── web/                  # 构建出的网页及相对路径数据资源
├── results/<run_id>/         # manifest、指标、表格与图；发表结果有索引
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── frontend/
│   └── fixtures/             # 小型、可离线运行的合成样例
├── scripts/                  # 迁移字节与目录结构审计
├── docs/                     # 保留现有文档与 history
│   └── references/           # 文献 PDF 与来源索引
└── archive/                  # 只存已确认不被当前流程依赖的历史材料
```

`artifacts/` 默认存再生产物，但不能仅因“理论上可重建”就丢弃无法重新获取的旧产物。`results/` 里的小型运行元数据与正式结果应可追踪，不能沿用对 `outputs/` 的全量忽略方式而丢失论文依据。`CITATION.cff`、LICENSE、论文目录在信息齐备或确有稿件时加入，不建立空壳。

### 数据流与权威来源

```text
冻结的采集快照 + 注释种子 + 配置
                    ↓
        抽取中间结果 → 标准化结果
                          ├→ 评估 → results/<run_id>
                          ├→ FAISS / 图导入
                          └→ 网页构建副本
```

每次明确选择一个 processed 版本供评估、索引和网页使用。网页副本只由构建流程生成；网页构建包保留可直接通过 HTTP 预览的相对资源路径，不使用依赖开发机的符号链接。

## 4. 迁移映射与特殊处理

| 现有位置 | 建议位置 | 迁移注意事项 |
| --- | --- | --- |
| `code_scraping/` | `src/weightloss/ingestion/` | 保留分页、页面身份、去重及失败不发布保护 |
| `code_extraction/` | `src/weightloss/extraction/` 与 `experiments/` | 先区分全量正式入口与 top10 试验；移除导入时读数据 |
| `code_standardization/` | `src/weightloss/standardization/`、`evaluation/`、`experiments/` | 先提取被 standardize_all 使用的函数，不能整体归档旧模块 |
| `code_embedding/` | `src/weightloss/embeddings/`、`experiments/`、`notebooks/` | 区分术语嵌入与历史说明书试验 |
| `code_chatbot/` | `src/weightloss/retrieval/` | 保留现有两类检索；目录重构不改变检索方案 |
| `code_pipeline/` | `src/weightloss/pipeline/` | 所有输入和输出可配置，默认保持行为 |
| `drug_catalog.py` / `dataset_provenance.py` | 包内 catalog / provenance | 移动前解除 ROOT 与文件位置耦合；默认目录不得误落到 src |
| `code_website/` | `apps/web/` + `artifacts/web/` | 源码与构建副本分开，更新 Flask、HTML、JS 和测试路径 |
| `config/drugs.json` | `configs/drugs.json` | 网站配置仍由同一源生成 |
| `data_webmd/` | `data/raw/webmd/<snapshot>/` | 日期取 collection manifest，不臆造采集日期；按字节保留 CSV |
| `data_extracted/` | 当前到 `data/interim/`；旧样本按实验归类 | 不把 raw10/top10 当当前全量结果 |
| `data_standardized/` | 当前到 `data/processed/`；旧实验到 `results/legacy/` | v2–v6 与对应报告配对保存；无法核实的配置标为 unknown |
| `data_embedded/ae.csv` | `data/external/terminology/` | 这是输入术语表，不是缓存 |
| `data_embedded/embedded_ae.csv` | `artifacts/embeddings/` | 当前仍被标准化依赖；来源与可重建性核实前继续保留 |
| 两处 `database_table/` | 历史索引到 archive；未来新索引到 artifacts | 先清点校验和；历史索引保持 stale 标记，不能用新 manifest 冒充有效索引 |
| `data_backup/pre_2026_refresh/` | 被读取的注释文件到 `data/external/annotation_seeds/` | 同步修改 refresh 输入；保留旧 manifest 与路径映射；其他材料逐项判断 |
| 其余 `data_backup/` | `archive/` | 包含已排除药物历史数据，保留排除语境，不并回现用样本 |
| `data_prescribing_information/` | `data/external/prescribing_information/` | 保留历史日期与局限说明 |
| `data_literature/` | `docs/references/` | 保留来源；不是当前研究样本 |
| `.cache/` | 保持缓存目录 | 检查采集 checkpoint 是否是唯一保存的来源响应，再确定清理策略 |

## 5. 分阶段执行计划

### P0：冻结可比较的基线

- 记录 Git revision、已有修改、受跟踪文件清单、数据校验和及当前测试输出。
- 生成迁移映射：旧路径、新路径、角色、是否当前运行依赖、校验和。
- 为当前快照记录 2,727 条数据、4 个通用名、8 个品牌、2,344 条历史复用与 383 条 pending 的基线状态。
- 验收：所有当前输入与受引用的历史产物均能定位；冻结输入后迁移不得改变其字节。

### P1：先解除路径与导入耦合

- 增加包定义和统一路径配置；数据目录通过配置/命令参数传入，包安装位置与研究数据位置独立。
- 将模块级 CSV 读取、模型初始化移入函数；把当前入口依赖的旧模块功能拆出。
- 按功能分组依赖：采集、模型抽取、标准化、检索、网页、开发测试。选择一套环境锁定方式并在干净环境验证，保留旧环境记录供历史实验参考。
- 添加统一 CLI；原脚本暂作薄包装，便于兼容 README 与既有操作。
- 验收：从不同工作目录可运行轻量校验；导入基础模块不读数据、不下载模型、不请求外部服务；现有回归通过。

### P2：数据与实验归位

- 按映射逐模块迁移；先更新读取逻辑，再切换规范路径，不留下两份可独立修改的主数据。
- 同步更新 `.gitignore` 与 `.gitattributes` 中依赖旧路径的规则，延续研究数据按字节保留的策略；不改写既有 Git 历史。
- 用采集快照 ID 区分原始数据，用 run ID 区分处理中间状态与最终结果；阶段完成后冻结结果，更新 run 指针必须显式。
- 将 UMLS 历史版本与报告配对归档；保留原文件名、哈希与旧路径索引，不补造历史 prompt、模型修订或运行日期。
- 抽取 checkpoint 写到当前 run 的工作位置；只有通过校验的结果才成为可供网站与检索消费的版本。
- 将网页复制动作改为独立构建步骤，保证输出目录可以静态预览。
- 验收：CSV 与历史文件迁移前后哈希一致；pending 语义、样本次序、品牌覆盖、网页回答与旧索引拒绝逻辑保持一致。

### P3：补齐研究复现记录与自动化

- 每次运行保存 manifest：run ID、开始/结束时间、状态、Git commit 和 dirty 状态、环境锁文件哈希、输入文件哈希、依赖的注释种子、配置、prompt/schema 哈希、模型 ID/可获得的修订信息、术语版本、参数、随机种子（若适用）、输出哈希、失败/重试及覆盖计数。
- LLM 调用记录可用的请求 ID 与输入输出关联；响应及研究文本按现有数据访问范围保存。外部模型存在非确定性，目标是可追溯、可回放和可解释的重复实验，不承诺再次调用逐字相同。
- 增加命令入口，例如 `make validate`、`make test`、`make reproduce-fixture`、`make build-web`；这些是拟新增命令，当前还不能运行。
- 将在线采集、模型调用、索引构建及 Neo4j 导入作为显式步骤；离线复现使用冻结输入和固定响应样例。
- CI 默认运行离线测试与小样例流程；数据库或模型集成测试单列，不把缺少凭据的跳过报告成已通过。
- 验收：新环境能从合成样例运行到网页数据与评估报告；关键结果能沿 manifest 找到准确输入和配置。

### P4：整理协作与论文交付文档

- 更新 README、数据字典、运行指南、CONTRIBUTING 与结果索引，逐条修复旧路径链接。
- 有论文结果后，为图表记录结果文件、生成命令与 run ID；明确探索性结果和正式结果。
- 软件许可、作者引用信息和数据发布范围分别落实。TODO 已记录评论全文发布问题；路径重构不自动改变数据发布范围。
- 验收：合作者按文档完成离线流程；未知的历史配置、无法获取的外部输入和未验证的在线步骤被明确标注。

建议先完成 P0–P2，再完成 P3；P4 随进展更新。每个阶段单独提交，便于定位回归和恢复，不需要一次性移动整个仓库。迁移期间不重新采集、不重新抽取，以免结构变化与数据变化混在同一基线里。

## 6. 验收与本次已执行的检查

2026-09-14 在当前本机环境执行：

| 检查 | 结果 |
| --- | --- |
| `python3 code_pipeline/validate_data.py` | passed；2,727 records、4 generics、8 brands；规范 CSV 与网站副本一致 |
| `python3 -m unittest discover -s tests -p 'test_*.py'` | 15 tests，OK |
| `node tests/review-assistant.test.js` | 3,963 checks，28 pairs，passed |

这些证明当前数据一致性与已有测试通过，不代表已验证全量 LLM 抽取、医学术语正确性、FAISS 重建、Neo4j 服务或跨平台安装。本次未调用模型、重新采集或修改数据库，也未实施重构。

未来重构需保留上述回归，并补充：可安装包的跨目录入口测试、无副作用导入测试、注释种子迁移后的刷新等价性、网页静态资源加载测试和小样例端到端测试。结构性迁移要求冻结产物哈希相同；在线模型重新运行则独立记录并评估差异。

## 7. 范围边界

本计划不决定 UMLS / MedDRA 的选择、图谱 schema、是否移除 TableRAG 或 Reddit 收集方法；它们仍以 TODO 的未决状态为准。已记录的标准化错误与缺失注释也不会通过目录重构自动修好，需另行完成方法修复与验证。

对于当前规模，单包、配置文件、manifest 和轻量自动化已足够；等数据量、实验数量或协作需求明确增长后，再评估是否引入额外的数据版本和实验管理服务。

实施收尾说明：P1 中“暂作薄包装”的过渡已结束，旧 code_* 入口已移除；测试已按 unit/integration 分层。本地隐藏环境和缓存不属于上述源码目录图。
