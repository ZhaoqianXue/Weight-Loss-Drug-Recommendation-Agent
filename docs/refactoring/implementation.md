# 架构重构实施报告

日期：2026-09-14。批准方案见 [架构计划](../architecture-refactoring-plan.md)。

## 完成内容

- 将维护代码收进可安装的 `src/weightloss/` 包，提供 `weightloss` 命令、集中配置及可选依赖组。临时兼容入口已在目录收尾中移除，统一使用 CLI。
- 将采集快照、处理中间数据、标准化结果、注释种子、术语输入、历史实验、文献和索引按职责归位；保留完整迁移映射和原始字节校验和。
- 当前数据设为冻结基线。新增 `new-run`、每次操作的输入/代码/环境记录、增量 checkpoint、验证与 `freeze`。不重新调用模型或采集数据。
- 将网页源码放在 `apps/web/`，构建到 `artifacts/web/`；网页数据来自唯一选定的 processed 数据集。静态预览不需要模型。Flask 的研究 chatbot 改为首次请求时初始化。
- 移除维护模块导入时读取研究 CSV、初始化模型或下载 NLTK 资源的行为；模型客户端与嵌入模型延迟创建。保留原有抽取、标准化、TableRAG 和 GraphRAG 方法。
- 增加合成数据与固定注释响应的离线回放、结构回归、模型适配层模拟测试、Makefile 和 CI 配置。
- 更新 README、协作指南、数据字典、运行指南、环境说明、结果索引及历史文档链接。

## 数据保护与迁移证据

[迁移前基线](baseline.json) 记录原 Git revision、130 个受跟踪文件哈希、原有数据校验及测试输出。[迁移映射](migration-map.json) 覆盖 114 项移动，标注哪些文件必须保持字节不变。运行 `python scripts/audit_migration.py` 可验证其中 91 个冻结文件。

当前原始、抽取和标准化数据仍为 2,727 条，4 个通用名和 8 个品牌；2,344 条历史注释复用、383 条 pending。标准化 CSV 的 SHA-256 仍为 `e4bb15982d884905efb723b9c0d68be6dc60e1e4ea56bfb4ebf0bb69607840d8`。

`pre_2026_refresh` 的注释仍是刷新输入，已迁入 `data/external/annotation_seeds/`。历史 FAISS 索引留在 archive，未添加伪造的新指纹。UMLS v2–v6、旧报告、笔记本和原依赖清单未被改写成当前结果。

## 验证

| 已执行检查 | 结果 |
| --- | --- |
| Python 离线回归 | 27 项通过 |
| 前端检查 | 3,963 项通过；28 组品牌对比 |
| 研究适配层模拟测试 | 5 项通过；未调用真实模型 |
| 迁移字节审计 | 114 项映射、91 个冻结文件通过 |
| 两个依赖环境兼容性 | 通过 |
| wheel 构建、独立安装及仓库外运行 | 通过 |
| 维护文档链接与 Python 语法 | 通过 |

本机执行记录汇总于 [verification.json](verification.json)。其中明确区分：

- 当前数据校验与迁移字节审计；
- 原有 15 项 Python 回归和新增结构回归；
- 前端 3,963 项检查、28 组品牌对比；
- 干净离线环境与研究环境；
- wheel 构建、独立环境安装、从仓库外运行；
- 模型适配层模拟测试，未进行真实模型调用；
- 两次合成样例运行的结果一致性、冻结保护和 HTTP/Flask 资源加载。

CI 配置覆盖 Ubuntu/macOS、Python 3.11 与 Node.js 22。本次在本机执行，未将尚未运行的远端 CI 记为通过。

## 环境适配

本机为 Intel macOS 14.1，Python 3.11.4 / Node.js 22.15.0。原 Python/uv 环境会报告 macOS 10.16，因此研究锁显式选择 macOS 14 的 wheel。Torch 在 Intel macOS 使用有可用 wheel 的 2.2.2，FAISS 保持 1.11.0；其他平台必须独立验证。离线与研究锁均保存版本和包哈希，原环境清单继续留档。

## 边界与保留决策

没有修改研究药物范围、术语方案、图谱 schema 或检索方法选择。没有重新爬取、调用付费模型、重建线上索引或修改 Neo4j 数据库。在线服务链路、模型输出医学正确性和跨平台研究环境不由本次离线检查证明。

LICENSE、CITATION.cff 的作者与授权信息尚未由项目提供；按照批准方案，不推测作者名单或代选许可。数据发布范围保持原决策状态。上述事项属于后续发布/方法工作，不影响已完成的结构迁移与本地离线复现。

全部改动保留在当前工作区，未改写 Git 历史。历史文档保留原日期语境，链接已尽可能指向迁移后文件。

## 目录收尾

已移除六个旧 `code_*` 目录及残留空目录，删除一次性 wheel 测试环境和仓库字节码缓存。离线及研究环境继续保留，属于隐藏的本地运行目录。测试已拆成 unit / integration / frontend / fixtures，研究适配测试归入 integration/research 并显式运行。

根目录的三份规划记录迁入 `docs/refactoring/planning/`；原 `outputs/chatbot-qa/task_plan.md` 按字节保存在 `docs/history/chatbot-qa-local-note.md`。迁移明细见 [cleanup-map.json](cleanup-map.json)，最新验证见 [cleanup-verification.json](cleanup-verification.json)。

新增 `scripts/audit_structure.py`，已加入 `make check` 和 CI，检查根目录、测试分层、已退役路径和包导入方式。源码结构不再保留临时兼容目录。
