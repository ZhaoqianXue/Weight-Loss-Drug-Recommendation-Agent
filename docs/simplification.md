# 项目简化验收

已完整实施批准的九目录方案。验收日期：2026-09-14（America/Phoenix）。

## 最终结构

```text
src/weightloss/
apps/web/
configs/
data/              # raw / external / interim / processed / indexes
results/<run_id>/  # run 记录、环境清单、指标及 demo/
tests/
requirements/
docs/
archive/
```

实际磁盘上的可见一级目录恰为以上九个。旧 experiments、notebooks、artifacts、scripts 已退出根目录，没有兼容入口或过渡空目录。历史材料集中在 archive；维护中的 Python 模块和网页源码分别保留在 src/weightloss 与 apps/web。

## 已完成的改动

- 历史实验、笔记本、结果、环境和项目记录归档；旧操作 manifest 保持原始字节。
- 术语嵌入 CSV 与 provenance 移到 data/external/terminology，仍为标准化输入。两份本地历史嵌入缓存保留在 .cache/legacy。
- 现存配置以及 new-run 和 fixture 默认值全部更新：网页输出为 results/<run_id>/demo，检索索引为 data/indexes/<run_id>。
- 新增 make reproduce、make demo，连同 make install、make test 构成四个主要入口；CLI 的研究命令继续可用。
- 可选依赖只保留 demo、research、dev。两份锁重新生成并安装验证，旧锁留档；研究环境移除 14 个无当前消费者的依赖包。
- 每个 run 按环境指纹保存一次清单，阶段日志引用并校验它；阶段参数、源码、输入输出哈希、模型标识、失败和覆盖记录继续保留。
- 普通 build-web 只更新当前 build manifest；与研究操作共享写锁，并记录选定 run、配置、数据和现有冻结记录的哈希。合成复现的 demo 引用最终冻结记录。
- README、当前指南、数据和结果说明、路线图及历史文档链接同步。旧审计退出默认 Make/CI，日常测试增加本地 Markdown 链接检查。

## 实际验收结果

| 检查 | 结果 |
| --- | --- |
| 迁移前旧审计 | 114 项映射、91 项冻结文件通过 |
| 最终九目录与迁移审计 | 193 项映射、111 项受保护文件字节一致；串联旧映射的 91 项冻结哈希再次通过 |
| 干净轻量环境安装 | make install 成功，17 个已安装包的依赖关系检查通过 |
| 干净研究环境安装 | 91 个已安装包的依赖关系检查通过；本地 .venv-research 也已同步 |
| Python 回归 | 33 项通过；保留原 27 项，新增 5 项日志/恢复检查及 1 项文档检查 |
| 研究适配器 | 5 项通过，模型和服务调用模拟，导入检查禁止下载和模型初始化 |
| 前端 | 3,963 项断言通过，覆盖 28 个药物对、8 个品牌、2,727 条记录 |
| make reproduce | 全流程通过；双次输出一致性由集成测试核验 |
| make demo | 实际启动并访问 7 个 HTTP 资源，全部 200；验证后关闭测试服务 |
| 研究数据 | 2,727 条、2,344 历史复用、383 pending；抽取/标准化哈希不变 |
| 当前文档 | 所有本地 Markdown 链接目标检查通过；当前代码、配置及指南无旧结构路径残留 |

安装和研究适配器验证平台为 Intel macOS 14.1、Python 3.11.4；前端使用 Node.js 22.15.0。Linux/ARM 研究环境及远程 CI 不在本次实际验证范围。未进行在线重新采集、真实模型实验或 Neo4j 写入；论文实验质量仍需单独验证。

## 证据与追溯

- [机器可读验收结果](../archive/project-history/simplification/verification.json)
- [执行日志](../archive/project-history/simplification/checks/)
- [本次逐文件迁移映射](../archive/project-history/simplification/migration-map.json)与[目录映射](../archive/project-history/simplification/moves.json)
- [一次性验收脚本](../archive/project-history/simplification/verify.py)，不加入日常 CI
- [批准的原方案](../archive/project-history/refactoring/simplification-plan.md)

冻结历史 JSON、数据和脚本中的旧路径保留原文，通过两轮迁移映射解析。历史 Markdown 中的旧目录图和执行命令属于当时的记录，当前使用说明以 README 和 docs/reproducibility.md 为准。
