# 审查依据

- 项目以 code_* 和 data_* 分组，已有 README、docs、tests、config。
- README 明确区分当前数据与历史实验；含数据 manifest、索引指纹及空数据库导入保护。
- 根目录没有统一的 Python 包和依赖环境定义；README 要求从仓库根目录运行脚本。
- TODO 中的图模式调整、术语方案和 TableRAG 移除仍是未决提案，目录重构不应代替这些决策。
- 开始检查时 git status --short 为空。
- 实际扫描：130 个 Git 跟踪文件；2 个 notebook 与脚本混放；data_standardized 同时含当前 CSV、top10、UMLS v2–v6 与报告。
- 多个维护入口 sys.path.insert；部分旧脚本在模块导入时读取数据或运行处理；table_loader/chatbot 使用相对工作目录路径。
- 公开依据：Wilson et al. (2017), Good enough practices in scientific computing, DOI 10.1371/journal.pcbi.1005510；Cookiecutter Data Science 官方 opinions 建议不可变 raw 与可再生产物。它们是实践指南而非唯一学术强制标准。
- 关键迁移约束：refresh_data.py 的 pre_2026_refresh 不是纯存档，而是当前注释复用的运行时输入，必须迁移为明确的 annotation seed 并保留 checksum。
- 关键迁移约束：standardize_all.py 导入 standardization.py 并依赖其模块级数据/模型；不能把整个旧标准化模块直接归档。
- 网站两份 CSV 由 publish_website 统一复制且受测试约束，属于已有控制的构建副本；重构应显式化发布边界，不应宣称现在是独立人工维护。
- 本机验证：数据一致性校验 passed，Python 15 tests OK，JavaScript 3963 checks passed。未验证外部模型和数据库链路。
- `.gitattributes` 中有 data_* 等旧路径规则，数据迁移时必须同步更新以保留字节级策略。

## 实施后的状态
上述缺口已按批准方案处理。当前 canonical 路径见 configs/pipeline.json；完整映射、基线和最终检查位于 docs/refactoring/。历史 LICENSE/作者元数据未提供，保留为独立发布决定，不代填。

目录收尾纠正：过渡 code_* 已移除，测试已按 unit/integration 分层；工作记录迁入 docs/refactoring/planning。目录合规现在由 scripts/audit_structure.py 显式验证。
