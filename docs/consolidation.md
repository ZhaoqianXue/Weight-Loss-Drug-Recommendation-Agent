# Consolidation decisions

The outer WeightLoss working directory is now the single repository root. The original Git metadata, commit history, branch, remote, and index were preserved. The initial migration was local; the user subsequently authorized publishing the consolidated version to GitHub.

## Conflict resolution

- Nine shared source/environment files were identical at inventory time and retained.
- `code_chatbot/chatbot.py`: retained the later outer prompts for review examples and numerical results. Removed embedded credentials and translated comments.
- `code_extraction/schema_extraction.py`: retained the outer schema and `model_dump()` serialization. Side-effect relations use the primary review drug. The historical associated-drug field is not restored because this migration preserves the later implementation rather than making an unvalidated scientific change.
- `code_standardization/standardization.py`: retained the outer version and replaced the credential literal with an environment lookup.
- `code_standardization/standardization_raw.ipynb`: retained outer cell sources, removed embedded credentials, and cleared outputs. The historical notebook remains in Git history.
- `code_website/chatbot.py`: replaced the duplicate implementation with a compatibility import of the canonical chatbot.
- Prescribing-information scripts: retained as experimental code, despite their deletion from the historical repository on June 2, 2025.
- Data differences, including the extraction sample, use the outer data unchanged. The baseline and all UMLS versions remain distinct.
- README: replaced the outdated overview with an inventory and explicit runtime limitations. The pre-migration modified README is in the external backup.

## English-only scope

Chinese comments, terminal messages, and chart labels in maintained files were translated into English. Notebook execution outputs were cleared and the standardization chart was regenerated from its saved analysis. Original historical Git objects are unchanged so the requested history remains intact.

## Recovery

An external backup at `../WeightLoss-consolidation-backup-20260905/` contains the original historical repository, outer source directories, and a SHA-256 manifest of the outer data. It excludes the removed presentation artifacts. This backup contains the original credential-bearing files and is not part of the consolidated repository.

The consolidated changes are prepared for a normal commit and push on main, incorporating the remote cleanup commits without rewriting published history. Historical commits may retain credentials; removing those would require a separate history-rewriting operation.
