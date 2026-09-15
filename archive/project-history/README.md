# Project history

These dated records describe earlier repository states. Their directory trees and command examples are historical, and are not instructions for the current checkout. Current structure and completed acceptance results are in [the simplification report](../../docs/simplification.md); current commands are in [the operations guide](../../docs/reproducibility.md).

- `refactoring/`: the original larger architecture, earlier migrations, planning and one-time audit tools.
- `history/`: collected execution notes and dataset audits.
- `operations/`: previous operation manifests, preserved byte-for-byte.
- `simplification/`: the current migration map, one-time acceptance script and verification record.

For paths embedded in immutable JSON or old scripts, apply `refactoring/migration-map.json` first, then `simplification/migration-map.json` (or its directory-level `moves.json`). Do not rewrite old scientific records to imply they ran under a newer configuration.
