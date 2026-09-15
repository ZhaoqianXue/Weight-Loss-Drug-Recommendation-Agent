# Results

Each `results/<run_id>/` contains the records and outputs for one selected run:

- `manifest.json`: initialization and parent input hashes.
- `operations/<operation_id>/manifest.json`: stage inputs, outputs, parameters, status and coverage.
- `environments/<fingerprint>.json`: environment inventory, shared by stage records in that run.
- `metrics.json`, `figures/`: produced only when the corresponding evaluation or figure is run.
- `frozen.json`: final validation, configuration and input hashes when explicitly frozen.
- `demo/`: generated site and current build manifest; ignored by Git.

The preserved `migrated-2026-09-14/manifest.json` describes a migrated baseline, not a new scientific experiment. Its original contents remain unchanged. Earlier operations are in [the operation archive](../archive/project-history/operations/), and historical experiment results are in [the results archive](../archive/results/). Old paths resolve through [the simplification map](../archive/project-history/simplification/migration-map.json).

Stage environment references are relative to the run directory. Provider events remain local and ignored by Git. Repeated ordinary web builds replace the demo manifest without creating full stage logs. Formal paper outputs should identify their run, generating command and digest. Synthetic smoke outputs normally live under `.cache/`.
