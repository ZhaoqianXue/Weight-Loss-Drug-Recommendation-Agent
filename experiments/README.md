# Experiments

Reusable maintained code is in `src/weightloss/`. New experiment configurations belong here; outputs and their manifests belong in `results/<run_id>/`.

`legacy/` preserves original extraction, UMLS standardization and evaluation scripts byte-for-byte. Their hard-coded old paths, environment assumptions and import-time execution are historical. They are intentionally not imported by the maintained package or CLI. Do not run them against the current frozen dataset. To reproduce an old experiment, reconstruct its inputs in a separate checkout/workspace using the migration map and the baseline Git revision, and document unavailable original settings as unknown.

Current standardization still uses the baseline algorithm extracted into `weightloss.standardization.baseline`. This reorganization does not adopt the proposed vocabulary or graph-schema changes in TODO. UMLS v2–v6 files and reports remain in `results/legacy/standardization/`; historical top10 extraction samples remain in `results/legacy/extraction/`.
