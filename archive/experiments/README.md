# Historical experiments

`extraction/`, `standardization/` and `evaluation/` preserve legacy scripts. `prescribing_information/` preserves the unused PI embedding scripts formerly packaged under `weightloss.embeddings`. Their old paths, environment assumptions and import-time behavior are retained byte-for-byte; the maintained CLI does not import them.

To reconstruct an old experiment, use a separate workspace and the [migration map](../project-history/simplification/migration-map.json), previous Git revision and [historical environments](../environments/). Missing original settings remain unknown. Do not run historical scripts against the frozen baseline.

Maintained code is in `src/weightloss/`; new run configurations belong in `configs/` and generated outputs in `results/<run_id>/`. The active standardization algorithm and terminology embeddings remain available. [Historical results](../results/) and the [undecided research roadmap](../../docs/roadmap.md) retain their scientific status.
