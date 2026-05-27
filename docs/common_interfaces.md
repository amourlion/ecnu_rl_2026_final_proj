# Common Data and Environment Interfaces

This project uses only the 2501 projects listed in `data/project_list.csv` for formal experiments.

The shared pipeline writes parquet artifacts under `artifacts/processed/`:

- `projects.parquet`: project metadata from `data/project/`.
- `entries.parquet`: parsed entry-level submission records.
- `workers.parquet`: worker quality with invalid or missing values filled from the train split mean.
- `events.parquet`: chronologically ordered worker arrival events.
- `splits.json`: event ids assigned to train, valid, and test by time.

Experiments should depend on `shared.envs.CrowdsourcingRecEnv` and `shared.metrics.evaluate_agent` instead of parsing raw JSON or redefining reward and metrics logic.

Build artifacts:

```bash
.venv/bin/python scripts/build_artifacts.py --data-dir data --output-dir artifacts/processed
```
