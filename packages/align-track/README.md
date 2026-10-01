# align-track

List align-system run folders, ADM names, and alignment targets. Install with the [repository setup](../../README.md#install).

## List runs

From the repository root, pass one run folder or a parent containing runs:

```bash
uv run --no-sync list-runs /path/to/runs
```

Run folders need `input_output.json`; `.hydra/config.yaml`, `timing.json`, and `scores.json` are read when present. The table shows **Run Path**, **ADM Name**, **Alignment**, and **Scenarios** (currently the input record count).
