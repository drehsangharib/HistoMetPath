# Config-driven pseudo-slide MIL runner

Run from the repository root:

```powershell
py -3.11 analysis\run_mil_pipeline.py --config configs\mil.yaml
```

The configuration uses JSON syntax saved as `.yaml`. JSON is valid YAML 1.2 and can
be read without an additional dependency. Change `model.pooling` to `mean` or `max`.

Outputs are written to `outputs/mil_config_run/` by default:

- `mil_summary.json`
- `mil_metrics.csv`

## Scientific boundary

This command evaluates synthetic bags derived from PCAM patch embeddings. The output
must be described as pseudo-slide MIL performance, not native WSI or patient-level
clinical performance. Threshold calibration uses the training partition and metrics
are reported on the held-out validation partition.
