"""
Experiment manifest utilities.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def build_manifest(experiment_name, config, outputs):
    return {
        "experiment_name": experiment_name,
        "created_utc": datetime.utcnow().isoformat(),
        "config": config,
        "outputs": outputs,
    }


def save_manifest(manifest, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
