import json
from pathlib import Path
import numpy as np
from analysis.run_mil_pipeline import run


def test_config_driven_pipeline(tmp_path: Path):
    rng = np.random.default_rng(42)
    bags = []
    labels = []
    for label in [0, 1] * 10:
        center = float(label) * 1.5
        bags.append(rng.normal(center, 0.2, size=(6, 8)).astype(np.float32))
        labels.append(label)
    embeddings = tmp_path / 'bags.npy'
    targets = tmp_path / 'labels.npy'
    np.save(embeddings, np.array(bags, dtype=object))
    np.save(targets, np.asarray(labels, dtype=np.int64))
    output_dir = tmp_path / 'outputs'
    config = {
        'seed': 42,
        'data': {'slide_embeddings': str(embeddings), 'slide_labels': str(targets)},
        'split': {'validation_fraction': 0.30},
        'model': {'pooling': 'mean', 'max_iter': 500},
        'evaluation': {'bootstrap_samples': 25},
        'output': {'directory': str(output_dir)},
    }
    config_path = tmp_path / 'mil.yaml'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    summary = run(config_path, project_root=tmp_path)
    assert summary['scientific_scope'].startswith('synthetic PCAM pseudo-slide')
    assert summary['train_bags'] + summary['validation_bags'] == 20
    assert (output_dir / 'mil_summary.json').is_file()
    assert (output_dir / 'mil_metrics.csv').is_file()
