import yaml

def test_internal_release_excludes_runtime_binaries():
 cfg=yaml.safe_load(open('configs/release/histometpath_internal_release.yaml',encoding='utf-8'))
 patterns=set(cfg['exclude_patterns'])
 assert '*.npy' in patterns
 assert '*.ckpt' in patterns
 assert '*.joblib' in patterns
 assert '.git' in patterns
