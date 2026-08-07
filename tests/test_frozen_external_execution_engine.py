import json,tempfile
from pathlib import Path
import pytest,yaml
from analysis.validate_frozen_external_execution_engine import atomic_json,consume_fixture

def cfg():return yaml.safe_load(open('configs/evaluation/frozen_external_execution_engine.yaml',encoding='utf-8'))
def test_real_execution_is_disabled():
 c=cfg();assert c['real_wsi_access_enabled'] is False;assert c['external_execution_enabled'] is False;assert c['execution_count_limit']==1;assert c['coordinates_per_view']==300;assert c['embedding_features']==512;assert c['primary_features']==512;assert c['secondary_features']==1024;assert c['primary_threshold']==0.2404209436418631
def test_fixture_consumption_is_one_time():
 lock={'lock_status':'sealed_pre_execution','execution_authorized':True,'execution_count_limit':1,'execution_count':0};started=consume_fixture(lock);assert lock['execution_count']==0;assert started['execution_count']==1;assert started['lock_status']=='execution_started'
 with pytest.raises(RuntimeError):consume_fixture(started)
def test_atomic_fixture_write():
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'state.json';atomic_json(p,{'execution_count':1});assert json.loads(p.read_text(encoding='utf-8'))['execution_count']==1;assert not list(Path(td).glob('*.tmp'))
