import copy,json,tempfile
from pathlib import Path
import pytest,yaml
from analysis.run_frozen_external_execution_engine import atomic_json,consume_lock,synthetic_integration

def cfg():return yaml.safe_load(open('configs/evaluation/frozen_external_execution_engine_runtime.yaml',encoding='utf-8'))
def test_runtime_execution_is_disabled_by_default():
 c=cfg();assert c['external_execution_enabled'] is False;assert c['real_wsi_access_enabled'] is False;assert c['execution_count_limit']==1;assert c['primary_threshold']==0.2404209436418631
def test_consumption_transition_and_refusal():
 original={'lock_status':'sealed_pre_execution','execution_authorized':True,'execution_count_limit':1,'execution_count':0};started=consume_lock(original);assert original['execution_count']==0;assert started['execution_count']==1;assert started['lock_status']=='execution_started'
 with pytest.raises(RuntimeError):consume_lock(started)
def test_atomic_json_replaces_without_tmp_files():
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'x.json';atomic_json(p,{'a':1});assert json.loads(p.read_text())=={'a':1};assert not list(Path(td).glob('*.tmp'))
def test_synthetic_integration_never_mutates_real_lock_fixture():
 c=cfg();lock={'lock_status':'sealed_pre_execution','execution_authorized':True,'execution_count_limit':1,'execution_count':0};before=copy.deepcopy(lock);report=synthetic_integration(c,lock);assert report['passed'] is True;assert report['real_wsi_accessed'] is False;assert report['real_lock_mutated'] is False;assert report['external_execution_consumed'] is False;assert lock==before
