from pathlib import Path
import yaml
from analysis.review_frozen_external_authorization import engine_ast_guards

def cfg():return yaml.safe_load(open('configs/evaluation/frozen_external_authorization_review.yaml',encoding='utf-8'))
def test_authorization_review_is_strictly_non_consuming():
 c=cfg();assert c['expected_execution_count']==0;assert c['execution_count_limit']==1;assert c['require_external_execution_disabled'] is True;assert c['require_real_wsi_access_disabled'] is True;assert c['prohibit_inference'] is True;assert c['prohibit_lock_mutation'] is True;assert c['prohibit_execution_token_use'] is True
def test_review_allows_exactly_its_four_precommit_files():
 c=cfg();assert set(c['allowed_precommit_paths'])=={'analysis/review_frozen_external_authorization.py','configs/evaluation/frozen_external_authorization_review.yaml','docs/FROZEN_EXTERNAL_AUTHORIZATION_REVIEW.md','tests/test_frozen_external_authorization_review.py'}
def test_final_engine_actual_guards_are_ast_validated():
 source=Path('analysis/run_frozen_external_execution_engine.py').read_text(encoding='utf-8');guards=engine_ast_guards(source);assert guards=={'execute_cli_mode':True,'dual_enablement_guard':True,'token_guard':True,'consume_before_encoder':True,'second_execution_refusal':True,'failure_sealing':True,'completion_sealing':True}
