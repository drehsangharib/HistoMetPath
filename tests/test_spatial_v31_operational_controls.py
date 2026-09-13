from pathlib import Path
import ast,importlib.util,json,pathlib,tempfile
import numpy as np
P = (Path(__file__).resolve().parents[1] / "core" / "wsi" / "run_camelyon16_spatial_sampler_v31.py")
s=importlib.util.spec_from_file_location('v31',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def arrays():return np.arange(600,dtype=np.int64).reshape(300,2),np.ones(300,dtype=np.float32)
def test_import_contract():assert callable(m.project_path) and callable(m.spatial_bin) and callable(m.allocate_density_budget)
def test_unique_arrays_accepted():c,f=arrays();assert m.validate_arrays(c,f,300)['unique_coordinate_count']==300
def test_duplicate_arrays_rejected():
 c=np.zeros((300,2),dtype=np.int64);f=np.ones(300,dtype=np.float32)
 try:m.validate_arrays(c,f,300)
 except RuntimeError as e:assert 'duplicate' in str(e)
 else:raise AssertionError('duplicate coordinates accepted')
def test_coordinate_dtype_rejected():
 c,f=arrays();c=c.astype(np.int32)
 try:m.validate_arrays(c,f,300)
 except RuntimeError as e:assert 'coordinate schema mismatch' in str(e)
 else:raise AssertionError('wrong coordinate dtype accepted')
def test_fraction_dtype_rejected():
 c,f=arrays();f=f.astype(np.float64)
 try:m.validate_arrays(c,f,300)
 except RuntimeError as e:assert 'tissue-fraction schema mismatch' in str(e)
 else:raise AssertionError('wrong fraction dtype accepted')
def test_nonfinite_fraction_rejected():
 c,f=arrays();f[0]=np.nan
 try:m.validate_arrays(c,f,300)
 except RuntimeError as e:assert 'non-finite' in str(e)
 else:raise AssertionError('nonfinite fraction accepted')
def test_missing_restart_returns_none():
 with tempfile.TemporaryDirectory() as td:assert m.validate_restart(td,{'slide':'s','label':'normal','path':'synthetic://one'},300) is None
def test_valid_restart_uses_canonical_source_path():
 row={'slide':'synthetic_001','label':'normal','path':'synthetic://one'}
 with tempfile.TemporaryDirectory() as td:
  d,cp,fp,rp=m.paths_for(td,row['slide']);d.mkdir();c,f=arrays();np.save(cp,c);np.save(fp,f);rp.write_text(json.dumps({'slide':row['slide'],'label':row['label'],'source_path':str(pathlib.Path(row['path'])),'coordinate_sha256':m.sha256_file(cp),'tissue_fraction_sha256':m.sha256_file(fp)}));assert m.validate_restart(td,row,300) is not None
def test_restart_rejects_bad_hash():
 row={'slide':'synthetic_001','label':'normal','path':'synthetic://one'}
 with tempfile.TemporaryDirectory() as td:
  d,cp,fp,rp=m.paths_for(td,row['slide']);d.mkdir();c,f=arrays();np.save(cp,c);np.save(fp,f);rp.write_text(json.dumps({'slide':row['slide'],'label':row['label'],'source_path':str(pathlib.Path(row['path'])),'coordinate_sha256':'0'*64,'tissue_fraction_sha256':m.sha256_file(fp)}));assert m.validate_restart(td,row,300) is None
def test_static_read_accounting_and_no_heavy_imports():
 text=P.read_text();tree=ast.parse(text);calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func).endswith('read_region')];assert len(calls)==1;assert 'region_reads+=1' in text and 'successful_region_reads+=1' in text;assert 'run_camelyon16_spatial_sampler import' not in text;assert 'run_camelyon16_spatial_sampler_v2 import' not in text
def test_atomic_and_execution_gates_present():
 text=P.read_text();assert 'tempfile.TemporaryDirectory' in text and 'os.replace(td,final_dir)' in text;assert '--execute-spatial-v31' in text
