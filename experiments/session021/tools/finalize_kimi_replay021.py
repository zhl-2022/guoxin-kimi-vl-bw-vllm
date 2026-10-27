"""Replay the unchanged fixed Git root entries after the combined 12-report audit."""
from pathlib import Path,PurePosixPath
import datetime as dt
import hashlib
import json
import os
import resource
import socket
import subprocess
import time
import uuid
import zipfile

BASE=Path('/root/private_data/guoxin-session021/coordinator001')
OUT=Path('/root/private_data/guoxin-session021/kimi-portable-replay002')
ZIP=Path('/root/session017-kimi-portable-project.zip')
ZIP_SHA='d5a4da75bf8c373c6abeb467c0e25c5199a7a8db01f58c09d86d2d7a7bcf29d2'
COMMIT='031051314c2a79ee76a9fbd57d363ba11244719b'


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):value.update(chunk)
    return value.hexdigest()


def save(record):
    temporary=OUT/'progress.tmp'
    temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(OUT/'progress.json')


def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    assert sha(ZIP)==ZIP_SHA
    OUT.mkdir(parents=True,exist_ok=False)
    project=OUT/'project';project.mkdir()
    with zipfile.ZipFile(ZIP) as bundle:
        assert bundle.testzip() is None
        for item in bundle.infolist():
            path=PurePosixPath(item.filename)
            assert not path.is_absolute() and '..' not in path.parts and '\\' not in item.filename
            assert (item.external_attr>>16)&0o170000 != 0o120000
        bundle.extractall(project)
    manifest=json.loads((project/'publisher016-manifest.json').read_text())
    for row in manifest['files']:
        p=project/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    patch=subprocess.run(['bash',str(project/'scripts/apply_profile016.sh')],capture_output=True,text=True,timeout=30)
    assert patch.returncode==0,patch.stderr
    record={'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'controller_pid':os.getpid(),
            'state':'checking_combined_performance','git_commit':COMMIT,'git_archive_sha256':ZIP_SHA,
            'source_sha256':sha(__file__),'root_entry_replayed':False,'phases':{},
            'submitted':False,'instance_shutdown_requested':False}
    save(record)
    combined=BASE/'combined-native-matrix.json'
    proof=json.loads(combined.read_text())
    assert proof['complete_native_matrix'] and proof['measured_requests']==608
    record.update(state='combined_performance_verified',complete_native_matrix_verified=True,
                  combined_proof_sha256=sha(combined),raw_sources_preserved=True)
    save(record)
    for port in (30000,30001,30109,30119,30129,30139):
        with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0,('Other active service',port)
    run_id=uuid.uuid4().hex
    env=os.environ.copy();env.update(KIMI_DATA_DIR='/root/private_data/guoxin-session016/ocr-stage001/data',
                                     KIMI_PYTHON_BIN='/usr/bin/python3',KIMI_SERVED_NAME='/root/private_data/guoxin-candidates/models/kimi-vl')
    def step(command,label,seconds):
        with (OUT/(label+'.log')).open('xb') as log:
            child=subprocess.Popen(command,cwd=project,stdout=log,stderr=subprocess.STDOUT,env=env)
            try:value=child.wait(timeout=seconds)
            except BaseException:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        record['phases'][label]=value;save(record);return value
    launched=False
    try:
        record.update(state='root_entry_gpu_replay',replay_run_id=run_id);save(record)
        launched=True
        assert step(['bash',str(project/'run.sh'),'--run-id',run_id],'launch',1320)==0
        assert step(['bash',str(project/'env_check.sh')],'environment',180)==0
        assert step(['bash',str(project/'smoke.sh')],'smoke',900)==0
        verify=project/'scripts/verify_kimi_ocr016.py'
        data=env['KIMI_DATA_DIR']
        assert step(['python3',str(project/'verify.py'),'--selfcheck','--data-dir',data],'selfcheck',180)==0
        assert step(['python3',str(project/'verify.py'),'--data-dir',data,'--model',env['KIMI_SERVED_NAME'],
                     '--output',str(OUT/'full-accuracy')],'verify',5400)==0
        result=json.loads((OUT/'full-accuracy/verify_result.json').read_text())
        assert result['complete'] and result['numeric_pass']
        record.update(root_entry_replayed=True,accuracy_score=result['score']['Final Score'],accuracy_result_sha256=sha(OUT/'full-accuracy/verify_result.json'))
    finally:
        if launched:
            record['phases']['stop']=step(['bash',str(project/'stop.sh'),'--run-id',run_id],'stop',100)
        with socket.socket() as sock:record['owned_service_stopped']=sock.connect_ex(('127.0.0.1',30129))!=0
        complete=record['root_entry_replayed'] and record['phases'].get('stop')==0 and record['owned_service_stopped']
        record.update(state='complete' if complete else 'replay_or_cleanup_failed',ended_utc=dt.datetime.now(dt.timezone.utc).isoformat());save(record)
    entries=[];archive=Path('/root/session021-kimi-portable-replay002.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
        for directory,dirs,names in os.walk(OUT,onerror=lambda e:(_ for _ in ()).throw(e)):
            dirs[:]=sorted(d for d in dirs if d not in ('__pycache__','.git'))
            for name in sorted(names):
                p=Path(directory)/name;assert not p.is_symlink();rel=p.relative_to(OUT).as_posix()
                bundle.write(p,rel);entries.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
        bundle.writestr('MANIFEST.json',json.dumps({'scope':'full portable root-entry replay plus fixed project source','files':entries},indent=2))
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():
            p=OUT/'progress.json';r=json.loads(p.read_text()) if p.exists() else {}
            r.update(state='failed',error=repr(exc),ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),submitted=False);save(r)
        raise
