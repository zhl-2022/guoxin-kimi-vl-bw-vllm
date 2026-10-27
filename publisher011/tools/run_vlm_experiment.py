"""Bounded, owned model runs; one process per SCNet GPU instance."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import uuid
import zipfile

BASE=Path('/root/private_data/guoxin-session010')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('model',choices=['kimi','robo']);args=ap.parse_args()
    assert Path('/dev/kfd').exists(),'Requires allocated BW device'
    assert json.loads((BASE/'READY.json').read_text())['ready']
    project=BASE/'projects'/args.model
    name='kimi-vl-bw-vllm' if args.model=='kimi' else 'robobrain-bw-vllm'
    task=project/'tasks'/name
    port=30129 if args.model=='kimi' else 30001
    for candidate in (30000,30001,30002,30119,30129,30139):
        with socket.socket() as sock:
            sock.settimeout(.5)
            assert sock.connect_ex(('127.0.0.1',candidate))!=0,'A prior service is active'
    result=BASE/(args.model+'001');result.mkdir(exist_ok=False)
    env=os.environ.copy();env.update(PYTHONPATH=str(project/'scripts'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    run_id=uuid.uuid4().hex
    record={'UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'run_id':run_id,'model':args.model,
            'phases':{},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'submitted':False}
    if args.model=='kimi':
        env.update(KIMI_MAX_LEN='32768',KIMI_MAX_RUNTIME_SECONDS='6900',KIMI_LOAD_TIMEOUT='900',KIMI_SERVED_NAME='kimi-vl-a3b-instruct')
        driver=project/'scripts/kimi_vl_runtime_005.py';mode='mmmu';max_tokens='512';model_name='kimi-vl-a3b-instruct'
        start=[sys.executable,str(driver),'launch','--task-dir',str(task),'--run-id',run_id]
    else:
        env.update(ROBOBRAIN_MAX_RUNTIME_SECONDS='9000',ROBOBRAIN_LOAD_TIMEOUT='1800',ROBOBRAIN_MAX_NUM_SEQS='4')
        driver=project/'scripts/recon2reason_runtime.py';mode='cvbench';max_tokens='4096';model_name='robobrain2-5-8b-nv'
        start=['bash',str(task/'run.sh'),'--run-id',run_id]
    (result/'run-start.json').write_text(json.dumps(record,indent=2))
    def step(argv,label,timeout):
        with (result/(label+'.log')).open('xb') as log:
            p=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:code=p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
                code=124
        record['phases'][label]=code;print(json.dumps({'phase':label,'exit':code}),flush=True)
        return code
    try:
        code=step([sys.executable,'-c',"import torch,json,importlib.metadata as m; assert torch.cuda.is_available(); print(json.dumps({'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'vllm':m.version('vllm'),'HIP':torch.version.hip}))"],'environment',90)
        if code:raise RuntimeError('GPU preflight failed')
        if step(start,'launch',2100):raise RuntimeError('Service did not launch')
        step([sys.executable,str(BASE/'tools/evaluate_vlm.py'),'--mode',mode,'--data',str(BASE/mode),
              '--output',str(result/'evaluation'),'--endpoint',f'http://127.0.0.1:{port}','--model',model_name,
              '--max-tokens',max_tokens],'evaluation',6000 if args.model=='kimi' else 7800)
    except Exception as exc:record['error']=str(exc)
    finally:
        step([sys.executable,str(driver),'stop','--task-dir',str(task),'--run-id',run_id],'stop',90)
        with socket.socket() as sock:
            sock.settimeout(1);record['service_stopped']=sock.connect_ex(('127.0.0.1',port))!=0
        if (task/'runtime').is_dir():shutil.copytree(task/'runtime',result/'runtime')
        record.update(ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),official_acceptance=False)
        score=result/'evaluation/verify_result.json'
        if score.is_file():record['accuracy_result']=json.loads(score.read_text())
        (result/'run-final.json').write_text(json.dumps(record,indent=2))
        with zipfile.ZipFile(f'/root/session010-{args.model}001-results.zip','x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(result.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(result).as_posix())
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
