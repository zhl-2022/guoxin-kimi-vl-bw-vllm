"""Only the two unfinished 16K/c64 cases, same verified profile; finite larger wall budget."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
import zipfile

PREVIOUS = Path('/root/private_data/guoxin-session016/kimi-ocrbench004')
OUT = Path('/root/private_data/guoxin-session021/kimi-performance006')
MODEL = '/root/private_data/guoxin-candidates/models/kimi-vl'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    prior = Path('/root/private_data/guoxin-session016/kimi-performance004/run-final.json')
    prior_deadline = time.monotonic() + 300
    while not prior.exists():
        assert time.monotonic() < prior_deadline,'Prior failed workload has not finished cleanup'
        time.sleep(5)
    assert json.loads(prior.read_text())['owned_service_stopped']
    deadline = time.monotonic() + 3600
    while not (PREVIOUS / 'run-final.json').exists():
        assert time.monotonic() < deadline,'No completed accuracy replay; nothing else was stopped'
        time.sleep(15)
    previous = json.loads((PREVIOUS / 'run-final.json').read_text())
    assert previous['complete'] and previous['score']['Final Score'] >= 847 and previous['owned_service_stopped']
    assert previous['long_prefill_safe']
    for port in (30000,30001,30109,30119,30129,30139):
        with socket.socket() as sock:
            assert sock.connect_ex(('127.0.0.1',port)) != 0, ('Existing local service',port)
    OUT.mkdir(parents=True,exist_ok=False)
    project = OUT / 'project'
    shutil.copytree(PREVIOUS / 'project',project,ignore=shutil.ignore_patterns('runtime','__pycache__'))
    driver = project / 'scripts/kimi_vl_runtime_005.py'
    assert sha(driver) == previous['driver_sha256']
    hook = OUT / 'math-sdpa-hook'
    shutil.copytree(PREVIOUS / 'math-sdpa-hook',hook,ignore=shutil.ignore_patterns('__pycache__'))
    audit = OUT / 'math-sdpa-worker-audits'
    audit.mkdir()
    task = project / 'tasks/kimi-vl-bw-vllm'
    run_id = uuid.uuid4().hex
    env = os.environ.copy()
    env.update(KIMI_MAX_LEN='32768',KIMI_MAX_RUNTIME_SECONDS='21600',KIMI_LOAD_TIMEOUT='1200',
               KIMI_SERVED_NAME=MODEL,VLLM_MLA_DISABLE='1',KIMI_SDP_AUDIT_DIR=str(audit),
               PYTHONPATH=str(hook)+':'+str(project / 'scripts'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               ROCM_PATH='/opt/dtk',HIP_PATH='/opt/dtk/hip')
    env['PATH'] = '/opt/ucx/bin:/opt/dtk/bin:/opt/dtk/llvm/bin:/opt/dtk/hip/bin:/opt/hyhal/bin:' + env.get('PATH','/usr/bin:/bin')
    env['LD_LIBRARY_PATH'] = ':'.join(['/opt/ucx/lib','/opt/dtk/dcc/gcvm/lib','/opt/dtk/hip/lib',
        '/opt/dtk/llvm/lib','/opt/dtk/lib','/opt/dtk/lib64','/opt/hyhal/lib','/opt/hyhal/lib64',
        '/opt/dtk/dushmem/lib','/opt/dtk/opencl/lib','/opt/mpi/lib','/opt/hwloc/lib',env.get('LD_LIBRARY_PATH','')])
    original_wrapper = Path('/root/t02_sglang_bench_004g.py')
    assert sha(original_wrapper) == 'ff8eef852e19f336e99884cc031353627ca304c6a7534c22375116758a850000'
    wrapper = OUT / 't02_sglang_bench-executed.py'
    text = original_wrapper.read_text()
    before = '{"temperature":0,"ignore_eos":true,"stream_options":{"include_usage":true}}'
    after = '{"temperature":0,"ignore_eos":true,"stream_options":{"include_usage":true},"bad_words":["<|media_pad|>","<|placeholder|>","<|media_start|>","<|media_end|>"]}'
    assert text.count(before) == 1
    wrapper.write_text(text.replace(before,after))
    shutil.copyfile(Path(__file__),OUT / 'executed-controller.py')
    compat = OUT / 'client-processor-compat'
    compat.mkdir()
    compat_source = '''from pathlib import Path
import hashlib,json,os,runpy
runpy.run_path(os.environ['KIMI_CLIENT_MATH_HOOK'])
from transformers import AutoProcessor
_load_processor = AutoProcessor.from_pretrained
def _kimi_image_id(*args,**kwargs):
    processor = _load_processor(*args,**kwargs)
    if type(processor).__name__ == 'KimiVLProcessor':
        assert str(args[0]) == '/root/private_data/guoxin-candidates/models/kimi-vl'
        assert processor.image_token == '<|media_pad|>'
        token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        assert processor.tokenizer.convert_ids_to_tokens(token_id) == processor.image_token
        previous = getattr(processor,'image_token_id',None)
        assert previous is None or previous == token_id
        processor.image_token_id = token_id
        original_vocab = processor.tokenizer.get_vocab()
        reserved = set(processor.tokenizer.all_special_ids)
        for token,value in original_vocab.items():
            text = token.decode(errors='replace') if isinstance(token,bytes) else str(token)
            if text.startswith('<|') and text.endswith('|>'): reserved.add(value)
        # The upstream generator removes image_token_id itself. Keep that one
        # sentinel in this intermediate view so list.remove remains valid.
        ordinary_vocab = {token:value for token,value in original_vocab.items()
                          if value not in reserved or value == token_id}
        assert token_id in reserved and ordinary_vocab
        processor.tokenizer.get_vocab = lambda:dict(ordinary_vocab)
        row = {'pid':os.getpid(),'processor':type(processor).__name__,'previous_image_token_id':previous,
               'image_token_id':token_id,'image_token':processor.image_token,
               'random_text_reserved_ids_excluded':len(reserved),'random_text_vocab_size':len(ordinary_vocab),
               'model_files_changed':False,'upstream_sglang_source_changed':False}
        with Path(os.environ['KIMI_CLIENT_COMPAT_AUDIT']).open('a') as stream:
            stream.write(json.dumps(row)+'\\n')
    return processor
AutoProcessor.from_pretrained = _kimi_image_id
'''
    (compat / 'sitecustomize.py').write_text(compat_source)
    client_env = env.copy()
    client_env.update(PYTHONPATH=str(compat)+':'+str(project/'scripts'),
                      KIMI_CLIENT_MATH_HOOK=str(hook/'sitecustomize.py'),
                      KIMI_CLIENT_COMPAT_AUDIT=str(OUT/'client-processor-compat.jsonl'))
    record = {'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'run_id':run_id,
              'source_sha256':sha(Path(__file__)),'driver_sha256':sha(driver),'wrapper_sha256':sha(wrapper),
              'accuracy_run_final_sha256':sha(PREVIOUS / 'run-final.json'),'accuracy_score':previous['score']['Final Score'],
              'same_model_processor_math_SDPA_context_and_sequences':True,'served_alias_only_changed_for_client':True,
              'client_only_processor_image_token_id_compatibility':True,
              'synthetic_random_text_domain':'ordinary vocabulary; session and multimodal sentinels excluded',
              'synthetic_output_bad_words':['<|media_pad|>','<|placeholder|>','<|media_start|>','<|media_end|>'],
              'owned_process_core_dumps_disabled':True,
              'compat_source_sha256':sha(compat/'sitecustomize.py'),
              'complete':False,'phases':{},'submitted':False,'instance_shutdown_requested':False}
    (OUT / 'run-start.json').write_text(json.dumps(record,indent=2))
    def step(argv,label,seconds,environment=None):
        with (OUT / (label+'.log')).open('xb') as stream:
            child = subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=environment or env)
            try:
                value = child.wait(timeout=seconds)
            except BaseException:
                child.terminate()
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
                raise
        record['phases'][label] = value
        return value
    try:
        assert step([sys.executable,str(driver),'launch','--task-dir',str(task),'--run-id',run_id],'launch',1260) == 0
        value = step([sys.executable,str(wrapper),'--run',
            '--client-python','/root/private_data/sglang-client-004c/bin/python','--allow-client-device-discovery',
            '--model-path',MODEL,'--served-model-name',MODEL,'--port','30129',
            '--runtime-dir',str(task / 'runtime'),'--expected-run-id',run_id,
            '--output-dir',str(OUT / 'performance'),'--timeout-seconds','9000','--total-timeout-seconds','19000','--load','16384:64'],
            'benchmark',19100,client_env)
        if (OUT / 'performance/run.json').exists():
            report = json.loads((OUT / 'performance/run.json').read_text())
            record['performance_status'] = report['status']
            record['rows'] = [{k:r.get(k) for k in ['name','status','exit_code','client_wall_seconds','result']} for r in report['rows']]
            record['complete'] = value == 0 and report['status'] == 'complete_client_reports'
    except BaseException as exc:
        record['error'] = repr(exc)
    finally:
        try: step([sys.executable,str(driver),'stop','--task-dir',str(task),'--run-id',run_id],'stop',90)
        except BaseException as exc: record['cleanup_error'] = repr(exc)
        with socket.socket() as sock:
            record['owned_service_stopped'] = sock.connect_ex(('127.0.0.1',30129)) != 0
        record['ended_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        (OUT / 'run-final.json').write_text(json.dumps(record,indent=2))
        files = []
        with zipfile.ZipFile('/root/session021-' + OUT.name + '.zip','x',zipfile.ZIP_DEFLATED) as bundle:
            for folder,dirs,names in os.walk(OUT,onerror=lambda e:(_ for _ in ()).throw(e)):
                dirs[:] = sorted(d for d in dirs if d != '__pycache__')
                for name in sorted(names):
                    path = Path(folder)/name
                    assert not path.is_symlink()
                    relative = path.relative_to(OUT).as_posix()
                    bundle.write(path,relative)
                    files.append({'path':relative,'bytes':path.stat().st_size,'sha256':sha(path)})
            bundle.writestr('MANIFEST.json',json.dumps({'files':files},indent=2))
        print(json.dumps(record),flush=True)


if __name__ == '__main__':
    main()
