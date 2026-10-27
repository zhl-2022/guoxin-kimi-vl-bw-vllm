"""Owned bounded vLLM max_num_seqs=1 diagnostic against the fixed12 HF inputs."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
import zipfile

BASE = Path('/root/private_data/guoxin-session011')
OLD = Path('/root/private_data/guoxin-session010')
PROJECT = OLD / 'projects/kimi'
TASK = PROJECT / 'tasks/kimi-vl-bw-vllm'
DRIVER = PROJECT / 'scripts/kimi_vl_runtime_005.py'


def main():
    out = BASE / 'kimi-single-sequence001'
    out.mkdir(exist_ok=False)
    run_id = uuid.uuid4().hex
    env = os.environ.copy()
    env.update(KIMI_MAX_LEN='32768', KIMI_MAX_RUNTIME_SECONDS='1500', KIMI_LOAD_TIMEOUT='900',
               KIMI_SERVED_NAME='kimi-vl-a3b-instruct', PYTHONPATH=str(PROJECT / 'scripts'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    record = {'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'complete': False,
              'not_full_accuracy_evaluation': True, 'run_id': run_id, 'max_num_seqs': 1,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'phases': {}}
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('18-minute diagnostic cap')))
    signal.alarm(1080)
    def step(args, label, timeout):
        with (out / (label + '.log')).open('xb') as log:
            p = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = p.wait(timeout=timeout)
            except BaseException:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait()
                raise
        record['phases'][label] = code
        assert code == 0, label
    try:
        with socket.socket() as sock:
            assert sock.connect_ex(('127.0.0.1', 30129)) != 0
        assert '"--max-num-seqs", "1"' in DRIVER.read_text()
        step([sys.executable, str(DRIVER), 'launch', '--task-dir', str(TASK), '--run-id', run_id], 'launch', 900)
        sys.path.insert(0, str(OLD / 'tools'))
        from evaluate_vlm003 import payload
        plan = json.loads((BASE / 'kimi-preparation002/plan.json').read_text())
        samples = {s['id']: s for s in (json.loads(x) for x in (OLD / 'mmmu/samples.jsonl').read_text().splitlines())}
        old = {r['id']: r for r in (json.loads(x) for x in (OLD / 'kimi004/evaluation/predictions.jsonl').read_text().splitlines())}
        native = {r['id']: r for r in (json.loads(x) for x in (BASE / 'kimi-reference001/predictions.jsonl').read_text().splitlines())}
        results = []
        # Order follows the predeclared fixed12 plan; no label-based selection.
        with (out / 'predictions.jsonl').open('x') as log:
            for key in plan['selected_ids']:
                body = payload(samples[key], OLD / 'mmmu', 'mmmu', 'kimi-vl-a3b-instruct', 4096)
                original_hash = hashlib.sha256(json.dumps(body, ensure_ascii=False).encode()).hexdigest()
                assert original_hash == old[key]['request_sha256']
                body['max_tokens'] = 128
                raw = json.dumps(body, ensure_ascii=False).encode()
                start = time.monotonic()
                req = urllib.request.Request('http://127.0.0.1:30129/v1/chat/completions', raw, {'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=150) as response:
                    value = json.loads(response.read(8 * 1024**2))
                text = value['choices'][0]['message']['content']
                prefix = 0
                for a, b in zip(text, native[key]['native_text']):
                    if a != b:
                        break
                    prefix += 1
                row = {'id': key, 'original_4096_request_sha256': original_hash,
                       'probe_request_sha256': hashlib.sha256(raw).hexdigest(), 'only_generation_budget_changed': True,
                       'text': text, 'raw_response': value, 'native_text': native[key]['native_text'],
                       'common_prefix_with_native': prefix, 'old_vllm_text': old[key]['text'],
                       'wall_seconds': time.monotonic() - start}
                log.write(json.dumps(row) + '\n'); log.flush(); results.append(row)
                print(json.dumps({'done': len(results), 'id': key, 'common_prefix_with_native': prefix}), flush=True)
        record.update(complete=True, count=len(results), max_tokens=128)
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        signal.alarm(0)
        try:
            step([sys.executable, str(DRIVER), 'stop', '--task-dir', str(TASK), '--run-id', run_id], 'stop', 90)
        except Exception as exc:
            record['stop_error'] = repr(exc)
        with socket.socket() as sock:
            record['service_stopped'] = sock.connect_ex(('127.0.0.1', 30129)) != 0
        record['ended_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        with zipfile.ZipFile('/root/session011-kimi-single-sequence001.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in out.rglob('*'):
                if path.is_file():
                    archive.write(path, path.relative_to(out).as_posix())
            archive.write(Path(__file__), 'executed-kimi_single_sequence_probe.py')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
