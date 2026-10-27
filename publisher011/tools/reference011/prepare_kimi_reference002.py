"""No-card checkpoint audit and isolated official Transformers reference setup."""
from pathlib import Path
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys

BASE = Path('/root/private_data/guoxin-session011')
OLD = Path('/root/private_data/guoxin-session010')
MODEL = Path('/root/private_data/guoxin-candidates/models/kimi-vl')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert not Path('/dev/kfd').exists(), 'Preparation must not allocate a GPU'
    BASE.mkdir(exist_ok=True)
    out = BASE / 'kimi-preparation002'
    out.mkdir(exist_ok=False)
    lock = json.loads((OLD / 'projects/kimi/tasks/kimi-vl-bw-vllm/model-lock.json').read_text())
    assert lock['revision'] == '398eede0903cd983a2bfa0cc634e9ac1d843f375'
    record = {'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'model': str(MODEL),
              'revision': lock['revision'], 'files': [], 'GPU_allocated': False}
    try:
        for row in lock['files']:
            path = MODEL / row['path']
            assert path.is_file() and not path.is_symlink() and path.stat().st_size == row['size']
            actual = sha(path)
            assert actual == row['sha256'], row['path']
            record['files'].append({'path': row['path'], 'bytes': path.stat().st_size, 'sha256': actual})
            print(json.dumps({'hashed': row['path']}), flush=True)
        samples = [json.loads(line) for line in (OLD / 'mmmu/samples.jsonl').read_text().splitlines()]
        assert len(samples) == 900
        groups = [([s for s in samples if s['question_type'] == 'multiple-choice' and len(s['images']) == 1], 8),
                  ([s for s in samples if s['question_type'] == 'multiple-choice' and len(s['images']) > 1], 2),
                  ([s for s in samples if s['question_type'] != 'multiple-choice'], 2)]
        selected = []
        for values, count in groups:
            assert len(values) >= count
            selected.extend(sorted(values, key=lambda s: hashlib.sha256(('kimi-framework-parity011:' + s['id']).encode()).digest())[:count])
        ids = [s['id'] for s in selected]
        assert len(ids) == len(set(ids)) == 12
        plan = {'selected_ids': ids, 'selection': 'fixed SHA256 per group:8 single-image MC,2 multi-image MC,2 open; no answer/correctness selection',
                'not_a_full_accuracy_benchmark': True, 'generation': {'do_sample': False, 'max_new_tokens': 128},
                'reference': 'previous full900 vLLM protocol003/004; same exact input payload', 'input_files_sha256': sha(OLD / 'mmmu/samples.jsonl')}
        (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        env = BASE / 'kimi-hf4513-002'
        assert not env.exists()
        subprocess.run([sys.executable, '-m', 'venv', '--without-pip', '--system-site-packages', str(env)], check=True)
        wheels = Path('/root/kimi-reference011-wheels')
        command = [sys.executable, '-m', 'pip', '--python', str(env / 'bin/python'), 'install', '--no-index', '--find-links', str(wheels),
                   'transformers==4.51.3', 'tokenizers==0.21.4', 'huggingface-hub==0.30.2']
        with (out / 'install.log').open('xb') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        probe = "import torch,transformers,json;from transformers import AutoModelForCausalLM,AutoProcessor;print(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.cuda.is_available()}))"
        subprocess.run([str(env / 'bin/python'), '-c', probe], check=True, timeout=150,
                       stdout=(out / 'environment-probe.json').open('xb'), stderr=(out / 'environment-probe.stderr.log').open('xb'))
        record.update(complete=True, selected_ids=ids, python=str(env / 'bin/python'), all20_checkpoint_files_exact=True,
                      official_recommended_transformers='4.51.3', source_sha256=sha(Path(__file__)))
    except Exception as exc:
        record.update(complete=False, error=repr(exc))
        raise
    finally:
        record['ended_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
