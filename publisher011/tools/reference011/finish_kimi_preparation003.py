"""Recognize only the observed no-HIP preflight constraint; GPU probe still required."""
from pathlib import Path
import datetime as dt
import hashlib
import importlib.metadata
import json

BASE = Path('/root/private_data/guoxin-session011')
assert not Path('/dev/kfd').exists()
out = BASE / 'kimi-preparation002'
result = json.loads((out / 'result.json').read_text())
lock = json.loads(Path('/root/private_data/guoxin-session010/projects/kimi/tasks/kimi-vl-bw-vllm/model-lock.json').read_text())
actual = {r['path']: (r['bytes'], r['sha256']) for r in result['files']}
expected = {r['path']: (r['size'], r['sha256']) for r in lock['files']}
assert actual == expected and len(actual) == 20
error = (out / 'environment-probe.stderr.log').read_text()
assert 'No HIP GPUs are available' in error
assert (out / 'plan.json').is_file()
env = BASE / 'kimi-hf4513-002'
packages = {d.metadata['Name'].lower().replace('_', '-'): d.version
            for d in importlib.metadata.distributions(path=[str(env / 'lib/python3.10/site-packages')])}
assert packages['transformers'] == '4.51.3' and packages['tokenizers'] == '0.21.4'
assert packages['huggingface-hub'] == '0.30.2'
record = {'UTC': dt.datetime.now(dt.timezone.utc).isoformat(), 'no_card_preparation_complete': True,
          'all20_checkpoint_files_exact': True, 'isolated_pinned_packages_installed': True,
          'CPU_AutoProcessor_import_pass': False, 'GPU_import_probe_still_required': True,
          'observed_constraint': 'HCU torchvision initializes HIP during AutoProcessor import; no-card environment has no HIP GPU',
          'previous_raw_result_preserved': True, 'python': str(env / 'bin/python'),
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (out / 'READY003.json').open('x') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record), flush=True)
