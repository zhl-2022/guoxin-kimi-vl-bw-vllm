"""Save the owned probe runtime and installed attention implementation for offline diagnosis."""
from pathlib import Path
import hashlib
import importlib.metadata as md
import json
import zipfile

output = Path('/root/session011-kimi-runtime-context001.zip')
entries = {}
runtime = Path('/root/private_data/guoxin-session010/projects/kimi/tasks/kimi-vl-bw-vllm/runtime')
for path in runtime.rglob('*'):
    if path.is_file() and path.stat().st_size < 12 * 1024**2:
        entries['owned-runtime/' + path.relative_to(runtime).as_posix()] = path
for distribution in ('vllm', 'vllm-hcu'):
    dist = md.distribution(distribution)
    for item in dist.files or ():
        name = item.as_posix()
        wanted = name.endswith('.py') and any(part in name for part in
            ('envs.py', 'kimi_vl', 'moonvit', 'deepseek_v2', 'platform', 'attention/selector',
             'attention/backends', 'engine/arg_utils', 'config/attention'))
        path = Path(dist.locate_file(item))
        if wanted and path.is_file() and path.stat().st_size < 2 * 1024**2:
            entries[distribution + '/' + name] = path
assert entries and any(name.startswith('owned-runtime/') for name in entries)
rows = []
with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
    for name, path in entries.items():
        data = path.read_bytes()
        archive.writestr(name, data)
        rows.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    archive.writestr('MANIFEST.json', json.dumps({'files': rows, 'scope': 'selected installed source and owned runtime only; not full environment export'}, indent=2))
    archive.write(Path(__file__), 'executed-export_kimi_context.py')
print(json.dumps({'files': len(rows), 'bytes': output.stat().st_size, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}), flush=True)
