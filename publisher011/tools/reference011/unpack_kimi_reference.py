"""Verify and unpack the fixed reference preparation bundle on no-card SCNet."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import subprocess
import sys
import zipfile

path = Path('/root/session011-kimi-reference-prep001.zip')
assert not Path('/dev/kfd').exists()
assert hashlib.sha256(path.read_bytes()).hexdigest() == '5e3a42aa18f9bd685cf5fd91eddf538a297f7465bada0ae1e4664927e5daf5a7'
with zipfile.ZipFile(path) as archive:
    assert archive.testzip() is None
    rows = json.loads(archive.read('MANIFEST.json'))['files']
    assert len(rows) == 20 and len(archive.namelist()) == 21
    for row in rows:
        relative = PurePosixPath(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        data = archive.read(row['path'])
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
        destination = Path('/root').joinpath(*relative.parts)
        assert not destination.exists()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(data)
with Path('/root/session011-kimi-preparation001.log').open('xb') as log:
    child = subprocess.Popen([sys.executable, '-u', '/root/prepare_kimi_reference.py'],
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps({'preparation_pid': child.pid, 'GPU': False}), flush=True)
