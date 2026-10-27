"""Restore an image-baked pinned checkpoint to the validated canonical path."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil


def main():
    destination = Path(os.environ.get('KIMI_MODEL_PATH', '/root/private_data/guoxin-candidates/models/kimi-vl'))
    if destination.exists():
        print('Existing checkpoint retained; the native launcher performs full hash validation.')
        return
    source = Path('/opt/guoxin-kimi021/model')
    if not source.is_dir():
        raise FileNotFoundError('Pinned model missing; prepare it with scripts/download_kimi_vl_005.py')
    lock = json.loads((Path(__file__).resolve().parents[1]/'tasks/kimi-vl-bw-vllm/model-lock.json').read_text())
    assert lock['revision'] == '398eede0903cd983a2bfa0cc634e9ac1d843f375'
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(destination.parent).free > sum(row['size'] for row in lock['files']) + 1024**3
    destination.mkdir(exist_ok=False)
    for row in lock['files']:
        name = PurePosixPath(row['path']); assert not name.is_absolute() and '..' not in name.parts
        path = source/row['path']; assert path.is_file() and not path.is_symlink()
        target = destination/row['path']; target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(); count = 0
        with path.open('rb') as reader, target.open('xb') as writer:
            for block in iter(lambda: reader.read(16*1024**2), b''):
                writer.write(block); digest.update(block); count += len(block)
        assert count == row['size'] and digest.hexdigest() == row['sha256']
    print('Image checkpoint restored byte-for-byte; no symlink or weight conversion used.')


if __name__ == '__main__':
    main()
