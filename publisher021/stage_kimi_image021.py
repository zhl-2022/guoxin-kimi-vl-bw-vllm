"""Copy pinned model/data to new system-disk paths; no GPU or service actions."""
from pathlib import Path, PurePosixPath
import datetime as dt
import hashlib
import json
import os
import shutil


DEST = Path('/opt/guoxin-kimi021')
MODEL = Path('/root/private_data/guoxin-candidates/models/kimi-vl')
DATA = Path('/root/private_data/guoxin-session016/ocr-stage001/data')
PROJECT = Path('/root/private_data/guoxin-session021/kimi-portable-replay002/project')


def save(record):
    temp = DEST/'stage-progress.tmp'; temp.write_text(json.dumps(record, indent=2)+'\n'); temp.replace(DEST/'stage-progress.json')


def copy_checked(source, target, expected=None):
    assert source.is_file() and not source.is_symlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(); count = 0
    with source.open('rb') as reader, target.open('xb') as writer:
        for block in iter(lambda: reader.read(16*1024**2), b''):
            writer.write(block); digest.update(block); count += len(block)
    actual = digest.hexdigest()
    assert target.stat().st_size == source.stat().st_size == count
    if expected: assert actual == expected, str(source)
    return {'path': str(target.relative_to(DEST)), 'bytes': count, 'sha256': actual}


def main():
    lock = json.loads((PROJECT/'tasks/kimi-vl-bw-vllm/model-lock.json').read_text())
    assert lock['revision'] == '398eede0903cd983a2bfa0cc634e9ac1d843f375'
    need = sum(row['size'] for row in lock['files'])
    assert shutil.disk_usage('/opt').free > need + 10*1024**3
    DEST.mkdir(exist_ok=False)
    record = {'UTC': dt.datetime.now(dt.timezone.utc).isoformat(), 'pid': os.getpid(),
              'state': 'copying_pinned_model', 'files': [], 'GPU_requests': 0,
              'model_services_modified': False, 'existing_files_overwritten': False, 'complete': False}
    save(record)
    for row in lock['files']:
        name = PurePosixPath(row['path']); assert not name.is_absolute() and '..' not in name.parts
        record['copying'] = row['path']; save(record)
        record['files'].append(copy_checked(MODEL/row['path'], DEST/'model'/row['path'], row['sha256']))
        save(record)
    record['files'].append(copy_checked(PROJECT/'tasks/kimi-vl-bw-vllm/model-lock.json', DEST/'model-lock.json'))
    assert hashlib.sha256((DATA/'samples.jsonl').read_bytes()).hexdigest() == '303fae6ff39f3716846cfafd34c6b62f9cb3bd66630f86034e71c6c7ebb42e6f'
    record.update(state='copying_complete_OCRBench1000'); save(record)
    for folder, dirs, names in os.walk(DATA, onerror=lambda e: (_ for _ in ()).throw(e)):
        dirs[:] = sorted(d for d in dirs if d != '__pycache__')
        for name in sorted(names):
            source = Path(folder)/name
            record['files'].append(copy_checked(source, DEST/'data'/source.relative_to(DATA)))
    record.update(state='model_and_data_staged_code_pending', complete=True, ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    save(record)
    with (DEST/'IMAGE-STAGE-COMPLETE.json').open('x') as stream: stream.write(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'complete': True, 'files': len(record['files']), 'bytes': sum(x['bytes'] for x in record['files'])}), flush=True)


if __name__ == '__main__':
    main()
