"""Reuse persistent pinned weights or fetch exact official files before GPU launch."""
from pathlib import Path, PurePosixPath
import argparse
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    lock = json.loads((project/'tasks/kimi-vl-bw-vllm/model-lock.json').read_text())
    assert lock['revision'] == '398eede0903cd983a2bfa0cc634e9ac1d843f375'
    destination = Path(os.environ.get('KIMI_MODEL_PATH', '/root/private_data/guoxin-candidates/models/kimi-vl'))
    base = Path('/root/private_data/guoxin-candidates/models')
    assert destination.is_absolute() and destination.is_relative_to(base) and destination != base
    assert '..' not in destination.parts
    namespace = destination.relative_to(base).parts[0]
    assert namespace == 'kimi-vl' or namespace.startswith('kimi-vl-')
    for part in (destination, *destination.parents): assert not part.is_symlink()
    reusable = destination.is_dir()
    for row in lock['files']:
        name = PurePosixPath(row['path']); assert not name.is_absolute() and '..' not in name.parts
        path = destination/row['path']
        reusable = reusable and path.is_file() and not path.is_symlink() and path.stat().st_size == row['size']
    command = [sys.executable, str(project/'scripts/download_kimi_vl_005.py'), 'download',
               '--provider', 'modelscope', '--model-dir', str(destination), '--max-seconds', '6900']
    plan = {'strategy':'reuse_persistent_for_native_full_hash_check' if reusable else 'download_exact_official_files',
            'model_revision':lock['revision'], 'file_count':len(lock['files']), 'destination':str(destination),
            'command_if_missing_or_partial':command, 'GPU_requests':0, 'model_weights_in_image':False,
            'plan_only':args.plan_only}
    print(json.dumps(plan),flush=True)
    if args.plan_only: return
    if reusable:
        print('Existing files retained; the native launcher performs full SHA256 validation.')
        return
    assert sys.platform == 'linux', 'Downloads are confined to the SCNet Linux persistent model directory'
    # The existing transfer helper pins every official URL, size and SHA, retains
    # verified/partial files, and refuses overwriting a different final file.
    subprocess.run(command,check=True,timeout=7000)


if __name__ == '__main__':
    main()
