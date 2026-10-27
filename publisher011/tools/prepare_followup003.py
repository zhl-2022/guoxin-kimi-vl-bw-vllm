"""Prepare isolated Kimi profile and recover the already verified SeedTTS download."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

BASE=Path('/root/private_data/guoxin-session010')
HERE=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    record={}
    for name in ('evaluate_vlm003.py','run_vlm_experiment003.py'):
        destination=BASE/'tools'/name
        assert not destination.exists()
        shutil.copy2(HERE/name,destination)
        record[name]=digest(destination)
    old=BASE/'projects/kimi/scripts/kimi_vl_runtime_005.py'
    new=old.with_name('kimi_vl_runtime_010003.py')
    text=old.read_text();before=digest(old)
    marker='"--max-num-seqs", "1"'
    assert text.count(marker)==1 and not new.exists()
    new.write_text(text.replace(marker,'"--max-num-seqs", "4"'))
    record['runtime_profile']={'before_sha256':before,'after_sha256':digest(new),'max_num_seqs':4}
    seed=BASE/'seedtts';partial=seed/'en.parquet.partial';complete=seed/'en.parquet'
    if not complete.exists() and partial.exists():
        expected='5849b41b49cae996328c06d2c5791717c3bafc369bddfa1ec4f86761bb8bc0ca'
        record['seedtts_partial_sha256']=digest(partial)
        if partial.stat().st_size==247555423 and record['seedtts_partial_sha256']==expected:
            partial.rename(complete)
            record['seedtts_recovered_without_download']=True
    (BASE/'FOLLOWUP003-READY.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    if complete.exists() and not (BASE/'AUK-READY.json').exists():
        with Path('/root/session010-auk-stage003.log').open('xb') as f:
            process=subprocess.Popen([sys.executable,'-u',str(BASE/'auk-tools/stage_auk002.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'auk_cpu_preparation_pid':process.pid}),flush=True)


if __name__=='__main__':main()
