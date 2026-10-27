"""Stage two immutable VLM experiments on the shared SCNet filesystem in CPU mode."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil
import stat
import zipfile

BASE=Path('/root/private_data/guoxin-session010')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def extract(source,target,expected):
    assert sha(source)==expected
    target.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(source) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))
        for i in z.infolist():
            rel=PurePosixPath(i.filename)
            assert not rel.is_absolute() and '..' not in rel.parts and not stat.S_ISLNK(i.external_attr>>16)
            if i.is_dir():continue
            p=target/rel;p.parent.mkdir(parents=True,exist_ok=True)
            with z.open(i) as a,p.open('xb') as b:shutil.copyfileobj(a,b)


def main():
    assert not Path('/dev/kfd').exists(),'Stage in no-card mode'
    BASE.mkdir(parents=True,exist_ok=True)
    assert not (BASE/'READY.json').exists()
    lock=json.loads(Path('/root/session010-transfer-lock.json').read_text())
    for label,filename in [('mmmu','session010-mmmu-validation-data001.zip'),('cvbench','session010-cvbench-data001.zip'),('tools','session010-vlm-tools001.zip')]:
        target=BASE/label
        if target.exists():
            assert (target/'READY.json').exists() if label!='tools' else (target/'evaluate_vlm.py').exists()
            continue
        extract(Path('/root')/filename,target,lock[filename]['sha256'])
    records=[]
    for short,slug,task in [('kimi','guoxin-kimi-vl-bw-vllm','kimi-vl-bw-vllm'),('robo','guoxin-robobrain-bw-vllm','robobrain-bw-vllm')]:
        source=Path('/opt/guoxin-reviewed008/projects')/slug
        destination=BASE/'projects'/short
        assert source.is_dir() and not destination.exists()
        shutil.copytree(source,destination,ignore=shutil.ignore_patterns('.git','__pycache__'))
        old=destination/'tasks'/task/'runtime'
        if old.is_dir():old.rename(old.with_name('runtime-inherited-history'))
        records.append({'project':short,'source':str(source),'destination':str(destination)})
    # A serving-capacity adjustment only; all five-image MMMU samples remain in the test.
    script=BASE/'projects/kimi/scripts/kimi_vl_runtime_005.py'
    before=sha(script);text=script.read_text()
    assert text.count("'{\"image\":2,\"video\":0}'")==1
    text=text.replace("'{\"image\":2,\"video\":0}'","'{\"image\":{\"count\":5,\"width\":1024,\"height\":1024},\"video\":0}'")
    # Keep one active sequence for the initial full accuracy run; HTTP requests may queue.
    script.write_text(text)
    model=Path('/root/private_data/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV')
    robo=json.loads((BASE/'projects/robo/tasks/robobrain-bw-vllm/task.json').read_text())['model']
    assert sha(model/'config.json')==robo['config_sha256']
    hashes={name:sha(model/name) for name in robo['preflight']['shard_sha256']}
    assert hashes==robo['preflight']['shard_sha256']
    record={'ready':True,'projects':records,'datasets':['mmmu','cvbench'],
            'kimi_capacity_change':{'before_sha256':before,'after_sha256':sha(script),'image_limit':5,'profiling_width':1024,'profiling_height':1024,'weights_or_processor_modified':False},
            'robo_weight_sha256':hashes,'gpu_loaded':False}
    (BASE/'READY.json').write_text(json.dumps(record,indent=2)+'\n')
    Path('/root/session010-stage-result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
