"""Replay all 1000 OCRBench rows with the pinned public scorer and raw replies."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse
import ast
import base64
import datetime as dt
import hashlib
import io
import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
SCORER_SHA='27e6bc94e8a0b51d2ee401694fc7b80bcd8a36c51d324fbf4a37cce4b7c9764d'
SAMPLES_SHA='303fae6ff39f3716846cfafd34c6b62f9cb3bd66630f86034e71c6c7ebb42e6f'
ANNOTATIONS_SHA='b2dc45f48eb5f3b7807d7063d15aad57f8d589f79ecb8f1f22dafd66fa1c2412'
MODEL_REVISION='398eede0903cd983a2bfa0cc634e9ac1d843f375'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_inputs(data):
    data=data.resolve()
    assert sha(data/'samples.jsonl')==SAMPLES_SHA
    assert sha(data/'annotations.json')==ANNOTATIONS_SHA
    source=ROOT/'publisher016/vlmeval-image-vqa.py'
    assert sha(source)==SCORER_SHA
    baseline=json.loads((ROOT/'baselines/ocrbench-publisher.json').read_text())
    assert baseline['publisher_score']==867 and baseline['minimum']==847
    assert baseline['revision']==MODEL_REVISION
    samples=[json.loads(x) for x in (data/'samples.jsonl').read_text().splitlines()]
    annotations=json.loads((data/'annotations.json').read_text())
    assert len(samples)==len(annotations)==len({s['id'] for s in samples})==1000
    assert {s['id'] for s in samples}=={a['id'] for a in annotations}
    for sample in samples:
        path=(data/sample['image']).resolve()
        assert path.is_relative_to(data) and not path.is_symlink()
        assert sha(path)==sample['image_sha256'],sample['id']
    return samples,annotations,source


def payload(sample,data,model):
    from PIL import Image
    path=data/sample['image']
    image=Image.open(path)
    if image.mode=='RGBA':
        image=Image.alpha_composite(Image.new('RGBA',image.size,(255,255,255,255)),image).convert('RGB')
    buffer=io.BytesIO();image.convert('RGB').save(buffer,format=sample['image_format'])
    mime=mimetypes.guess_type(path.name)[0] or 'image/jpeg'
    content=[{'type':'image_url','image_url':{'url':'data:'+mime+';base64,'+base64.b64encode(buffer.getvalue()).decode()}},
             {'type':'text','text':sample['question']}]
    return {'temperature':0,'top_p':1,'max_tokens':4096,'seed':42,'model':model,
            'messages':[{'role':'user','content':content}],'stream':False}


def score_predictions(source,annotations,predictions,output):
    mapping={r['id']:r for r in predictions}
    assert len(mapping)==1000 and all(r['status']=='ok' for r in predictions)
    assert set(mapping)=={a['id'] for a in annotations}
    frame=[dict(a,prediction=mapping[a['id']]['text'],answer=repr(a['answer'])) for a in annotations]
    class Frame:
        iloc=frame
        def __len__(self):return len(frame)
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OCRBench')
    namespace={'ImageBaseDataset':object,'load':lambda _:Frame(),'tqdm':lambda x:x,
               'eval':ast.literal_eval,'get_intermediate_file_path':lambda *x:output/'score.json',
               'dump':lambda value,p:Path(p).write_text(json.dumps(value,indent=2))}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'<pinned OCRBench class>','exec'),namespace)
    return namespace['OCRBench'].evaluate(output/'predictions.jsonl')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selfcheck',action='store_true')
    parser.add_argument('--data-dir',type=Path,default=Path(os.environ.get('KIMI_DATA_DIR',str(ROOT/'data/ocrbench'))))
    parser.add_argument('--base-url',default='http://127.0.0.1:'+os.environ.get('KIMI_PORT','30129'))
    parser.add_argument('--model',default=os.environ.get('KIMI_SERVED_NAME','kimi-vl-a3b-instruct'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--rescore',type=Path,help='Read archived complete predictions; performs no HTTP requests')
    args=parser.parse_args();data=args.data_dir.resolve()
    samples,annotations,source=checked_inputs(data)
    if args.selfcheck:
        print(json.dumps({'selfcheck':True,'rows':1000,'all_images_hashed':True,
              'scorer_sha256':SCORER_SHA,'GPU_requests':0,'accuracy_inferred':False}));return 0
    url=urllib.parse.urlparse(args.base_url)
    assert url.scheme=='http' and url.hostname in ('127.0.0.1','localhost') and not url.username and not url.password
    output=args.output or ROOT/('verify-ocrbench-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True,exist_ok=False)
    record={'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'rows':1000,'model_revision':MODEL_REVISION,
            'scorer_sha256':SCORER_SHA,'samples_sha256':SAMPLES_SHA,'annotations_sha256':ANNOTATIONS_SHA,
            'script_sha256':sha(__file__),'publisher_baseline':867,'threshold':847,'complete':False,
            'rescore_only':bool(args.rescore),'numeric_pass':False,'submitted':False,'labels_sent_to_model':False}
    (output/'run-start.json').write_text(json.dumps(record,indent=2))
    def request(sample):
        tick=time.perf_counter();row={'id':sample['id'],'image_sha256':sample['image_sha256']}
        try:
            raw=json.dumps(payload(sample,data,args.model),ensure_ascii=False).encode()
            row['request_sha256']=hashlib.sha256(raw).hexdigest()
            req=urllib.request.Request(args.base_url.rstrip('/')+'/v1/chat/completions',raw,{'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=600) as response:value=json.loads(response.read(8*1024**2))
            choice=value['choices'][0];text=choice['message']['content']
            assert choice['finish_reason'] in ('stop','length') and isinstance(text,str)
            row.update(status='ok',text=text,finish_reason=choice['finish_reason'],raw_response=value)
        except urllib.error.HTTPError as exc:
            row.update(status='failed',error=str(exc),error_body=exc.read(16384).decode(errors='replace'))
        except Exception as exc:row.update(status='failed',error=repr(exc))
        row['wall_seconds']=time.perf_counter()-tick;return row
    predictions=[]
    try:
        if args.rescore:
            raw=args.rescore.read_bytes();(output/'predictions.jsonl').write_bytes(raw)
            predictions=[json.loads(x) for x in raw.decode().splitlines()]
        else:
            with (output/'predictions.jsonl').open('x') as log,ThreadPoolExecutor(max_workers=4) as pool:
                for future in as_completed([pool.submit(request,s) for s in samples]):
                    row=future.result();predictions.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
                    if len(predictions)%25==0:print(json.dumps({'done':len(predictions),'failed':sum(r['status']!='ok' for r in predictions)}),flush=True)
        record['generation_complete']=len(predictions)==1000 and all(r['status']=='ok' for r in predictions)
        if record['generation_complete']:
            score=score_predictions(source,annotations,predictions,output)
            record.update(complete=True,score=score,numeric_pass=score['Final Score']>=847,
                          budget_exhausted=sum(r['finish_reason']=='length' for r in predictions))
    except BaseException as exc:record['error']=repr(exc)
    finally:
        record['ended_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        (output/'run-final.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
        (output/'verify_result.json').write_text(json.dumps(record,indent=2)+'\n')
    return 0 if record['complete'] and record['numeric_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
