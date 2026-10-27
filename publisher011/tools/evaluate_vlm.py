"""Full pinned MMMU/CV-Bench HTTP evaluation with original upstream scoring."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import base64
import datetime as dt
import hashlib
import importlib.util
import json
import math
import mimetypes
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]


def prompt(sample, mode):
    question=sample['question']
    if mode=='cvbench':
        question+=''.join('\n'+chr(65+i)+'. '+x for i,x in enumerate(sample['options']))
        return question+"\nAnswer with the option's letter from the given choices directly.<think></think><answer>"
    if sample['question_type']=='multiple-choice':
        choices=''.join(f'({chr(65+i)}) {x}\n' for i,x in enumerate(sample['options']))
        return question+'\n\n'+choices+"\n\nAnswer with the option's letter from the given choices directly."
    return question+'\n\nAnswer the question using a single word or phrase.'


def payload(sample,data,mode,model,max_tokens):
    content=[]
    for image in sample['images']:
        path=(data/image['path']).resolve()
        if not path.is_relative_to(data.resolve()) or path.is_symlink():raise ValueError('Image outside dataset')
        if sha(path)!=image['sha256']:raise ValueError('Image identity changed')
        mime=mimetypes.guess_type(path.name)[0] or 'image/png'
        if mime=='image/x-ms-bmp':mime='image/bmp'
        content.append({'type':'image_url','image_url':{'url':'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()}})
    content.append({'type':'text','text':prompt(sample,mode)})
    return {'model':model,'messages':[{'role':'user','content':content}],
            'temperature':0,'top_p':1,'max_tokens':max_tokens,'stream':False,'seed':42}


def score(samples,results,mode):
    if len(samples)!=len(results) or {s['id'] for s in samples}!={r['id'] for r in results}:
        raise ValueError('Result IDs differ')
    mapped={r['id']:r for r in results}
    if len(mapped)!=len(results):raise ValueError('Duplicate result identity')
    failed=[r['id'] for r in results if r['status']!='ok']
    if failed:return [],{'complete':False,'failed_ids':failed,'numeric_comparison_pass':False}
    scored=[]
    if mode=='mmmu':
        import mmmu_eval
        random.seed(42);fallback=0
        for sample in samples:
            response=mapped[sample['id']]['text']
            if sample['question_type']=='multiple-choice':
                answers={chr(65+i):s for i,s in enumerate(sample['options'])}
                before=random.getstate()
                parsed=mmmu_eval.parse_multi_choice_response(response,list(answers),answers)
                fallback+=before!=random.getstate()
                okay=mmmu_eval.eval_multi_choice(sample['answer'],parsed)
            else:
                parsed=mmmu_eval.parse_open_response(response)
                okay=mmmu_eval.eval_open(sample['answer'],parsed)
            scored.append({'id':sample['id'],'parsed':parsed,'reference':sample['answer'],'correct':bool(okay)})
        value=100*sum(r['correct'] for r in scored)/len(scored)
        result={'accuracy_percent':value,'publisher_baseline':57.0,'threshold_percent':55.0,
                'upstream_random_fallback_count':fallback,'metric':'full-validation micro accuracy'}
    else:
        from flagevalmm.evaluator.common_types import evaluate_multiple_choice
        import cvbench_eval
        annotations={s['id']:{'answer':s['answer'][1], 'source':s['source']} for s in samples}
        predictions=[]
        for s in samples:
            text=mapped[s['id']]['text']
            if '</think>' in text:
                match=re.search(r'<answer>(.*?)</answer>',text,re.DOTALL)
                text=match.group(1).strip() if match else ''
            predictions.append({'question_id':s['id'],'answer':text})
        metrics=cvbench_eval.get_result(annotations,predictions)
        for pred in predictions:
            scored.append({'id':pred['question_id'],'parsed':pred['answer'],'raw_text':pred['raw_answer'],
                           'reference':pred['label'],'correct':bool(pred['correct'])})
        value=metrics['accuracy']
        result={'accuracy_percent':value,'publisher_baseline':94.58,'threshold_percent':92.58,
                'metric':'FlagEvalMM source-weighted CV-Bench; intermediate rounding preserved','detail':metrics}
    result.update(complete=True,sample_count=len(samples),failed_ids=[],numeric_comparison_pass=value>=result['threshold_percent'],
                  official_acceptance=False,submitted=False)
    return scored,result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['mmmu','cvbench'],required=True)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--endpoint',required=True)
    ap.add_argument('--model',required=True)
    ap.add_argument('--concurrency',type=int,choices=[1,2,4],default=4)
    ap.add_argument('--max-tokens',type=int,required=True)
    args=ap.parse_args()
    assert args.endpoint.startswith('http://127.0.0.1:')
    assert args.max_tokens==({'mmmu':512,'cvbench':4096}[args.mode])
    samples=rows(args.data/'samples.jsonl');identity=read(args.data/'READY.json')
    assert sha(args.data/'samples.jsonl')==identity['samples_sha256']
    assert len(samples)==len({s['id'] for s in samples})==({'mmmu':900,'cvbench':2638}[args.mode])
    args.output.mkdir(parents=True,exist_ok=False)
    protocol={'UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'mode':args.mode,'identity':identity,
              'model':args.model,'endpoint':args.endpoint,'max_tokens':args.max_tokens,'temperature':0,'top_p':1,
              'reference_answers_sent_to_model':False,'images':'all original images in source key order; no subset removal',
              'source_sha256':sha(Path(__file__)),'vendor_sources':{p.relative_to(ROOT/'vendor').as_posix():sha(p)
              for p in (ROOT/'vendor').rglob('*') if p.is_file() and '__pycache__' not in p.parts}}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    def request(sample):
        result={'id':sample['id'],'image_sha256':[x['sha256'] for x in sample['images']]}
        start=time.perf_counter()
        try:
            body=payload(sample,args.data,args.mode,args.model,args.max_tokens)
            raw=json.dumps(body,ensure_ascii=False).encode()
            result['request_sha256']=hashlib.sha256(raw).hexdigest()
            with urllib.request.urlopen(urllib.request.Request(args.endpoint+'/v1/chat/completions',raw,{'Content-Type':'application/json'}),timeout=300) as response:
                value=json.loads(response.read(8*1024*1024))
            choice=value['choices'][0]
            assert choice['finish_reason']=='stop', 'Response did not finish naturally'
            assert isinstance(choice['message']['content'],str) and choice['message']['content'].strip()
            result.update(status='ok',text=choice['message']['content'],raw_response=value)
        except urllib.error.HTTPError as exc:result.update(status='failed',error=str(exc),error_body=exc.read(16384).decode(errors='replace'))
        except Exception as exc:result.update(status='failed',error=str(exc))
        result['wall_seconds']=time.perf_counter()-start
        return result
    results=[];start=time.perf_counter()
    with (args.output/'predictions.jsonl').open('x',encoding='utf-8') as output,ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures=[pool.submit(request,s) for s in samples]
        for future in as_completed(futures):
            row=future.result();results.append(row)
            output.write(json.dumps(row,ensure_ascii=False)+'\n');output.flush()
            if len(results)%25==0:print(json.dumps({'done':len(results),'failed':sum(r['status']!='ok' for r in results)}),flush=True)
    scored,result=score(samples,results,args.mode)
    result['wall_seconds']=time.perf_counter()-start
    (args.output/'verify_result.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'scored.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in scored),'utf-8')
    print(json.dumps(result),flush=True)
    if not result['complete']:raise SystemExit(2)
    selected=sorted(samples,key=lambda s:hashlib.sha256(('vlm010-bench-v1:'+s['id']).encode()).digest())[:24]
    bench=args.output/'bench';bench.mkdir()
    (bench/'protocol.json').write_text(json.dumps({'selected':[s['id'] for s in selected],'warmup':'full accuracy run before measurement'},indent=2))
    rounds=[]
    for concurrency in (1,4):
        for iteration in (1,2):
            tick=time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:calls=list(pool.map(request,selected))
            wall=time.perf_counter()-tick
            (bench/f'c{concurrency}-round{iteration}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in calls),'utf-8')
            successful=[r for r in calls if r['status']=='ok'];latency=sorted(r['wall_seconds'] for r in successful)
            summary={'concurrency':concurrency,'round':iteration,'requests':len(calls),'success':len(successful),
                     'wall_seconds':wall,'request_per_second':len(successful)/wall,
                     'latency_p50_seconds':statistics.median(latency) if latency else None,
                     'latency_p95_seconds':latency[math.ceil(len(latency)*.95)-1] if latency else None}
            rounds.append(summary)
    (bench/'summary.json').write_text(json.dumps({'rounds':rounds,'all_successful':all(r['success']==24 for r in rounds)},indent=2))
    print(json.dumps({'benchmark_complete':True,'all_successful':all(r['success']==24 for r in rounds)}),flush=True)
    raise SystemExit(0 if result['numeric_comparison_pass'] else 2)


if __name__=='__main__':main()
