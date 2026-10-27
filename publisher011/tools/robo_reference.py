"""Compare fixed-input native Transformers outputs with the completed vLLM run.

Samples are selected by a fixed hash, without consulting labels or correctness.
This diagnostic is not a substitute for a complete benchmark score.
"""
from pathlib import Path
import hashlib
import json
import time
import zipfile
import sys
import socket

BASE=Path('/root/private_data/guoxin-session010')
OUT=BASE/'robo-reference001'
MODEL='/root/private_data/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV'
sys.path.insert(0,str(BASE/'tools'))
from evaluate_vlm002 import prompt


def main():
    OUT.mkdir(exist_ok=False)
    record={'complete':False,'not_a_full_accuracy_benchmark':True}
    try:
        with socket.socket() as s:
            assert s.connect_ex(('127.0.0.1',30001))!=0
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForImageTextToText
        assert torch.cuda.is_available()
        model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='eager',device_map='cuda:0')
        model.eval()
        processor=AutoProcessor.from_pretrained(MODEL)
        samples=[json.loads(x) for x in (BASE/'cvbench/samples.jsonl').read_text().splitlines()]
        selected=sorted(samples,key=lambda s:hashlib.sha256(('robo-framework-parity010:'+s['id']).encode()).digest())[:48]
        vllm={r['id']:r for r in (json.loads(x) for x in (BASE/'robo002/evaluation/predictions.jsonl').read_text().splitlines())}
        record.update(selected_ids=[s['id'] for s in selected],torch=torch.__version__,device=torch.cuda.get_device_name(0),
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),attention='Transformers eager native reference',
                      max_new_tokens=768,do_sample=False,labels_used_for_selection=False)
        results=[]
        with (OUT/'predictions.jsonl').open('x') as target:
            for sample in selected:
                image=(BASE/'cvbench'/sample['images'][0]['path'])
                assert hashlib.sha256(image.read_bytes()).hexdigest()==sample['images'][0]['sha256']
                messages=[{'role':'user','content':[{'type':'image','image':str(image)},{'type':'text','text':prompt(sample,'cvbench')}]}]
                text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                inputs=processor(text=[text],images=[Image.open(image).convert('RGB')],padding=True,return_tensors='pt').to(model.device)
                started=time.perf_counter()
                with torch.inference_mode():
                    tokens=model.generate(**inputs,max_new_tokens=768,do_sample=False)
                generated=tokens[:,inputs.input_ids.shape[1]:]
                output=processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                previous=vllm[sample['id']]
                row={'id':sample['id'],'text':output,'vllm_text':previous['text'],'exact_text_match':output==previous['text'],
                     'native_prompt_tokens':inputs.input_ids.shape[1],
                     'vllm_prompt_tokens':previous['raw_response']['usage']['prompt_tokens'],
                     'input_token_sha256':hashlib.sha256(inputs.input_ids.cpu().numpy().tobytes()).hexdigest(),
                     'image_grid_thw':inputs.image_grid_thw.cpu().tolist(),'generated_ids':generated.cpu().tolist(),
                     'wall_seconds':time.perf_counter()-started}
                target.write(json.dumps(row)+'\n');target.flush();results.append(row)
                print(json.dumps({'done':len(results),'text_match':sum(r['exact_text_match'] for r in results)}),flush=True)
        record.update(complete=True,count=len(results),exact_text_matches=sum(r['exact_text_match'] for r in results),
                      prompt_token_count_matches=sum(r['native_prompt_tokens']==r['vllm_prompt_tokens'] for r in results))
    except Exception as e:
        record['error']=repr(e)
        raise
    finally:
        (OUT/'result.json').write_text(json.dumps(record,indent=2)+'\n')
        with zipfile.ZipFile('/root/session010-robo-reference001.zip','x',zipfile.ZIP_DEFLATED) as z:
            for path in OUT.rglob('*'):
                if path.is_file():z.write(path,path.relative_to(OUT).as_posix())


if __name__=='__main__':main()
