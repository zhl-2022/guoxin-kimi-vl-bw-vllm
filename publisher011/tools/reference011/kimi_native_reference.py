"""Bounded fixed-input prefix diagnostic; not a full accuracy benchmark."""
from pathlib import Path
import base64
import datetime as dt
import hashlib
import io
import json
import os
import signal
import socket
import sys
import time
import zipfile

BASE = Path('/root/private_data/guoxin-session011')
OLD = Path('/root/private_data/guoxin-session010')
MODEL = '/root/private_data/guoxin-candidates/models/kimi-vl'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Diagnostic hard time limit')))
    signal.alarm(2700)
    prep = BASE / 'kimi-preparation001'
    assert json.loads((prep / 'result.json').read_text())['complete']
    plan = json.loads((prep / 'plan.json').read_text())
    out = BASE / 'kimi-reference001'
    out.mkdir(exist_ok=False)
    record = {'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'complete': False,
              'not_a_full_accuracy_benchmark': True, 'plan': plan, 'source_sha256': sha(Path(__file__))}
    try:
        for port in (30129, 30001, 30000):
            with socket.socket() as sock:
                assert sock.connect_ex(('127.0.0.1', port)) != 0, 'Prior service active'
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor
        assert torch.cuda.is_available() and transformers.__version__ == '4.51.3'
        sys.path.insert(0, str(OLD / 'tools'))
        from evaluate_vlm003 import payload
        samples = {s['id']: s for s in (json.loads(line) for line in (OLD / 'mmmu/samples.jsonl').read_text().splitlines())}
        previous = {r['id']: r for r in (json.loads(line) for line in (OLD / 'kimi004/evaluation/predictions.jsonl').read_text().splitlines())}
        assert len(previous) == 900 and sha(OLD / 'mmmu/samples.jsonl') == plan['input_files_sha256']
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map='cuda:0',
                                                     trust_remote_code=True, attn_implementation='eager', local_files_only=True)
        model.eval()
        processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
        record.update(torch=torch.__version__, transformers=transformers.__version__, HIP=torch.version.hip,
                      device=torch.cuda.get_device_name(0), attention='unchanged publisher eager', precision='bfloat16')
        results = []
        with (out / 'predictions.jsonl').open('x') as stream:
            for key in plan['selected_ids']:
                sample = samples[key]
                body = payload(sample, OLD / 'mmmu', 'mmmu', 'kimi-vl-a3b-instruct', 4096)
                body_sha = hashlib.sha256(json.dumps(body, ensure_ascii=False).encode()).hexdigest()
                assert body_sha == previous[key]['request_sha256'], 'Original HTTP request differs'
                messages = []
                images = []
                for message in body['messages']:
                    content = []
                    for item in message['content']:
                        if item['type'] == 'image_url':
                            raw = base64.b64decode(item['image_url']['url'].split(',', 1)[1])
                            image = Image.open(io.BytesIO(raw)).convert('RGB')
                            images.append(image)
                            content.append({'type': 'image', 'image': '<in-memory-pinned-image>'})
                        else:
                            content.append(item)
                    messages.append({'role': message['role'], 'content': content})
                text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = processor(images=images, text=text, return_tensors='pt', padding=True, truncation=False).to(model.device)
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                ids = generated[0, inputs.input_ids.shape[1]:].cpu().tolist()
                answer = processor.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                old_text = previous[key]['text']
                common = 0
                for a, b in zip(answer, old_text):
                    if a != b:
                        break
                    common += 1
                row = {'id': key, 'request_sha256': body_sha, 'native_text': answer, 'previous_vllm_text': old_text,
                       'native_generated_ids': ids, 'native_is_prefix_of_vllm': old_text.startswith(answer),
                       'common_character_prefix': common, 'native_prompt_tokens': inputs.input_ids.shape[1],
                       'vllm_prompt_tokens': previous[key]['raw_response']['usage']['prompt_tokens'],
                       'input_ids_sha256': hashlib.sha256(inputs.input_ids.cpu().numpy().tobytes()).hexdigest(),
                       'image_grid_hws': inputs.image_grid_hws.cpu().tolist(), 'wall_seconds': time.perf_counter() - started}
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                results.append(row)
                print(json.dumps({'done': len(results), 'prefix_match': row['native_is_prefix_of_vllm'],
                                  'prompt_tokens_match': row['native_prompt_tokens'] == row['vllm_prompt_tokens']}), flush=True)
        record.update(complete=True, count=len(results), prefix_matches=sum(r['native_is_prefix_of_vllm'] for r in results),
                      prompt_token_count_matches=sum(r['native_prompt_tokens'] == r['vllm_prompt_tokens'] for r in results))
    except Exception as exc:
        record['error'] = repr(exc)
        raise
    finally:
        signal.alarm(0)
        record['ended_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        archive = Path('/root/session011-kimi-reference001.zip')
        if not archive.exists():
            with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as target:
                for directory in (prep, out):
                    for path in sorted(directory.rglob('*')):
                        if path.is_file():
                            target.write(path, path.relative_to(BASE).as_posix())
                target.write(Path(__file__), 'executed-kimi_native_reference.py')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
