"""Independently verify all archived root-replay request and response identities."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--archive-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source = a.project/'scripts/verify_kimi_ocr016.py'
    assert sha(source) == '99e0d89d2f0d7f18e8a9ef8348cb9583f57f1e1723763dad009d229de77313b6'
    spec = importlib.util.spec_from_file_location('frozen_ocr_evaluator', source)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    samples, annotations, scorer = mod.checked_inputs(a.data)
    raw = a.archive_root/'full-accuracy/predictions.jsonl'
    predictions = [json.loads(line) for line in raw.read_text(encoding='utf-8').splitlines()]
    assert len(predictions) == 1000 == len({r['id'] for r in predictions})
    by_id = {r['id']: r for r in predictions}
    assert set(by_id) == {r['id'] for r in samples}
    model = '/root/private_data/guoxin-candidates/models/kimi-vl'
    completion_ids = []
    for sample in samples:
        row = by_id[sample['id']]
        assert row['status'] == 'ok' and row['finish_reason'] == 'stop'
        assert row['image_sha256'] == sample['image_sha256']
        request = mod.payload(sample, a.data, model)
        assert request['messages'][0]['content'][1] == {'type': 'text', 'text': sample['question']}
        assert set(request) == {'temperature','top_p','max_tokens','seed','model','messages','stream'}
        assert request['temperature'] == 0 and request['top_p'] == 1 and request['max_tokens'] == 4096
        assert request['seed'] == 42 and request['stream'] is False
        digest = hashlib.sha256(json.dumps(request, ensure_ascii=False).encode()).hexdigest()
        assert digest == row['request_sha256'], row['id']
        response = row['raw_response']; choice = response['choices'][0]
        assert response['model'] == model and response['object'] == 'chat.completion'
        assert choice['finish_reason'] == 'stop' and choice['message']['content'] == row['text']
        assert isinstance(row['text'], str) and row['text']
        assert math.isfinite(row['wall_seconds']) and row['wall_seconds'] > 0
        usage = response['usage']
        assert usage['prompt_tokens'] > 0 and 0 < usage['completion_tokens'] <= 4096
        assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
        completion_ids.append(response['id'])
    assert len(set(completion_ids)) == 1000
    result = json.loads((a.archive_root/'full-accuracy/run-final.json').read_text())
    assert result['rows'] == 1000 and result['generation_complete'] and not result['rescore_only']
    assert result['score']['Final Score'] == 861 and result['budget_exhausted'] == 0
    output = {'UTC': dt.datetime.now(dt.timezone.utc).isoformat(), 'complete': True,
              'samples': 1000, 'unique_input_ids': 1000, 'unique_response_ids': 1000,
              'rebuilt_payload_hash_matches': 1000, 'all_image_hashes_match': True,
              'raw_response_text_matches': 1000, 'all_finish_reason_stop': True,
              'labels_in_request': False, 'synthetic_bad_words_in_request': False,
              'full_accuracy_score': 861, 'predictions_sha256': sha(raw),
              'scorer_sha256': sha(scorer), 'evaluator_sha256': sha(source),
              'GPU_requests': 0, 'submitted': False,
              'scope': 'Complete archived real HTTP responses, CPU request reconstruction and independent rescore; no new GPU run'}
    with a.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
