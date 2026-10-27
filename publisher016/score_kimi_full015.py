"""CPU audit and published boxed-final scoring for the complete math-SDPA run."""
from pathlib import Path
import ast
import copy
import datetime as dt
import hashlib
import json
import sys
import zipfile

BASE = Path('/root/private_data/guoxin-session010')
RUN = Path('/root/private_data/guoxin-session013/whole-attempt2')
STAGE = Path('/root/private_data/guoxin-session014/robo-spatial-stage015')
OUT = Path('/root/private_data/guoxin-session014/kimi-scoring-full015')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def main():
    final = json.loads((RUN / 'run-final.json').read_text())
    assert final['complete'] and final['count'] == 900 and final['owned_service_stopped']
    data = BASE / 'mmmu'
    assert sha(data / 'samples.jsonl') == '90f833664dc9d0b4ce6b8ce1d9035aac89d6a1c153aa462995c5977af66017ea'
    samples = rows(data / 'samples.jsonl')
    predictions = rows(RUN / 'predictions.jsonl')
    previous = {r['id']:r for r in rows(BASE / 'kimi004/evaluation/predictions.jsonl')}
    assert len(samples) == len(predictions) == len({r['id'] for r in predictions}) == 900
    mapped = {r['id']:r for r in predictions}
    assert set(mapped) == set(previous) == {s['id'] for s in samples}
    sys.path[:0] = [str(BASE / 'tools'), str(BASE / 'tools/vendor')]
    from evaluate_vlm003 import payload, score
    for sample in samples:
        result = mapped[sample['id']]
        assert result['status'] == 'ok'
        assert result['text'] == result['raw_response']['choices'][0]['message']['content']
        assert result['raw_response']['usage']['prompt_tokens'] == previous[sample['id']]['raw_response']['usage']['prompt_tokens']
        body = json.dumps(payload(sample,data,'mmmu','kimi-vl-a3b-instruct',4096),ensure_ascii=False).encode()
        assert hashlib.sha256(body).hexdigest() == result['request_sha256'] == previous[sample['id']]['request_sha256']
        assert result['image_sha256'] == [i['sha256'] for i in sample['images']]
    original_rows, original = score(samples,predictions,'mmmu')
    extractor_source = STAGE / 'sources/kimi-vl-extractor-source.py'
    tree = ast.parse(extractor_source.read_text())
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'extract_boxed_content')
    namespace = {}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<pinned VLMEvalKit extractor>','exec'),namespace)
    extract = namespace['extract_boxed_content']
    transformed = copy.deepcopy(predictions)
    changed = 0
    for row in transformed:
        value = extract(row['text'])
        changed += value != row['text']
        row['text'] = value
    boxed_rows, boxed = score(samples,transformed,'mmmu')
    OUT.mkdir(parents=True,exist_ok=False)
    for label,value in [('raw-scored.jsonl',original_rows),('boxed-scored.jsonl',boxed_rows)]:
        (OUT / label).write_text(''.join(json.dumps(r,ensure_ascii=False) + '\n' for r in value))
    record = {'UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'complete':True,'sample_count':900,
              'all_requests_images_prompt_counts_verified':True,
              'original_score':original,'boxed_final_score':boxed,'boxes_extracted':changed,
              'publisher_baseline':57,'absolute_two_point_lower':55,
              'relative_two_percent_lower':55.86,
              'primary_metric':'Published VLMEvalKit boxed-final parser followed by original MMMU scorer',
              'numeric_comparison_pass':boxed['accuracy_percent'] >= 55,
              'stricter_relative_comparison_pass':boxed['accuracy_percent'] >= 55.86,
              'budget_exhausted':sum(r['finish_reason']=='length' for r in predictions),
              'raw_predictions_sha256':sha(RUN / 'predictions.jsonl'),
              'raw_run_final_sha256':sha(RUN / 'run-final.json'),
              'raw_archive_sha256':sha(Path('/root/session013-kimi-whole-attempt2.zip')),
              'score_source_sha256':sha(Path(__file__)),
              'extractor_source_sha256':sha(extractor_source),
              'GPU_requests_made_by_scorer':0,'submitted':False,'official_acceptance':False}
    (OUT / 'verify_result.json').write_text(json.dumps(record,indent=2))
    with zipfile.ZipFile('/root/session014-kimi-scoring-full015.zip','x',zipfile.ZIP_DEFLATED) as bundle:
        for p in sorted(OUT.iterdir()):
            bundle.write(p,p.name)
    print(json.dumps(record),flush=True)


if __name__ == '__main__':
    main()
