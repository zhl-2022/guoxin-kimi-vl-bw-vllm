"""Combine disjoint, strictly audited native rows; never rewrite producer runs."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json


ALL = {(n, c, r) for n in (4096, 16384) for c in (1, 8, 64) for r in (1, 2)}
REMAINING = {(16384, 64, r) for r in (1, 2)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def select_rows(old, new):
    """Refuse missing, duplicated, failed or unexpectedly resampled rows."""
    selected = []
    for report, expected in ((old, ALL - REMAINING), (new, REMAINING)):
        assert report['source_provenance']['verified_for_reanalysis']
        rows = [row for row in report['rows'] if row['eligible_client_report']]
        keys = [(row['text_target_tokens'], row['concurrency'], row['round']) for row in rows]
        assert len(keys) == len(set(keys)) and set(keys) == expected, (keys, expected)
        for row in rows:
            assert row['process_status'] == 'exited' and row['exit_code'] == 0 and row['child_reaped']
            assert row['expected_requests'] == max(8, 2 * row['concurrency'])
            selected.append(row)
    assert len(selected) == 12 and sum(row['expected_requests'] for row in selected) == 608
    return selected


def runtime_config(attempt):
    paths = list((attempt / 'performance/runtime/before').glob('config-*.json'))
    assert len(paths) == 1
    config = json.loads(paths[0].read_text())
    normalized = {key: value for key, value in config.items() if key not in ('task', 'run_id')}
    return paths[0], normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-analysis', type=Path, required=True)
    parser.add_argument('--new-analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in (args.old_analysis, args.new_analysis)]
    selected = select_rows(*reports)
    sources = []
    profiles = []
    immutable = ['scripts/kimi_vl_runtime_005.py', 'tasks/kimi-vl-bw-vllm/model-lock.json',
                 'tasks/kimi-vl-bw-vllm/upstream-lock.json']
    manifests = []
    for path, report in zip((args.old_analysis, args.new_analysis), reports):
        run = Path(report['source_run_path'])
        assert sha(run) == report['source_run_sha256']
        attempt = run.parent.parent
        config_path, config = runtime_config(attempt)
        profiles.append(config)
        hashes = {name: sha(attempt / 'project' / name) for name in immutable}
        hashes['math_hook'] = sha(attempt / 'math-sdpa-hook/sitecustomize.py')
        hashes['compat_hook'] = sha(attempt / 'client-processor-compat/sitecustomize.py')
        hashes['wrapper'] = sha(attempt / 't02_sglang_bench-executed.py')
        manifests.append(hashes)
        accepted = [row for row in report['rows'] if row['eligible_client_report']]
        for row in accepted:
            assert sha(run.parent / row['native_file']) == row['native_file_sha256']
        sources.append({'analysis': str(path), 'analysis_sha256': sha(path),
                        'run': str(run), 'run_sha256': sha(run),
                        'runtime_config': str(config_path), 'runtime_config_sha256': sha(config_path),
                        'eligible_rows': [row['name'] for row in accepted],
                        'rejected_rows_preserved': [row['name'] for row in report['rows'] if not row['eligible_client_report']]})
    assert profiles[0] == profiles[1], 'Actual service configurations differ beyond task path and run UUID'
    assert manifests[0] == manifests[1], 'Model lock, executable driver or token policy source differs'
    result = {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'complete_native_matrix': True, 'native_reports': 12, 'measured_requests': 608,
              'combination_policy': 'Ten exited0/reaped old reports plus only the two previously unfinished cases; no complete row resampled',
              'source_reports': sources, 'rows': selected, 'identical_profile': profiles[0],
              'identical_source_hashes': manifests[0], 'source_files_modified': False,
              'client_wall_budget_change_only': {'old_case_seconds': 5400, 'new_case_seconds': 9000,
                                                  'new_total_seconds': 19000, 'service_seconds': 21600},
              'output_target_tokens': 1024, 'official_acceptance': False, 'submitted': False,
              'exact_output_tokens_independently_verified': False}
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('complete_native_matrix', 'native_reports', 'measured_requests')}))


if __name__ == '__main__':
    main()
