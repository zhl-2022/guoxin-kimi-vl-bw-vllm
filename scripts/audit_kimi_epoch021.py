"""Audit the fixed token-policy delta without inventing any accuracy score."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import sys


WRAPPER = 'eab500bbc9bd30208a6020d3dae8cef2df5a247e1391066c5303025fd6cef83a'
BODY = {'temperature': 0, 'ignore_eos': True, 'stream_options': {'include_usage': True},
        'bad_words': ['<|media_pad|>', '<|placeholder|>', '<|media_start|>', '<|media_end|>']}
SOURCE = {'driver.py': '11985f3c844f03b320f674309ea19060e1bc2f1ac93619c031092d574af22dcc',
          'client-hook.py': '8dc94af68fe8e5d2690c14beedb3ccba394bd62b88f2744e0520cf13635a3c59',
          'math-hook.py': 'c22963a86380fc77093323cb610eeb7355fda9033d8b34ff036a53566b7f1256'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_profile(actual, reference):
    assert {k: v for k, v in actual.items() if k not in ('task', 'run_id')} == reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--epoch', type=Path, required=True)
    args = parser.parse_args(); epoch = args.epoch
    for name, expected in SOURCE.items():
        assert sha(epoch/'sources'/name) == expected, name
    assert sha(epoch/'performance/sources/t02_sglang_bench.py') == WRAPPER
    configs = list((epoch/'performance/runtime/before').glob('config-*.json')); assert len(configs) == 1
    check_profile(json.loads(configs[0].read_text()), json.loads((args.project/'baselines/performance-profile021.json').read_text()))
    base = args.project/'scripts/analyze_sglang_results_004.py'
    assert sha(base) == 'd73ecca728b47b8f58fed407f13c9a496e73ba8e165ee08b740afd9065efdaa8'
    spec = importlib.util.spec_from_file_location('native_epoch021', base)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    assert module.APPROVED_DRIVER_SHA256 == 'ff8eef852e19f336e99884cc031353627ca304c6a7534c22375116758a850000'
    module.APPROVED_DRIVER_SHA256 = WRAPPER
    contract = module.row_contract
    original_body = '{"temperature":0,"ignore_eos":true,"stream_options":{"include_usage":true}}'
    def profile_contract(run, row):
        view = copy.deepcopy(row)
        try:
            at = view['command'].index('--extra-request-body') + 1
            assert json.loads(view['command'][at]) == BODY
            view['command'][at] = original_body
        except (ValueError, KeyError, IndexError, AssertionError, TypeError):
            name, errors = contract(run, row)
            return name, errors + ['Declared four-placeholder token policy absent or different']
        return contract(run, view)
    module.row_contract = profile_contract
    report = module.analyze(epoch/'performance/run.json', epoch/'analysis')
    print(json.dumps(report['summary']))
    return 0 if report['summary']['selected_reports_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
