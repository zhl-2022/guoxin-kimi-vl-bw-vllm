"""Run twelve native cases as two owned six-hour service epochs.

Model, client and sampling stay fixed. Exclusive directories and UUID checks
protect previous outputs and other services. No Notebook shutdown is performed.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import uuid


GROUPS = [('epoch1', ['4096:1', '4096:8', '4096:64', '16384:1', '16384:8'], 15000),
          ('epoch2', ['16384:64'], 19000)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(actual):
    return {k: v for k, v in actual.items() if k not in ('task', 'run_id')}


def checked_id(value):
    assert isinstance(value, str) and len(value) == 32 and all(x in '0123456789abcdef' for x in value)
    return value


def check_owner(state, config, expected):
    assert state['status'] == 'ready' and state['run_id'] == config['run_id']
    checked_id(state['run_id'])
    assert profile(config) == expected, 'Existing project service differs; left untouched'
    return state['run_id']


def registered(loads):
    return {(int(n), int(c), r) for n, c in (x.split(':') for x in loads) for r in (1, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(); project = args.project.resolve()
    expected = json.loads((project/'baselines/performance-profile021.json').read_text())
    all_keys = set().union(*(registered(loads) for _, loads, _ in GROUPS))
    assert len(all_keys) == 12 and sum(max(8, 2*c) for n, c, r in all_keys) == 608
    client = os.environ.get('KIMI_CLIENT_PYTHON')
    if not client:
        baked = Path('/opt/guoxin/kimi-client021/bin/python')
        client = str(baked if baked.exists() else project/'.venv-client/bin/python')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    # Keep the venv interpreter spelling: resolving its symlink bypasses pyvenv.cfg.
    client = str(Path(client).expanduser().absolute())
    out = (args.output_dir or Path(os.environ.get('KIMI_BENCH_OUTPUT', str(project/'bench'/(stamp+'-'+uuid.uuid4().hex[:8]))))).expanduser().resolve()
    plan = {'mode': 'plan_only' if args.plan_only else 'run', 'output': str(out), 'client_python': client,
            'epochs': [{'name': name, 'loads': loads, 'total_seconds': seconds,
                        'registered_cases': sorted(registered(loads))} for name, loads, seconds in GROUPS],
            'per_case_seconds': 9000, 'service_seconds_per_epoch': 21600,
            'measured_requests': 608, 'output_tokens': 1024, 'model_requests_started': False,
            'Notebook_shutdown_requested': False, 'submitted': False}
    if args.plan_only:
        print(json.dumps(plan, indent=2)); return 0
    assert sys.platform == 'linux'
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    out.mkdir(parents=True, exist_ok=False)
    record = dict(plan, started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), phases={}, complete=False)
    def save():
        temp = out/'summary.tmp'; temp.write_text(json.dumps(record, indent=2)+'\n'); temp.replace(out/'summary.json')
    def step(argv, directory, label, seconds, env=None):
        with (directory/(label+'.log')).open('xb') as log:
            child = subprocess.Popen(argv, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = child.wait(timeout=seconds)
            except BaseException:
                child.send_signal(signal.SIGINT)
                child.wait(timeout=120)
                raise
        record['phases'][str(directory.relative_to(out))+'/'+label] = code; save(); return code
    save()
    if not Path(client).is_file():
        assert client == str(project/'.venv-client/bin/python'), 'Explicit client missing; no replacement guessed'
        assert step([sys.executable, str(project/'scripts/prepare_bench_client021.py'),
                     '--output', str(project/'.venv-client')], out, 'prepare-client', 1000) == 0
    task = project/'tasks/kimi-vl-bw-vllm'; runtime = task/'runtime'
    driver = project/'scripts/kimi_vl_runtime_005.py'; wrapper = project/'scripts/t02_sglang_bench_016.py'
    sources = {'driver.py': driver, 'client-hook.py': project/'scripts/kimi-client-hook/sitecustomize.py',
               'math-hook.py': project/'scripts/kimi-math-hook/sitecustomize.py'}
    expected_sha = {'driver.py': '11985f3c844f03b320f674309ea19060e1bc2f1ac93619c031092d574af22dcc',
                    'client-hook.py': '8dc94af68fe8e5d2690c14beedb3ccba394bd62b88f2744e0520cf13635a3c59',
                    'math-hook.py': 'c22963a86380fc77093323cb610eeb7355fda9033d8b34ff036a53566b7f1256'}
    for name, source in sources.items(): assert sha(source) == expected_sha[name], name
    assert sha(wrapper) == 'eab500bbc9bd30208a6020d3dae8cef2df5a247e1391066c5303025fd6cef83a'
    def closed():
        with socket.socket() as sock: return sock.connect_ex(('127.0.0.1', expected['port'])) != 0
    for other_port in (30000, 30001, 30002, 30109, 30119, 30139):
        with socket.socket() as sock:
            assert sock.connect_ex(('127.0.0.1', other_port)) != 0, ('Other existing service left untouched', other_port)
    state_path = runtime/'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['status'] == 'ready':
            checked_id(state['run_id'])
            config = json.loads((runtime/('config-'+state['run_id']+'.json')).read_text())
            prior_id = check_owner(state, config, expected)
            assert step([sys.executable, str(driver), 'stop', '--task-dir', str(task), '--run-id', prior_id],
                        out, 'stop-previous-owned-service', 100) == 0
        else:
            assert state['status'] in ('stopped', 'failed', 'expired'), 'Existing launch still active; left untouched'
    assert closed(), 'Port occupied by an unidentified service; left untouched'
    selected = []
    for name, loads, seconds in GROUPS:
        epoch = out/name; (epoch/'sources').mkdir(parents=True, exist_ok=False)
        for label, source in sources.items(): shutil.copyfile(source, epoch/'sources'/label)
        run_id = uuid.uuid4().hex
        (epoch/'run-start.json').write_text(json.dumps({'run_id': run_id, 'loads': loads,
            'source_sha256': sha(Path(__file__)), 'accuracy_measured_by_benchmark': False, 'output_budget': 1024}, indent=2)+'\n')
        launched = False
        try:
            launched = True; record['model_requests_started'] = True; save()
            assert step([sys.executable, str(driver), 'launch', '--task-dir', str(task), '--run-id', run_id], epoch, 'launch', 1320) == 0
            actual = json.loads((runtime/('config-'+run_id+'.json')).read_text())
            assert profile(actual) == expected, 'Service profile differs from declared profile'
            env = os.environ.copy(); env.update(PYTHONPATH=str(project/'scripts/kimi-client-hook')+':'+str(project/'scripts'),
                KIMI_CLIENT_MATH_HOOK=str(sources['math-hook.py']), KIMI_CLIENT_COMPAT_AUDIT=str(epoch/'client-compat.jsonl'))
            command = [sys.executable, str(wrapper), '--run', '--client-python', client,
                '--allow-client-device-discovery', '--model-path', expected['model_path'], '--served-model-name', expected['served_name'],
                '--port', str(expected['port']), '--runtime-dir', str(runtime), '--expected-run-id', run_id,
                '--output-dir', str(epoch/'performance'), '--timeout-seconds', '9000', '--total-timeout-seconds', str(seconds)]
            for load in loads: command += ['--load', load]
            step(command, epoch, 'native-client', seconds+150, env)
            assert step([sys.executable, str(project/'scripts/audit_kimi_epoch021.py'), '--project', str(project),
                         '--epoch', str(epoch)], epoch, 'native-audit', 180) == 0
            audit = json.loads((epoch/'analysis/analysis.json').read_text())
            keys = {(r['text_target_tokens'], r['concurrency'], r['round']) for r in audit['rows'] if r['eligible_client_report']}
            assert keys == registered(loads)
            selected += [dict(row, epoch=name) for row in audit['rows']]
        finally:
            if launched:
                assert step([sys.executable, str(driver), 'stop', '--task-dir', str(task), '--run-id', run_id], epoch, 'stop', 100) == 0
                assert closed(), 'Owned service cleanup is not confirmed'
    assert len(selected) == 12 and {(r['text_target_tokens'],r['concurrency'],r['round']) for r in selected} == all_keys
    for iteration in (1, 2):
        with (out/f'round{iteration}.jsonl').open('xb') as stream:
            for row in selected:
                if row['round'] == iteration:
                    source = out/row['epoch']/'analysis'/row['normalized_native_file']
                    native = json.loads(source.read_text())
                    stream.write((json.dumps(native, separators=(',', ':'), allow_nan=False)+'\n').encode())
    record.update(complete=True, native_reports=12, measured_requests=608, rows=selected,
                  round_files='Derived compact views; only request_rate Infinity normalization, raw files retained',
                  output_length_boundary='Native usage/fallback counts; no independent wire-token assertion',
                  ended_utc=dt.datetime.now(dt.timezone.utc).isoformat()); save()
    print(json.dumps({'output': str(out), 'complete': True, 'native_reports': 12, 'measured_requests': 608}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
