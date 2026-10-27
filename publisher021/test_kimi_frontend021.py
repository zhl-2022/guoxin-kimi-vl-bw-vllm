"""CPU-only safety checks against the twelve actually measured native argv."""
from pathlib import Path
import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--measured-matrix', type=Path, required=True)
    parser.add_argument('--raw-run', action='append', type=Path, required=True)
    args = parser.parse_args(); project = args.project.resolve()
    measured = json.loads(args.measured_matrix.read_text())
    assert measured['complete_native_matrix'] and len(measured['rows']) == 12
    source_commands = {}
    expected_rows = {row['name']: row for row in measured['rows']}
    for path in args.raw_run:
        raw = json.loads(path.read_text())
        for row in raw['rows']:
            expected_row = expected_rows.get(row['name'])
            if (expected_row and row.get('exit_code') == 0 and row.get('process_status') == 'exited'
                    and row.get('native_result_sha256') == expected_row['native_file_sha256']):
                assert row['name'] not in source_commands
                source_commands[row['name']] = row['command']
    assert set(source_commands) == set(expected_rows)
    spec = importlib.util.spec_from_file_location('bench_control021', project/'scripts/bench_matrix021.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    reference = json.loads((project/'baselines/performance-profile021.json').read_text())
    for name in ['bench_matrix021.py', 'audit_kimi_epoch021.py', 'prepare_bench_client021.py']:
        ast.parse((project/'scripts'/name).read_text())
    with tempfile.TemporaryDirectory(prefix='kimi021-control-') as temp:
        temp = Path(temp); interpreter = temp/'venv/bin/python'; interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        env = os.environ.copy(); env['KIMI_CLIENT_PYTHON'] = str(interpreter)
        output = temp/'must-not-exist'
        command = [sys.executable, str(project/'scripts/bench_matrix021.py'), '--project', str(project),
                   '--plan-only', '--output-dir', str(output)]
        plan = json.loads(subprocess.run(command, capture_output=True, text=True, env=env, check=True).stdout)
        assert plan['client_python'] == str(interpreter) and not output.exists()
        expected = {row['name']: row for row in measured['rows']}; generated = {}
        for group in plan['epochs']:
            argv = [sys.executable, str(project/'scripts/t02_sglang_bench_016.py'),
                    '--client-python', str(interpreter), '--model-path', reference['model_path'],
                    '--served-model-name', reference['served_name'], '--port', str(reference['port']),
                    '--output-dir', str(temp/group['name']), '--timeout-seconds', '9000',
                    '--total-timeout-seconds', str(group['total_seconds'])]
            for load in group['loads']: argv += ['--load', load]
            native = json.loads(subprocess.run(argv, capture_output=True, text=True, check=True).stdout)
            assert not native['client_started']
            for row in native['rows']:
                assert row['name'] not in generated
                generated[row['name']] = row
                actual = expected[row['name']]
                def comparable(command):
                    command = list(command[1:])
                    for option in ('--output-file', '--output-dir'):
                        if option in command:
                            command[command.index(option)+1] = '<exclusive-output-path>'
                    return command
                # Compare with real recorded native invocations, not a second
                # copy of the new planner's own implementation.
                assert comparable(row['command']) == comparable(source_commands[row['name']]), row['name']
                assert row['expected_requests'] == actual['expected_requests']
                assert row['output_target_tokens'] == 1024
            assert group['total_seconds']+1320+150+180+100 < 21600
        assert len(generated) == 12 and sum(row['expected_requests'] for row in generated.values()) == 608
        run_id = '1'*32; state = {'status': 'ready', 'run_id': run_id}
        config = dict(reference, task='/example/owned/task', run_id=run_id)
        assert module.check_owner(state, config, reference) == run_id
        invalid = [(dict(state, run_id='../../other'), config),
                   (dict(state, status='starting'), config),
                   (state, dict(config, run_id='2'*32)),
                   (state, dict(config, model_path='/other/model')),
                   (state, dict(config, port=12345))]
        for bad_state, bad_config in invalid:
            try: module.check_owner(bad_state, bad_config, reference)
            except AssertionError: pass
            else: raise AssertionError('Unrelated or mismatched owner accepted')
    print(json.dumps({'CPU_control_checks_passed': True, 'registered_cases': 12,
                      'planned_measurement_requests': 608, 'owner_rejections': len(invalid),
                      'venv_symlink_preserved': True, 'plan_created_output': False,
                      'GPU_model_requests': 0, 'this_is_not_a_model_accuracy_result': True}))


if __name__ == '__main__':
    main()
