"""Safely replace only the two under-budgeted Kimi cases, then release owned queues.

No Notebook shutdown, remote B action, credential use or platform limit change.
The waiting Robo process is held in memory, with its original wait deadline
unchanged. AuK remains alive and waiting on that same Robo PID.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import resource
import signal
import socket
import subprocess
import time


OUT = Path('/root/private_data/guoxin-session021/coordinator001')
OLD = Path('/root/private_data/guoxin-session016/kimi-performance005')
NEW = Path('/root/private_data/guoxin-session021/kimi-performance006')
ROBO = Path('/root/private_data/guoxin-session017/robo-refspatial001')
REPLAY = Path('/root/private_data/guoxin-session017/kimi-portable-replay001')
PORTS = (30000, 30001, 30002, 30109, 30119, 30129, 30139)
PLAN = {
    'performance': ('/root/run_kimi_performance021.py', '984c3cf889afae24639930c20db4f71309a0c01dd223c9f5aee5577b842f6593'),
    'replay': ('/root/finalize_kimi_replay021.py', 'b2d171f0f8c3ef791c667695b0ce5a1ff610da091b55cd2a026fc81673e6ff40'),
    'analyzer': ('/root/analyze_kimi_performance017.py', 'e0923f4d71233dd7d43e1070024f4b18aa6ff234ef872bf73dc5363484b8e819'),
    'base_analyzer': ('/root/analyze_sglang_results_004.py', 'd73ecca728b47b8f58fed407f13c9a496e73ba8e165ee08b740afd9065efdaa8'),
    'combiner': ('/root/combine_kimi_native021.py', 'ece6c506223d11a134ce1b920180d686135b4d027b386c76c7e8f7d3bb9a806b'),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(record):
    temp = OUT / 'progress.tmp'
    temp.write_text(json.dumps(record, indent=2) + '\n')
    temp.replace(OUT / 'progress.json')


def process(pid):
    root = Path('/proc') / str(pid)
    try:
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        argv = (root / 'cmdline').read_bytes().split(b'\x00')
        return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]), 'token': fields[19],
                'argv': [item.decode() for item in argv if item]}
    except FileNotFoundError:
        return None


def matches(identity):
    value = process(identity['pid'])
    return value and value['state'] != 'Z' and value['token'] == identity['token'] and value['argv'] == identity['argv']


def signal_owned(identity, sig):
    assert matches(identity), ('Process identity changed; signal refused', identity)
    os.kill(identity['pid'], sig)


def wait_gone(identity, seconds):
    deadline = time.monotonic() + seconds
    while matches(identity):
        assert time.monotonic() < deadline, ('Owned cleanup did not finish', identity)
        time.sleep(1)


def ports_closed():
    for port in PORTS:
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                return False
    return True


def run(record, argv, label, timeout):
    with (OUT / (label + '.log')).open('xb') as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        record.update(state=label, current_child=process(child.pid))
        save(record)
        # Finite parent timeout exceeds each child's own service lifetime.
        # On an unexpected parent timeout, preserve the child and hold queues
        # for the next monitor to inspect; never kill an unknown descendant.
        value = child.wait(timeout=timeout)
    record['phases'][label] = value
    record['current_child'] = None
    save(record)
    return value


def audit(record, attempt, label):
    destination = OUT / label
    value = run(record, ['python3', PLAN['analyzer'][0], '--run', str(attempt / 'performance/run.json'),
                        '--analysis-dir', str(destination), '--base-analyzer', PLAN['base_analyzer'][0],
                        '--reference-wrapper', '/root/t02_sglang_bench_004g.py'], label, 180)
    assert value in (0, 1) and (destination / 'analysis.json').is_file()
    return json.loads((destination / 'analysis.json').read_text())


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for path, expected in PLAN.values():
        assert sha(path) == expected, path
    assert sha('/root/run_kimi_performance016e.py') == '52118c539a0a1428bb20c2d9ca8ef9b6bcafd2ada77d7233f49d89f8eb9445c7'
    assert sha('/root/session017-kimi-portable-project.zip') == 'd5a4da75bf8c373c6abeb467c0e25c5199a7a8db01f58c09d86d2d7a7bcf29d2'
    identities = {pid: process(pid) for pid in (38105, 43060, 43964, 45258)}
    expected_scripts = {38105: '/root/run_kimi_performance016e.py', 43060: '/root/finalize_kimi_replay017.py',
                        43964: '/root/private_data/guoxin-session017/refspatial-stage001/tools/run_refspatial_queue017.py',
                        45258: '/root/private_data/guoxin-session019/auk-zhhard-stage001/tools/run_auk_zhhard_queue019.py'}
    for pid, name in expected_scripts.items():
        assert identities[pid] and name in identities[pid]['argv'] and identities[pid]['state'] not in ('Z', 'T')
    assert json.loads((REPLAY / 'progress.json').read_text())['state'] == 'waiting_performance005'
    robo = json.loads((ROBO / 'progress.json').read_text())
    assert robo['state'] == 'waiting_owned_kimi_performance_and_replay' and not robo['GPU_started']
    auk = json.loads(Path('/root/private_data/guoxin-session019/auk-zhhard001/progress.json').read_text())
    assert auk['state'] == 'waiting_owned_robo_queue' and not auk['GPU_started']
    parent_run = json.loads((OLD / 'performance/run.json').read_text())
    active = [row for row in parent_run['rows'] if row['status'] == 'running']
    assert len(active) == 1 and active[0]['name'] == 'round1_16384_1024_c64' and active[0]['allocated_timeout_seconds'] == 5400
    wrapper = []
    for folder in Path('/proc').glob('[0-9]*'):
        candidate = process(int(folder.name))
        if candidate and candidate['ppid'] == 38105 and str(OLD / 't02_sglang_bench-executed.py') in candidate['argv']:
            wrapper.append(candidate)
    assert len(wrapper) == 1
    OUT.mkdir(parents=True, exist_ok=False)
    record = {'started_utc': now(), 'controller_pid': os.getpid(), 'source_sha256': sha(__file__),
              'state': 'validated_transition_preflight', 'owned_identities': identities,
              'wrapper_identity': wrapper[0], 'phases': {}, 'robo_held': False, 'robo_released': False,
              'old_raw_retained': True, 'instance_shutdown_requested': False, 'submitted': False,
              'reason': 'Observed ~98min/case exceeds90min allocation; replace only two unfinished cases with finite150min budget',
              'Auk_waiter_unchanged': True, 'B_instance_untouched': True}
    save(record)
    try:
        signal_owned(identities[43964], signal.SIGSTOP)
        record.update(robo_held=True, state='holding_waiting_robo_only')
        save(record)
        stop_deadline = time.monotonic() + 5
        while process(43964)['state'] != 'T':
            assert time.monotonic() < stop_deadline, 'Waiting Robo did not enter scheduler hold'
            time.sleep(0.05)
        signal_owned(identities[43060], signal.SIGINT)
        wait_gone(identities[43060], 60)
        assert json.loads((REPLAY / 'progress.json').read_text())['state'] == 'failed'
        # Signal the client wrapper, whose bounded() reaps its native process
        # group. Let the original parent do its normal service stop and ZIP.
        signal_owned(wrapper[0], signal.SIGINT)
        record.update(state='waiting_original_graceful_cleanup', interruption_utc=now())
        save(record)
        wait_gone(identities[38105], 240)
        assert json.loads((OLD / 'run-final.json').read_text())['owned_service_stopped']
        assert ports_closed()
        old = audit(record, OLD, 'old-ten-analysis')
        assert old['summary']['eligible_client_reports'] == 10 and old['summary']['eligible_client_reported_requests'] == 352
        run(record, ['python3', '-u', PLAN['performance'][0]], 'remaining_two_native_cases', 22500)
        assert json.loads((NEW / 'run-final.json').read_text())['owned_service_stopped'] and ports_closed()
        new = audit(record, NEW, 'new-two-analysis')
        assert new['summary']['eligible_client_reports'] == 2 and new['summary']['eligible_client_reported_requests'] == 256
        assert run(record, ['python3', '/root/combine_kimi_native021.py',
                            '--old-analysis', str(OUT / 'old-ten-analysis/analysis.json'),
                            '--new-analysis', str(OUT / 'new-two-analysis/analysis.json'),
                            '--output', str(OUT / 'combined-native-matrix.json')], 'combined_native_audit', 180) == 0
        record['complete_matrix_verified'] = True
        save(record)
        assert run(record, ['python3', '-u', PLAN['replay'][0]], 'fixed_root_replay', 8500) == 0
        replay = json.loads(Path('/root/private_data/guoxin-session021/kimi-portable-replay002/progress.json').read_text())
        assert replay['state'] == 'complete' and replay['root_entry_replayed'] and replay['owned_service_stopped']
        record.update(state='complete_Kimi_matrix_and_root_replay', root_replay_score=replay['accuracy_score'])
    except BaseException as exc:
        record.update(state='transition_or_continuation_failed', error=repr(exc))
    finally:
        active_child = record.get('current_child')
        safe = ports_closed() and not matches(identities[38105]) and not matches(identities[43060])
        if active_child and matches(active_child):
            safe = False
        if record['robo_held'] and safe:
            signal_owned(identities[43964], signal.SIGCONT)
            record.update(robo_released=True, robo_released_utc=now())
        elif record['robo_held']:
            record['recovery_required_before_queue_release'] = True
        record['ended_utc'] = now()
        save(record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
