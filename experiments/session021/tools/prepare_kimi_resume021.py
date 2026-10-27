"""Derive two bounded continuation scripts without changing the running sources."""
from pathlib import Path
import ast
import hashlib
import json


HERE = Path(__file__).parent
OUT = HERE.parents[1] / 'evidence/session021/prepared'


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    original = HERE / 'run_kimi_performance016.py'
    text = original.read_text(encoding='utf-8')
    text = replace_once(text, 'kimi-performance005', 'kimi-performance006')
    text = replace_once(text, "OUT = Path('/root/private_data/guoxin-session016/kimi-performance006')",
                        "OUT = Path('/root/private_data/guoxin-session021/kimi-performance006')")
    text = replace_once(text, "'--timeout-seconds','5400','--total-timeout-seconds','20000'",
                        "'--timeout-seconds','9000','--total-timeout-seconds','19000','--load','16384:64'")
    text = replace_once(text, "'benchmark',20100,client_env", "'benchmark',19100,client_env")
    text = replace_once(text, "'/root/session016-' + OUT.name", "'/root/session021-' + OUT.name")
    text = replace_once(text, '"""Six template loads, two rounds, on the full OCRBench-verified configuration."""',
                        '"""Only the two unfinished 16K/c64 cases, same verified profile; finite larger wall budget."""')
    # The existing compatibility policy, driver, environment and 1024 output
    # budget remain byte-for-byte the same literals in this derived controller.
    ast.parse(text)
    performance = OUT / 'run_kimi_performance021.py'
    performance.write_text(text, encoding='utf-8', newline='\n')

    original_replay = HERE / 'finalize_kimi_replay017.py'
    replay = original_replay.read_text(encoding='utf-8')
    replay = replace_once(replay, "BASE=Path('/root/private_data/guoxin-session016/kimi-performance005')",
                          "BASE=Path('/root/private_data/guoxin-session021/coordinator001')")
    replay = replace_once(replay, "OUT=Path('/root/private_data/guoxin-session017/kimi-portable-replay001')",
                          "OUT=Path('/root/private_data/guoxin-session021/kimi-portable-replay002')")
    begin = replay.index('    deadline=time.monotonic()+28800\n')
    end = replay.index('    for port in (30000,30001,30109,30119,30129,30139):\n', begin)
    replay = replay[:begin] + '''    combined=BASE/'combined-native-matrix.json'
    proof=json.loads(combined.read_text())
    assert proof['complete_native_matrix'] and proof['measured_requests']==608
    record.update(state='combined_performance_verified',complete_native_matrix_verified=True,
                  combined_proof_sha256=sha(combined),raw_sources_preserved=True)
    save(record)
''' + replay[end:]
    replay = replace_once(replay, "'/root/session017-kimi-portable-replay001.zip'",
                          "'/root/session021-kimi-portable-replay002.zip'")
    replay = replace_once(replay, "'state':'waiting_performance005'", "'state':'checking_combined_performance'")
    replay = replace_once(replay, '"""Wait for owned performance005, audit all loads, then replay fixed Git entries."""',
                          '"""Replay the unchanged fixed Git root entries after the combined 12-report audit."""')
    ast.parse(replay)
    replay_path = OUT / 'finalize_kimi_replay021.py'
    replay_path.write_text(replay, encoding='utf-8', newline='\n')
    files = [original, original_replay, performance, replay_path]
    manifest = {'files': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                           'bytes': p.stat().st_size} for p in files],
                'changes': ['Only unfinished 16K/c64 two rounds; case9000s/total19000s/parent19100s',
                            'Service remains finite21600s; model, 1024 budget, sampling and token policy unchanged',
                            'New root replay uses combined derived proof and new exclusive output paths'],
                'running_files_modified': False}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
