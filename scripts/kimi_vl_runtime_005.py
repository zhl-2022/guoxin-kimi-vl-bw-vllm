#!/usr/bin/env python3
"""Pinned Kimi-VL pilot configuration; no dependency installation or architecture overrides.

Copyright (c) 2026 the task author. Copyright scope only; no new license selected.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import recon2reason_runtime as supervisor
from download_candidate_models_20261023 import check_relative, file_digest

MODEL = "moonshotai/Kimi-VL-A3B-Instruct"
REVISION = "398eede0903cd983a2bfa0cc634e9ac1d843f375"
TASK = "kimi-vl-bw-vllm"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_task(task):
    task = Path(task).resolve()
    data = read(task / "task.json")
    if task.name != TASK or data.get("model_id") != MODEL or data.get("model_revision") != REVISION:
        raise ValueError("Only the pinned Kimi-VL task is accepted")
    return task


def configuration(task):
    value = os.environ.get("KIMI_MODEL_PATH", "/root/private_data/guoxin-candidates/models/kimi-vl")
    if not value:
        raise ValueError("KIMI_MODEL_PATH must be a local directory")
    model = str(Path(value).expanduser().resolve())
    port = int(os.environ.get("KIMI_PORT", "30129"))
    memory = float(os.environ.get("KIMI_MEM_FRAC", "0.80"))
    length = int(os.environ.get("KIMI_MAX_LEN", "4096"))
    load = int(os.environ.get("KIMI_LOAD_TIMEOUT", "900"))
    runtime = int(os.environ.get("KIMI_MAX_RUNTIME_SECONDS", "1800"))
    grace = int(os.environ.get("KIMI_STOP_GRACE_SECONDS", "30"))
    served = os.environ.get("KIMI_SERVED_NAME", model)
    if (not 1024 <= port <= 65535 or not 0 < memory < 1 or not 1024 <= length <= 32768
            or not 60 <= load <= runtime <= 6900 or not 1 <= grace <= 60):
        raise ValueError("Invalid pilot port/memory/context/time bounds; service lifetime max 6900 seconds")
    if not served or any(x in served for x in "\r\n"):
        raise ValueError("Invalid served model name")
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
        "--tokenizer", model, "--served-model-name", served, "--host", "127.0.0.1", "--port", str(port),
        "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--max-model-len", str(length),
        "--gpu-memory-utilization", str(memory), "--max-num-seqs", "1",
        "--max-num-batched-tokens", str(length), "--enforce-eager", "--trust-remote-code",
        "--limit-mm-per-prompt", '{"image":2,"video":0}']
    return {"task":str(Path(task).resolve()), "framework":"vllm", "precision":"bf16", "model_id":MODEL,
        "model_revision":REVISION, "model_path":model, "served_name":served, "host":"127.0.0.1",
        "port":port, "tp":1, "max_len":length, "mem_frac":memory, "load_timeout":load,
        "max_runtime":runtime, "stop_grace":grace, "max_num_seqs":1, "command":command,
        "url":f"http://127.0.0.1:{port}", "trust_remote_code_scope":"Exact reviewed publisher files only; verified before actual launch",
        "attention_backend_override":None, "MLA_or_MoE_kernel_forced":False,
        "GPU_verified":False, "official_accuracy_verified":False, "official_performance_verified":False,
        "standard_submission_ready":False}


def checkpoint(task, model_dir, hash_weights=False):
    lock = read(task / "model-lock.json")
    if lock.get("model_id") != MODEL or lock.get("revision") != REVISION:
        raise ValueError("Checkpoint lock identity differs")
    base = Path(model_dir).resolve()
    rows = []
    for item in lock["files"]:
        relative = check_relative(item["path"])
        path = base.joinpath(*relative.parts)
        okay = path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(base)
        okay = okay and all(not ancestor.is_symlink() for ancestor in path.parents if ancestor != base.parent)
        row = {"path":item["path"], "size_ok":bool(okay and path.stat().st_size == item["size"]), "identity_ok":False}
        if row["size_ok"] and (hash_weights or item["kind"] != "weight"):
            row["actual_sha256"] = file_digest(path)
            row["identity_ok"] = row["actual_sha256"] == item["sha256"]
        elif row["size_ok"]:
            row["reason"] = "weight_hash_not_requested"
        rows.append(row)
    source_lock = read(task / "upstream-lock.json")
    review = read(task / "source-review-approved.json")
    if (source_lock.get("revision") != REVISION or review.get("revision") != REVISION
            or review.get("static_review_approved") is not True):
        raise ValueError("Pinned publisher source review is absent/unapproved")
    reviewed = {row["path"]:row["sha256"] for row in source_lock["files"] if row["path"].endswith(".py")}
    required = {row["path"]:row["sha256"] for row in lock["files"] if row["path"].endswith(".py")}
    if required != reviewed or review.get("reviewed_python_sha256") != required:
        raise ValueError("Remote Python source differs from approved exact file set")
    extra = []
    if base.is_dir():
        def walk_error(error): raise error
        for directory, dirs, names in os.walk(base, followlinks=False, onerror=walk_error):
            for name in dirs:
                if (Path(directory)/name).is_symlink():
                    raise ValueError("Checkpoint directory contains a symlink")
            for name in names:
                path = Path(directory)/name
                relative = path.relative_to(base).as_posix()
                if path.suffix in {".py", ".pyc", ".so", ".pyd", ".safetensors", ".bin", ".pth", ".pt"} and relative not in {x["path"] for x in lock["files"]}:
                    extra.append(relative)
    config = read(base / "config.json") if (base/"config.json").is_file() else {}
    architecture_ok = config.get("architectures") == ["KimiVLForConditionalGeneration"]
    return {"model_id":MODEL, "revision":REVISION, "directory":str(base), "weights_hashed":hash_weights,
        "files":rows, "extra_executable_files":extra, "architecture_matches_publisher":architecture_ok,
        "source_review_static_only":True, "complete":bool(rows) and all(x["identity_ok"] for x in rows)
            and not extra and architecture_ok, "GPU_verified":False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch","stop","env","preflight","verify","bench"))
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--hash-weights", action="store_true")
    parser.add_argument("--selfcheck", "--selfcheck-only", action="store_true")
    args = parser.parse_args(argv); task = checked_task(args.task_dir)
    if args.action == "stop":
        supervisor.checked_run_id(args.run_id)
        if args.dry_run:
            print(json.dumps({"status":"plan_only","process_signalled":False,"run_id":args.run_id})); return 0
        return supervisor.stop(task,args.run_id)
    config = configuration(task)
    if args.action == "preflight":
        result = checkpoint(task,config["model_path"],args.hash_weights)
        print(json.dumps(result,indent=2)); return 0 if result["complete"] else 2
    if args.action == "env" and args.dry_run:
        packages = {}
        for name in ("torch","vllm","vllm-hcu","transformers","tiktoken","Pillow"):
            try: packages[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: packages[name]=None
        print(json.dumps({"metadata_only":True,"configuration":config,"packages":packages,"GPU_tested":False},indent=2));return 0
    if args.action == "launch" and not args.dry_run:
        if sys.platform != "linux": return supervisor.fail("Actual launch requires SCNet Linux")
        result = checkpoint(task,config["model_path"],True)
        if not result["complete"]:
            print(json.dumps({"checkpoint_identity_failed":result},indent=2)); return 2
    original = supervisor.configuration
    supervisor.configuration = lambda _task: config
    try:
        if args.action == "launch": return supervisor.launch(task,args.dry_run,args.run_id)
        if args.action == "env": return supervisor.environment(task)
        if args.action == "bench" and config["max_len"] < 17408:
            print(json.dumps({"status":"pilot_profile_not_full_T02_load","executed":False,
                "reason":"4K pilot cannot contain the template 16K+1K load; choose and verify a larger profile before full bench",
                "official_performance_verified":False,"submission_ready":False},indent=2)); return 0 if args.dry_run else 2
        return supervisor.official(task,args.action,args.dry_run,args.selfcheck)
    finally: supervisor.configuration = original


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (OSError,ValueError,KeyError) as exc:
        print(json.dumps({"error":str(exc),"submission_ready":False}));raise SystemExit(2)
