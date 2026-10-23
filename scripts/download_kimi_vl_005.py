#!/usr/bin/env python3
"""Exact Kimi checkpoint verification/copy or resumable official downloads. No GPU/pip.

Copyright scope only; publisher notices remain with their original files.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

from download_candidate_models_20261023 import check_relative, download_file, file_digest, json_new, verified_file
from kimi_vl_runtime_005 import MODEL, REVISION, checkpoint

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/kimi-vl-bw-vllm"
BASE = Path("/root/private_data/guoxin-candidates")


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def source_files(task, provider):
    lock = read(task / "model-lock.json"); sources = read(task / "download-sources.json")
    if (lock.get("model_id") != MODEL or lock.get("revision") != REVISION
            or sources.get("original_huggingface_revision") != REVISION):
        raise ValueError("Pinned model identity differs")
    expected = {x["path"]:x for x in lock["files"]}
    if len(expected) != len(lock["files"]) or set(expected) != {x["path"] for x in sources["files"]}:
        raise ValueError("Exact source/lock file sets differ")
    rows = []
    if provider == "modelscope" and sources.get("modelscope_complete") is not True:
        raise ValueError("ModelScope source mapping is not fully verified")
    for item in sources["files"]:
        check_relative(item["path"]); required = expected[item["path"]]
        if (item["size"] != required["size"] or item["sha256"] != required["sha256"]
                or not re.fullmatch(r"[0-9a-f]{64}",item["sha256"])):
            raise ValueError("Source hash/size differs from pinned checkpoint")
        revision = REVISION if provider == "huggingface" else item["modelscope_revision"]
        model = MODEL if provider == "huggingface" else sources["modelscope_model_id"]
        domain = "huggingface.co" if provider == "huggingface" else "modelscope.cn/models"
        url = f"https://{domain}/{model}/resolve/{revision}/{item['path']}"
        if not re.fullmatch(r"[0-9a-f]{40}",revision) or item[provider+"_url"] != url:
            raise ValueError("Only exact official immutable source URLs are accepted")
        rows.append({**required,"download_url":url})
    return lock, rows


def reject_links(path):
    for item in (Path(path),*Path(path).parents):
        if item.is_symlink(): raise ValueError("Symlink destination/ancestor is refused")


def destination_path(value):
    path = Path(value)
    if not path.is_absolute() or not path.is_relative_to(BASE / "models") or path == BASE/"models":
        raise ValueError("Destination must stay below the dedicated private-data models directory")
    namespace=path.relative_to(BASE/"models").parts[0]
    if namespace!="kimi-vl" and not namespace.startswith("kimi-vl-"):
        raise ValueError("Only the Kimi namespace is writable; other candidate models are protected")
    if ".." in path.parts: raise ValueError("Unnormalized destination refused")
    reject_links(path)
    return path


def apply(rows, destination, max_seconds, copy_source=None, reuse_verified_only=False):
    destination = destination_path(destination)
    reject_links(BASE); BASE.mkdir(parents=True,exist_ok=True)
    identity = uuid.uuid4().hex; lockpath = BASE/"download.lock"
    descriptor = os.open(lockpath,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(descriptor,"w") as handle:json.dump({"run_id":identity,"pid":os.getpid()},handle)
    report = {"run_id":identity,"started_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode":"copy_existing" if copy_source else "download", "status":"failed", "files":[],
        "destination":str(destination), "weight_identity":"Exact per-file SHA256, never a mutable main label", "GPU_imported":False}
    deadline = time.monotonic()+max_seconds
    try:
        destination.mkdir(parents=True,exist_ok=True)
        needed = sum(row["size"] for row in rows if not (destination/row["path"]).is_file())
        if shutil.disk_usage(destination).free < needed + 512*1024*1024:
            raise ValueError("Insufficient storage for full checkpoint plus reserve")
        for row in rows:
            target = destination.joinpath(*check_relative(row["path"]).parts)
            reject_links(target); target.parent.mkdir(parents=True,exist_ok=True)
            if copy_source:
                source = Path(copy_source)/row["path"];reject_links(source)
                if not verified_file(source,row,deadline):
                    if reuse_verified_only:
                        report["files"].append({"path":row["path"],"status":"source_missing_or_mismatched_not_reused"})
                        continue
                    raise ValueError("Existing SCNet file identity mismatch: "+row["path"])
                if target.exists():
                    if not verified_file(target,row,deadline):raise ValueError("Existing destination differs; refusing overwrite")
                    result={"path":row["path"],"status":"reused_verified"}
                else:
                    partial=target.with_name(target.name+".copy-"+identity)
                    with source.open("rb") as inp,partial.open("xb") as out:
                        while chunk:=inp.read(4*1024*1024):
                            if time.monotonic()>=deadline:raise TimeoutError("Copy deadline; owned partial retained")
                            out.write(chunk)
                    if not verified_file(partial,row,deadline):raise ValueError("Copied bytes differ")
                    # Exclusive publication; never replace a concurrently created final file.
                    os.link(partial,target);partial.unlink()
                    result={"path":row["path"],"status":"copied_verified"}
            else:
                result=download_file(row,target,row["download_url"],deadline)
            report["files"].append(result)
            print(json.dumps({"event":"file_verified",**result}),flush=True)
        report["status"]="completed_verified_subset_reuse" if reuse_verified_only else "completed_all_files_verified"
        return 0
    except Exception as exc:
        report["error"]=f"{type(exc).__name__}: {exc}";raise
    finally:
        report["finished_utc"]=dt.datetime.now(dt.timezone.utc).isoformat()
        json_new(BASE/f"checkpoint-transfer-{identity}.json",report)
        if lockpath.is_file() and read(lockpath).get("run_id")==identity:lockpath.unlink()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("plan","download","verify","copy-existing"))
    parser.add_argument("--provider",choices=("huggingface","modelscope"),default="modelscope")
    parser.add_argument("--model-dir",type=Path,default=BASE/"models/kimi-vl")
    parser.add_argument("--source",type=Path)
    parser.add_argument("--reuse-verified-files-only",action="store_true",
                        help="copy-existing only: seed complete matching files from a failed clone, skip missing/mismatched sources")
    parser.add_argument("--max-seconds",type=int,default=1800)
    args=parser.parse_args(argv);lock,rows=source_files(TASK,args.provider)
    if args.action=="plan":
        print(json.dumps({"mode":"offline_plan","model_id":MODEL,"revision":REVISION,"files":rows,
            "weight_bytes":lock["weight_bytes"],"destination":str(args.model_dir),"downloads_performed":False,"GPU_imported":False},indent=2));return 0
    if args.action=="verify":
        result=checkpoint(TASK,args.model_dir,True);print(json.dumps(result,indent=2));return 0 if result["complete"] else 2
    if sys.platform!="linux" or not 60<=args.max_seconds<=6900:
        parser.error("Apply requires SCNet Linux and max-seconds 60..6900")
    if args.action=="copy-existing" and (not args.source or not args.source.is_absolute()):
        parser.error("copy-existing requires an absolute --source; never infer a clone path")
    if args.reuse_verified_files_only and args.action!="copy-existing":
        parser.error("--reuse-verified-files-only only applies to copy-existing")
    return apply(rows,args.model_dir,args.max_seconds,args.source if args.action=="copy-existing" else None,args.reuse_verified_files_only)


if __name__=="__main__":
    try:raise SystemExit(main())
    except (OSError,ValueError,KeyError) as exc:
        print(json.dumps({"error":str(exc),"GPU_imported":False}));raise SystemExit(2)
