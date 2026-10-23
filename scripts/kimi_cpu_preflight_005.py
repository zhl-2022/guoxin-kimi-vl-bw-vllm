#!/usr/bin/env python3
"""Hidden-device CPU-only Kimi config/tokenizer/processor preflight.

This helper never opens checkpoint weights or imports publisher model classes.
Device discovery is confined to this hidden-device child, never the live server.
Its result is never a GPU/accuracy certification.
"""
from __future__ import annotations

import argparse
import builtins
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import traceback
import zipfile

REVISION = "398eede0903cd983a2bfa0cc634e9ac1d843f375"
DENIED_ROOTS = {"vllm", "tensorflow", "jax", "cupy"}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def forbidden(name):
    # Transformers' lazy AutoProcessor/GenerationMixin imports generic model
    # definitions. Importing a class is not instantiation or weight loading.
    return name.split(".")[0] in DENIED_ROOTS or "modeling_kimi_vl" in name.split(".")


def stage(task, model, output):
    """Copy only hash-locked source/config bytes to a fresh isolated directory."""
    task, model, output = Path(task), Path(model), Path(output)
    lock = json.loads((task / "model-lock.json").read_text())
    upstream = json.loads((task / "upstream-lock.json").read_text())
    review = json.loads((task / "source-review-approved.json").read_text())
    if any(x.get("revision") != REVISION for x in (lock, upstream, review)) or review.get("static_review_approved") is not True:
        raise ValueError("Fixed model/source review identity differs")
    required = {r["path"]: r for r in lock["files"] if r["kind"] != "weight"}
    approved = {r["path"]: r["sha256"] for r in upstream["files"] if r["path"].endswith(".py")}
    if approved != review["reviewed_python_sha256"] or len(approved) != 5:
        raise ValueError("Exact five reviewed publisher Python files required")
    archive_raw = (task / upstream["archive_file"]).read_bytes()
    if digest(archive_raw) != upstream["archive_sha256"]:
        raise ValueError("Fixed publisher archive SHA differs")
    output.mkdir(parents=True, exist_ok=False)
    isolated = output / "checkpoint-config-only"
    isolated.mkdir()
    records = []
    with zipfile.ZipFile(task / upstream["archive_file"]) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or archive.testzip() is not None:
            raise ValueError("Publisher archive duplicate/CRC failure")
        for name, row in required.items():
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError("Only checkpoint root source/config paths allowed")
            source = model / name
            if source.is_file() and not source.is_symlink():
                raw = source.read_bytes(); origin = str(source.resolve())
            elif name in archive.namelist():
                raw = archive.read(name); origin = "fixed_publisher_source_archive"
            else:
                records.append({"path": name, "available": False})
                continue
            if len(raw) != row["size"] or digest(raw) != row["sha256"]:
                raise ValueError("Source/config identity differs: " + name)
            if name.endswith(".py") and approved.get(name) != digest(raw):
                raise ValueError("Publisher Python is outside approved exact set")
            (isolated / name).write_bytes(raw)
            records.append({"path": name, "available": True, "origin": origin, "bytes": len(raw), "sha256": digest(raw)})
    if not all((isolated / name).is_file() for name in approved):
        raise ValueError("Not all five approved source files are available")
    return isolated, records


def restrict_imports(attempts, weight_attempts=None):
    if any(forbidden(name) for name in sys.modules) or "torch" in sys.modules:
        raise RuntimeError("Forbidden module already imported before preflight")
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if forbidden(name):
            attempts.append(name)
            raise ImportError("CPU processor preflight refuses model/runtime import: " + name)
        return original(name, *args, **kwargs)

    builtins.__import__ = guarded
    # A second guard covers importlib.import_module, including dynamic HF code.
    class Finder:
        def find_spec(self, fullname, path=None, target=None):
            if forbidden(fullname):
                attempts.append(fullname)
                raise ImportError("CPU processor preflight refuses dynamic model/runtime import: " + fullname)
    sys.meta_path.insert(0, Finder())
    def audit(event, args):
        if event == "open":
            name = args[0]
            if isinstance(name, (str, bytes, os.PathLike)):
                text = os.fsdecode(name)
                if Path(text).suffix.lower() in {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx"}:
                    if weight_attempts is not None:
                        weight_attempts.append(text)
                    raise RuntimeError("CPU preflight refuses checkpoint weight open: " + text)
        if event in {"socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.system"}:
            raise RuntimeError("Strict offline preflight refuses " + event)
    sys.addaudithook(audit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.output_dir.absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        parser.error("Output must be a NEW nonsymlink directory")
    result = {"schema": "kimi-hidden-device-cpu-preflight-005-v1", "started_utc": datetime.now(timezone.utc).isoformat(),
              "revision": REVISION, "script_sha256": digest(Path(__file__).read_bytes()),
              "weight_files_opened": False, "GPU_allocation_requested": False, "device_discovery_requested_in_hidden_child": True,
              "AutoProcessor": {"executed": False},
              "official_accuracy_verified": False, "official_performance_verified": False, "GPU_verified": False,
              "forbidden_import_attempts": [], "forbidden_weight_open_attempts": [], "modules": {}, "status": "failed"}
    try:
        isolated, records = stage(args.task_dir, args.model_dir, out)
        result["source_files"] = records
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[name] = "1"
        for name in ("USE_TF", "USE_FLAX"):
            os.environ[name] = "0"
        os.environ["USE_TORCH"] = "1"
        for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            os.environ[name] = "-1"
        result["child_visibility"] = {n: os.environ[n] for n in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")}
        os.environ["HF_MODULES_CACHE"] = str(out / "isolated-hf-modules")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        restrict_imports(result["forbidden_import_attempts"], result["forbidden_weight_open_attempts"])
        if hasattr(signal, "SIGALRM"):
            def expired(signum, frame):
                raise TimeoutError("CPU preflight 55-second internal deadline")
            signal.signal(signal.SIGALRM, expired); signal.alarm(55)
        import torch
        result["torch_version"] = str(torch.__version__)
        result["device_count"] = torch.cuda.device_count()
        result["cuda_initialized_before"] = torch.cuda.is_initialized()
        if result["device_count"] != 0 or result["cuda_initialized_before"]:
            raise RuntimeError("Hidden-device CPU precondition failed; processor was not loaded")
        result["transformers_version"] = importlib.metadata.version("transformers")
        from transformers import AutoConfig, AutoTokenizer, AutoProcessor
        config = AutoConfig.from_pretrained(str(isolated), trust_remote_code=True, local_files_only=True)
        result["AutoConfig"] = {"executed": True, "class": type(config).__name__, "model_type": config.model_type}
        objects = {"config": config}
        if not (isolated / "tiktoken.model").is_file():
            result["AutoTokenizer"] = {"executed": False, "reason": "Pinned tiktoken.model not downloaded yet"}
            result["status"] = "config_checked_tokenizer_and_processor_deferred"
        else:
            tokenizer = AutoTokenizer.from_pretrained(str(isolated), trust_remote_code=True, local_files_only=True)
            ids = tokenizer.encode("Kimi CPU preflight 中文", add_special_tokens=False)
            result["AutoTokenizer"] = {"executed": True, "class": type(tokenizer).__name__, "sample_token_count": len(ids), "sample_ids": ids}
            objects["tokenizer"] = tokenizer
            processor = AutoProcessor.from_pretrained(str(isolated), trust_remote_code=True, local_files_only=True)
            from PIL import Image
            batch = processor(images=[Image.new("RGB", (56, 56), (255, 0, 0))],
                              text=["<|media_pad|>Describe this image."], return_tensors="pt")
            tensors = {k: {"shape": list(v.shape), "device": str(v.device)}
                       for k, v in batch.items() if torch.is_tensor(v)}
            if not tensors or any(v["device"] != "cpu" for v in tensors.values()):
                raise RuntimeError("Processor outputs are not exclusively CPU tensors")
            result["AutoProcessor"] = {"executed": True, "class": type(processor).__name__, "tensor_outputs": tensors}
            objects["processor"] = processor
            objects["image_processor"] = processor.image_processor
            result["status"] = "CPU_config_tokenizer_processor_checked"
        for name, value in objects.items():
            path = Path(inspect.getfile(type(value)))
            result["modules"][name] = {"module_file": str(path), "sha256": digest(path.read_bytes())}
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        if "torch" in sys.modules:
            result["cuda_initialized_after"] = sys.modules["torch"].cuda.is_initialized()
            if result["cuda_initialized_after"]:
                result["status"] = "failed_device_initialized"
        result["ended_utc"] = datetime.now(timezone.utc).isoformat()
        result["forbidden_modules_loaded"] = [n for n in sys.modules if forbidden(n)]
        if result["forbidden_modules_loaded"]:
            result["status"] = "failed_forbidden_module_loaded"
        if out.is_dir():
            with (out / "result.json").open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "CPU_config_tokenizer_processor_checked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
