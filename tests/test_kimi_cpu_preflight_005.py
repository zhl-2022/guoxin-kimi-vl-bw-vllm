"""CPU-only safety/identity tests; no actual Torch/Transformers/GPU import."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/kimi_cpu_preflight_005.py"
spec = importlib.util.spec_from_file_location("kimi_cpu_preflight_005", SCRIPT)
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)


class CpuPreflightContract(unittest.TestCase):
    def test_stage_has_only_small_fixed_files_no_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path, records = probe.stage(ROOT / "tasks/kimi-vl-bw-vllm", Path(directory)/"missing-model", Path(directory)/"new")
            self.assertFalse(any(p.suffix == ".safetensors" for p in path.iterdir()))
            self.assertEqual(len(list(path.glob("*.py"))), 5)
            self.assertEqual(json.loads((path/"config.json").read_text())["architectures"], ["KimiVLForConditionalGeneration"])
            self.assertEqual([r["path"] for r in records if not r["available"]], ["model.safetensors.index.json", "tiktoken.model"])

    def test_modified_config_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)/"model"; model.mkdir()
            (model/"config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "identity differs: config.json"):
                probe.stage(ROOT / "tasks/kimi-vl-bw-vllm", model, Path(directory)/"new")

    def test_output_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/"old"; out.mkdir(); (out/"sentinel").write_text("old")
            with self.assertRaises(FileExistsError):
                probe.stage(ROOT/"tasks/kimi-vl-bw-vllm", Path(directory)/"model", out)
            self.assertEqual((out/"sentinel").read_text(), "old")

    def test_runtime_and_actual_model_imports_blocked(self):
        code = ("import runpy,importlib; m=runpy.run_path(" + repr(str(SCRIPT)) + "); a=[]; m['restrict_imports'](a); "
                "importlib.import_module('modeling_kimi_vl')")
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refuses dynamic model/runtime import", result.stderr)

    def test_restriction_classification_allows_cpu_processor_dependencies(self):
        self.assertTrue(probe.forbidden("vllm"))
        self.assertTrue(probe.forbidden("transformers_modules.local.modeling_kimi_vl"))
        self.assertFalse(probe.forbidden("transformers.models.qwen2.modeling_qwen2"))
        self.assertFalse(probe.forbidden("transformers.models.auto.modeling_auto"))
        self.assertFalse(probe.forbidden("torch"))
        self.assertFalse(probe.forbidden("torchvision.transforms"))
        self.assertFalse(probe.forbidden("amdsmi"))
        self.assertFalse(probe.forbidden("pynvml"))
        self.assertFalse(probe.forbidden("transformers.modeling_outputs"))

    def test_weight_open_is_blocked_without_loading_framework(self):
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory)/"fake.safetensors"; weight.write_bytes(b"fixture")
            code = ("import runpy; m=runpy.run_path(" + repr(str(SCRIPT)) + "); "
                    "m['restrict_imports']([]); open(" + repr(str(weight)) + ", 'rb')")
            result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refuses checkpoint weight open", result.stderr)


if __name__ == "__main__":
    unittest.main()
