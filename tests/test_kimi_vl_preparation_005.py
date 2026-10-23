"""CPU-only Kimi configuration, lock, trust-code and shell contracts; never GPU proof."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import kimi_vl_runtime_005 as runtime
import download_kimi_vl_005 as transfer
TASK=ROOT/"tasks/kimi-vl-bw-vllm"


class KimiCPUContractTests(unittest.TestCase):
    def config(self,extra=None):
        with patch.dict(os.environ,extra or {},clear=True):return runtime.configuration(TASK)

    def test_pilot_real_architecture_no_backend_guess(self):
        value=self.config();args=value["command"]
        self.assertEqual(value["model_id"],runtime.MODEL)
        self.assertEqual((value["tp"],value["max_len"],value["max_num_seqs"]),(1,4096,1))
        self.assertIn("--enforce-eager",args);self.assertIn("--trust-remote-code",args)
        self.assertEqual(args[args.index("--limit-mm-per-prompt")+1],'{"image":2,"video":0}')
        self.assertFalse(any("override" in arg or "backend" in arg for arg in args))
        self.assertFalse(value["GPU_verified"])

    def test_bounds_and_task_isolation(self):
        base=self.config({"MODEL_ID":"foreign","MODEL_PATH":"foreign","FRAMEWORK":"sglang","EXTRA_ARGS_JSON":"bad"})
        self.assertEqual(base["model_id"],runtime.MODEL);self.assertEqual(base["framework"],"vllm")
        for field,value in (("KIMI_MAX_RUNTIME_SECONDS","7201"),("KIMI_LOAD_TIMEOUT","7000"),
                            ("KIMI_MEM_FRAC","nan"),("KIMI_MAX_LEN","0"),("KIMI_MODEL_PATH","")):
            with self.subTest(field=field),self.assertRaises(ValueError):self.config({field:value})

    def test_relative_model_resolved_once_and_no_venv_interpreter_resolution(self):
        value=self.config({"KIMI_MODEL_PATH":"relative-checkpoint"})
        self.assertEqual(value["model_path"],str((Path.cwd()/"relative-checkpoint").resolve()))
        self.assertEqual(value["command"][0],sys.executable)

    def test_exact_dual_source_locks_and_weight_set(self):
        hf,first=transfer.source_files(TASK,"huggingface");ms,second=transfer.source_files(TASK,"modelscope")
        self.assertEqual(hf,ms);self.assertEqual(len(first),20)
        weights=[x for x in first if x["kind"]=="weight"]
        self.assertEqual(len(weights),7);self.assertEqual(sum(x["size"] for x in weights),32816098528)
        self.assertTrue(all(runtime.REVISION in x["download_url"] for x in first))
        self.assertTrue(all("/resolve/master/" not in x["download_url"] for x in second))
        self.assertEqual({(x["path"],x["sha256"],x["size"]) for x in first},
                         {(x["path"],x["sha256"],x["size"]) for x in second})

    def fixture(self,folder):
        task=folder/"kimi-vl-bw-vllm";task.mkdir();model=folder/"model";model.mkdir()
        for name in ("task.json","model-lock.json","upstream-lock.json","source-review-approved.json"):
            shutil.copyfile(TASK/name,task/name)
        lock=runtime.read(task/"model-lock.json")
        for item in lock["files"]:
            # Test fixture substitutes tiny weight/vocab bytes; never a real model.
            source=ROOT/"evidence/kimi-vl-preparation-20261023/model-small-files"/item["path"]
            data=source.read_bytes() if source.is_file() else b"CPU fake weight/vocab only"
            (model/item["path"]).write_bytes(data)
            item.update(size=len(data),sha256=hashlib.sha256(data).hexdigest())
        (task/"model-lock.json").write_text(json.dumps(lock))
        return task,model

    def test_checkpoint_hashes_remote_code_architecture_and_extra_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            task,model=self.fixture(Path(temporary))
            self.assertTrue(runtime.checkpoint(task,model,True)["complete"])
            self.assertFalse(runtime.checkpoint(task,model,False)["complete"])
            target=model/"processing_kimi_vl.py";old=target.read_bytes();target.write_bytes(old+b"\n# changed")
            self.assertFalse(runtime.checkpoint(task,model,True)["complete"]);target.write_bytes(old)
            (model/"unexpected.py").write_text("raise RuntimeError('must never import')")
            self.assertFalse(runtime.checkpoint(task,model,True)["complete"])
            (model/"unexpected.py").unlink()
            config=runtime.read(model/"config.json");config["architectures"]=["Qwen3VLForConditionalGeneration"]
            (model/"config.json").write_text(json.dumps(config))
            self.assertFalse(runtime.checkpoint(task,model,True)["complete"])

    def test_unapproved_source_never_passes_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            task,model=self.fixture(Path(temporary));review=runtime.read(task/"source-review-approved.json")
            review["static_review_approved"]=False;(task/"source-review-approved.json").write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError,"unapproved"):runtime.checkpoint(task,model,True)

    def test_dry_stop_never_calls_shared_stop(self):
        with patch.object(runtime.supervisor,"stop",side_effect=AssertionError("must not stop")),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runtime.main(["stop","--task-dir",str(TASK),"--dry-run"]),0)

    def test_actual_launch_does_not_skip_failed_identity(self):
        with patch.object(runtime.sys,"platform","linux"),patch.object(runtime,"checkpoint",return_value={"complete":False}), \
                patch.object(runtime.supervisor,"launch",side_effect=AssertionError("must not launch")),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runtime.main(["launch","--task-dir",str(TASK)]),2)

    def test_exact_frozen_config_passed_to_existing_supervisor(self):
        previous=runtime.supervisor.configuration
        def fake_launch(task,dry,run_id):
            self.assertTrue(dry);self.assertEqual(runtime.supervisor.configuration(task)["max_len"],4096)
            self.assertEqual(runtime.supervisor.configuration(task)["command"][0],sys.executable)
            return 0
        with patch.dict(os.environ,{},clear=True),patch.object(runtime.supervisor,"launch",side_effect=fake_launch):
            self.assertEqual(runtime.main(["launch","--task-dir",str(TASK),"--dry-run"]),0)
        self.assertIs(runtime.supervisor.configuration,previous)

    def test_pilot_bench_blocks_full_matrix_without_claim(self):
        out=io.StringIO()
        with patch.dict(os.environ,{},clear=True),contextlib.redirect_stdout(out):
            self.assertEqual(runtime.main(["bench","--task-dir",str(TASK)]),2)
        value=json.loads(out.getvalue());self.assertFalse(value["executed"]);self.assertFalse(value["official_performance_verified"])

    def test_destination_stays_in_new_private_namespace(self):
        with self.assertRaises(ValueError):transfer.destination_path(Path("/root/private_data/other/model"))
        with self.assertRaises(ValueError):transfer.destination_path(transfer.BASE/"models/../escape")

    def test_copy_existing_hashes_and_preserves_mismatched_final(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);source=folder/"SCNet";source.mkdir();data=b"actual local bytes"
            (source/"test.json").write_bytes(data)
            row={"path":"test.json","size":len(data),"sha256":hashlib.sha256(data).hexdigest()}
            with patch.object(transfer,"BASE",folder/"private"),contextlib.redirect_stdout(io.StringIO()):
                dest=transfer.BASE/"models/kimi-vl-test"
                self.assertEqual(transfer.apply([row],dest,60,source),0)
                self.assertEqual((dest/"test.json").read_bytes(),data)
                (dest/"test.json").write_bytes(b"different")
                with self.assertRaisesRegex(ValueError,"overwrite"):transfer.apply([row],dest,60,source)
                self.assertEqual((dest/"test.json").read_bytes(),b"different")
                self.assertEqual(len(list(transfer.BASE.glob('checkpoint-transfer-*.json'))),2)
                self.assertFalse((transfer.BASE/"download.lock").exists())

    def test_failed_clone_seed_only_copies_matching_complete_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);source=folder/"partial-clone";source.mkdir();data=b"verified piece"
            (source/"good.json").write_bytes(data);(source/"bad.json").write_bytes(b"wrong")
            rows=[{"path":name,"size":len(data),"sha256":hashlib.sha256(data).hexdigest()}
                  for name in ("good.json","bad.json","missing.json")]
            with patch.object(transfer,"BASE",folder/"private"),contextlib.redirect_stdout(io.StringIO()):
                dest=transfer.BASE/"models/kimi-vl-test"
                self.assertEqual(transfer.apply(rows,dest,60,source,True),0)
                self.assertEqual([x.name for x in dest.iterdir()],["good.json"])
                self.assertEqual((source/"bad.json").read_bytes(),b"wrong")

    @unittest.skipUnless(sys.platform=="linux" and shutil.which("bash"),"Linux Bash wrapper test")
    def test_shells_syntax_dryrun_isolated_env_and_selfcheck(self):
        for script in TASK.glob('*.sh'):
            self.assertEqual(subprocess.run(['bash','-n',str(script)],capture_output=True).returncode,0)
        env={k:v for k,v in os.environ.items() if not k.startswith(('KIMI_','OFFICIAL_'))}
        env.update(KIMI_PYTHON_BIN=sys.executable,OFFICIAL_VERIFY_ARGS_JSON='foreign',OFFICIAL_VERIFY_SCRIPT='/foreign')
        result=subprocess.run(['bash',str(TASK/'run.sh'),'--dry-run'],env=env,capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(json.loads(result.stdout)['max_len'],4096)
        check=subprocess.run([sys.executable,'-B',str(TASK/'verify.py'),'--selfcheck'],env=env,capture_output=True,text=True,timeout=15)
        self.assertEqual(check.returncode,2,check.stdout+check.stderr)
        self.assertIn('missing OFFICIAL_VERIFY_SCRIPT',check.stdout+check.stderr)
        self.assertFalse((TASK/'runtime').exists())

    @unittest.skipUnless(sys.platform=="linux" and shutil.which("bash") and shutil.which("timeout"),"Linux controller test")
    def test_pilot_propagates_failed_stop_and_keeps_raw_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);task=folder/'tasks/kimi-vl-bw-vllm';task.mkdir(parents=True)
            scripts=folder/'scripts';scripts.mkdir()
            shutil.copyfile(TASK/'pilot.sh',task/'pilot.sh')
            (task/'env.sh').write_text('export PYTHON_BIN="'+sys.executable+'"\n')
            for leaf in ('run.sh','env_check.sh','smoke.sh'):(task/leaf).write_text('#!/bin/bash\necho CPU_fixture_only\nexit 0\n')
            (task/'stop.sh').write_text('#!/bin/bash\necho stop_failed_fixture\nexit 2\n')
            for leaf in ('kimi_vl_runtime_005.py','recon2reason_runtime.py','recon2reason_smoke.py'):
                (scripts/leaf).write_text('# Hash fixture only; never executes framework\n')
            result=subprocess.run(['bash',str(task/'pilot.sh')],capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,2,result.stdout+result.stderr)
            archives=list((task/'pilot-evidence').glob('*/pilot-result.json'));self.assertEqual(len(archives),1)
            record=json.loads(archives[0].read_text());self.assertEqual(record['controller_exit_code'],2)
            self.assertEqual(record['stop_exit_code'],2);self.assertFalse(record['submission_ready'])
            self.assertIn('stop_failed_fixture',(archives[0].parent/'stop.log').read_text())


if __name__=='__main__':unittest.main()
