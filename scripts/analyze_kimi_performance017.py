"""Audit profile005's declared token policy using the strict native-result auditor."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import sys

ORIGINAL='ff8eef852e19f336e99884cc031353627ca304c6a7534c22375116758a850000'
WRAPPER='eab500bbc9bd30208a6020d3dae8cef2df5a247e1391066c5303025fd6cef83a'
COMPAT='8dc94af68fe8e5d2690c14beedb3ccba394bd62b88f2744e0520cf13635a3c59'
BODY={'temperature':0,'ignore_eos':True,'stream_options':{'include_usage':True},
      'bad_words':['<|media_pad|>','<|placeholder|>','<|media_start|>','<|media_end|>']}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--analysis-dir',type=Path,required=True)
    parser.add_argument('--base-analyzer',type=Path,required=True)
    parser.add_argument('--reference-wrapper',type=Path,required=True)
    args=parser.parse_args();root=args.run.parent;attempt=root.parent
    start=json.loads((attempt/'run-start.json').read_text())
    assert start['wrapper_sha256']==WRAPPER and start['compat_source_sha256']==COMPAT
    assert start['accuracy_score']==861
    assert start['driver_sha256']=='11985f3c844f03b320f674309ea19060e1bc2f1ac93619c031092d574af22dcc'
    assert start['synthetic_output_bad_words']==BODY['bad_words']
    assert sha(attempt/'client-processor-compat/sitecustomize.py')==COMPAT
    assert sha(args.reference_wrapper)==ORIGINAL
    actual=root/'sources/t02_sglang_bench.py'
    assert sha(actual)==WRAPPER
    original_body='{"temperature":0,"ignore_eos":true,"stream_options":{"include_usage":true}}'
    expected_body=json.dumps(BODY,separators=(',',':'))
    original=args.reference_wrapper.read_bytes()
    assert original.count(original_body.encode())==1
    assert original.replace(original_body.encode(),expected_body.encode())==actual.read_bytes()
    spec=importlib.util.spec_from_file_location('strict_native017',args.base_analyzer)
    auditor=importlib.util.module_from_spec(spec);sys.modules[spec.name]=auditor;spec.loader.exec_module(auditor)
    assert auditor.APPROVED_DRIVER_SHA256==ORIGINAL
    auditor.APPROVED_DRIVER_SHA256=WRAPPER
    contract=auditor.row_contract
    def profile_contract(run,row):
        view=copy.deepcopy(row);command=view.get('command',[])
        try:
            index=command.index('--extra-request-body')+1
            if json.loads(command[index])!=BODY:
                name,issues=contract(run,row);return name,issues+['Profile005 token-suppression request body differs']
            command[index]=original_body
        except (ValueError,IndexError,TypeError,json.JSONDecodeError):
            name,issues=contract(run,row);return name,issues+['Profile005 body absent or invalid']
        return contract(run,view)
    auditor.row_contract=profile_contract
    report=auditor.analyze(args.run,args.analysis_dir)
    proof={'profile':'Kimi performance005','entry_sha256':sha(__file__),
           'base_analyzer_sha256':sha(args.base_analyzer),'original_wrapper_sha256':ORIGINAL,
           'executed_wrapper_sha256':WRAPPER,'client_compat_source_sha256':COMPAT,
           'wrapper_delta':'Exactly one extra-request-body literal adds the four declared bad_words; no other byte change',
           'row_contract_view':'Check the four bad_words exactly, then apply the original unchanged-field contract on an in-memory view',
           'originals_modified':False,'model_or_gpu_requests':0,
           'random_text_domain':'ordinary vocabulary; session and multimodal sentinels excluded',
           'output_domain':BODY['bad_words'],'summary':report['summary'],'submitted':False}
    with (args.analysis_dir/'profile005-proof.json').open('x') as stream:
        stream.write(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof,indent=2))
    return 0 if report['summary']['full_two_round_matrix_complete'] else 1


if __name__=='__main__':raise SystemExit(main())
