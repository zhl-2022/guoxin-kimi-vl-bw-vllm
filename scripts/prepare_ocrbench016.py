"""Fetch the exact public parquet and reconstruct the preserved 1000-row input."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import urllib.request

REVISION='92a54bd1384387c178d5a07140a2d85e0a3d12e1'
SHA='d9cc9f488e5310143d8d89956c19d0bdfd71e7089025da5202643f73720df7f5'
URL=f'https://huggingface.co/datasets/echo840/OCRBench/resolve/{REVISION}/data/test-00000-of-00001.parquet'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists(),'Refusing to overwrite any prepared or active dataset'
    if args.parquet:blob=args.parquet.read_bytes()
    else:
        with urllib.request.urlopen(URL,timeout=120) as response:blob=response.read(70*1024**2)
    assert len(blob)==67576988 and hashlib.sha256(blob).hexdigest()==SHA
    import pyarrow.parquet as pq
    from PIL import Image
    rows=pq.ParquetFile(io.BytesIO(blob)).read().to_pylist();assert len(rows)==1000
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'images').mkdir()
    samples=[];annotations=[]
    formats={'JPEG':'.jpg','PNG':'.png','BMP':'.bmp','WEBP':'.webp','TIFF':'.tiff','GIF':'.gif'}
    for index,row in enumerate(rows):
        raw=row['image']['bytes'];image=Image.open(io.BytesIO(raw));image.load()
        relative='images/'+str(index)+formats[image.format]
        (args.output/relative).write_bytes(raw)
        samples.append({'id':str(index),'image':relative,'image_sha256':hashlib.sha256(raw).hexdigest(),
             'question':row['question'],'category':row['question_type'],'dataset':row['dataset'],
             'image_format':image.format,'width':image.width,'height':image.height})
        annotations.append({'id':str(index),'answer':row['answer'],'category':row['question_type'],'dataset':row['dataset']})
    # The original preparation ran on Windows. Preserve those exact CRLF
    # bytes on Linux too, so the recorded input hashes remain reproducible.
    values={'samples.jsonl':''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in samples),
            'annotations.json':json.dumps(annotations,ensure_ascii=False,indent=2)}
    for name,text in values.items():(args.output/name).write_bytes(text.replace('\n','\r\n').encode())
    from verify_kimi_ocr016 import checked_inputs
    checked_inputs(args.output)
    ready={'dataset':'echo840/OCRBench','revision':REVISION,'samples':1000,'source_parquet_sha256':SHA,
           'source_url':URL,'selection':'all1000; no filtering','GPU_requests':0}
    (args.output/'READY.json').write_text(json.dumps(ready,indent=2)+'\n')
    print(json.dumps(ready))


if __name__=='__main__':main()
