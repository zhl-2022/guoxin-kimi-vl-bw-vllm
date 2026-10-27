from pathlib import Path
import hashlib,json,os,runpy
runpy.run_path(os.environ['KIMI_CLIENT_MATH_HOOK'])
from transformers import AutoProcessor
_load_processor = AutoProcessor.from_pretrained
def _kimi_image_id(*args,**kwargs):
    processor = _load_processor(*args,**kwargs)
    if type(processor).__name__ == 'KimiVLProcessor':
        assert Path(args[0]).resolve() == Path(os.environ.get('KIMI_MODEL_PATH','/root/private_data/guoxin-candidates/models/kimi-vl')).resolve()
        assert processor.image_token == '<|media_pad|>'
        token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        assert processor.tokenizer.convert_ids_to_tokens(token_id) == processor.image_token
        previous = getattr(processor,'image_token_id',None)
        assert previous is None or previous == token_id
        processor.image_token_id = token_id
        original_vocab = processor.tokenizer.get_vocab()
        reserved = set(processor.tokenizer.all_special_ids)
        for token,value in original_vocab.items():
            text = token.decode(errors='replace') if isinstance(token,bytes) else str(token)
            if text.startswith('<|') and text.endswith('|>'): reserved.add(value)
        # The upstream generator removes image_token_id itself. Keep that one
        # sentinel in this intermediate view so list.remove remains valid.
        ordinary_vocab = {token:value for token,value in original_vocab.items()
                          if value not in reserved or value == token_id}
        assert token_id in reserved and ordinary_vocab
        processor.tokenizer.get_vocab = lambda:dict(ordinary_vocab)
        row = {'pid':os.getpid(),'processor':type(processor).__name__,'previous_image_token_id':previous,
               'image_token_id':token_id,'image_token':processor.image_token,
               'random_text_reserved_ids_excluded':len(reserved),'random_text_vocab_size':len(ordinary_vocab),
               'model_files_changed':False,'upstream_sglang_source_changed':False}
        with Path(os.environ['KIMI_CLIENT_COMPAT_AUDIT']).open('a') as stream:
            stream.write(json.dumps(row)+'\n')
    return processor
AutoProcessor.from_pretrained = _kimi_image_id
