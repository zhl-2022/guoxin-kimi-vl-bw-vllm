import json,os
from pathlib import Path
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
if hasattr(torch.backends.cuda,'enable_cudnn_sdp'):
    torch.backends.cuda.enable_cudnn_sdp(False)
r={'pid':os.getpid(),'torch':torch.__version__,'flash':torch.backends.cuda.flash_sdp_enabled(),
   'mem_efficient':torch.backends.cuda.mem_efficient_sdp_enabled(),'math':torch.backends.cuda.math_sdp_enabled()}
assert not r['flash'] and not r['mem_efficient'] and r['math']
Path(os.environ['KIMI_SDP_AUDIT_DIR'],str(os.getpid())+'.json').write_text(json.dumps(r))
