#!/usr/bin/env python3
"""Static-only SM121 packed-NVFP4 matrix-shape legality sweep. Never launches CUDA."""
from pathlib import Path
from collections import Counter
import json,re,subprocess
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=Path('/workspace/gb10_2pflop_openfork_20260916/static_recon/legal_shape_sweep')
BASE=Path('/workspace/gb10_2pflop_openfork_20260916/static_recon/shape_matrix')
ptxas=subprocess.check_output(['bash','-lc','command -v ptxas'],text=True).strip()
raw=subprocess.check_output(['strings',ptxas],text=True,errors='ignore')
shapes=sorted(set(re.findall(r'm\d+n\d+k\d+',raw)) | {
'm8n8k32','m8n8k64','m8n8k128','m8n8k256','m16n8k32','m16n8k64','m16n8k128','m16n8k256',
'm16n16k32','m16n16k64','m16n16k128','m16n16k256','m32n8k32','m32n8k64','m32n8k128','m32n8k256',
'm32n16k64','m32n16k128','m64n8k64','m64n8k128'}, key=lambda x:tuple(map(int,re.findall(r'\d+',x))))
OUT.mkdir(parents=True,exist_ok=True)
results=[]
for family,base_name,orig in [('dense','dense_k64.ptx','m16n8k64'),('sparse','sparse_k128.ptx','m16n8k128')]:
    base=(BASE/base_name).read_text()
    for shape in shapes:
        p=OUT/f'{family}_{shape}.ptx'; p.write_text(base.replace(orig,shape))
        cub=OUT/f'{family}_{shape}.cubin'
        cp=subprocess.run([ptxas,'-arch=sm_121a',str(p),'-o',str(cub)],capture_output=True,text=True)
        err=' '.join(((cp.stderr or '')+(cp.stdout or '')).strip().split())
        low=err.lower()
        status='ASSEMBLED' if cp.returncode==0 else ('ILLEGAL_SHAPE' if 'illegal matrix shape' in low else ('UNKNOWN_SHAPE' if 'unknown modifier' in low else ('OPERAND_MISMATCH_ONLY' if 'argument vector size mismatch' in low else 'OTHER_REJECT')))
        results.append(dict(family=family,shape=shape,status=status,rc=cp.returncode,error=err[:500]))
(OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print('accepted',[(r['family'],r['shape']) for r in results if r['status']=='ASSEMBLED'])
for fam in ('dense','sparse'): print(fam,dict(Counter(r['status'] for r in results if r['family']==fam)))
