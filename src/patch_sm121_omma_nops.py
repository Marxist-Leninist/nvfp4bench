#!/usr/bin/env python3
import argparse,re,subprocess,shutil
from pathlib import Path

def section_offset(cubin, section):
    s=subprocess.check_output(['readelf','-SW',str(cubin)],text=True,stderr=subprocess.DEVNULL)
    for line in s.splitlines():
        if section in line:
            # [... ] name type addr offset size ...
            parts=line.split()
            i=parts.index(section)
            return int(parts[i+3],16), int(parts[i+4],16)
    raise RuntimeError('section not found '+section)

def sass(cubin, fun):
    return subprocess.check_output(['cuobjdump','--dump-sass','--function',fun,str(cubin)],text=True,stderr=subprocess.DEVNULL)

def insts(text):
    out=[]
    for line in text.splitlines():
        m=re.search(r'/\*([0-9a-fA-F]+)\*/\s+([^/;]+)',line)
        if m: out.append((int(m.group(1),16),m.group(2).strip(),line))
    return out

ap=argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('--fraction',type=float,default=1.0); ap.add_argument('--function',default='sass_patch_target'); a=ap.parse_args()
src=Path(a.src); dst=Path(a.dst); shutil.copy2(src,dst)
text=sass(src,a.function); ins=insts(text)
ommas=[x for x in ins if x[1].startswith('OMMA.SF.SP.168128')]
byaddr={x[0]:x for x in ins}
pairs=[]
for idx,o in enumerate(ommas):
    n=byaddr.get(o[0]+0x10)
    if n and n[1].startswith('NOP'):
        pairs.append((idx,o,n))
if not pairs: raise SystemExit('no OMMA->NOP slots')
sec=f'.text.{a.function}'; off,size=section_offset(src,sec)
data=bytearray(dst.read_bytes())
# Fill evenly across the eligible slots. Donor comes 8 OMMAs ahead to keep its accumulator dependency distant.
wanted=round(len(pairs)*a.fraction)
select=set()
if wanted:
    for k in range(wanted): select.add(round(k*(len(pairs)-1)/max(1,wanted-1)))
patched=[]
for pi,(idx,o,n) in enumerate(pairs):
    if pi not in select: continue
    donor=ommas[(idx+8)%len(ommas)]
    raw=data[off+donor[0]:off+donor[0]+16]
    if len(raw)!=16: raise RuntimeError('short donor')
    data[off+n[0]:off+n[0]+16]=raw
    patched.append((n[0],donor[0]))
dst.write_bytes(data)
print('function',a.function,'omma_static',len(ommas),'eligible_nops',len(pairs),'patched',len(patched),'fraction',a.fraction,'section_offset',hex(off),'section_size',hex(size))
print('patches',[(hex(x),hex(y)) for x,y in patched])
