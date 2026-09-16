#!/usr/bin/env python3
import argparse, hashlib, json, re, subprocess
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def section(cubin,name):
    out=subprocess.check_output(['readelf','-SW',str(cubin)],text=True,stderr=subprocess.DEVNULL)
    for line in out.splitlines():
        if f' {name} ' not in line: continue
        p=line.split(); i=p.index(name)
        return int(p[i+3],16),int(p[i+4],16)
    raise RuntimeError(f'missing section {name}')

def sass_map(cubin,fun):
    out=subprocess.check_output(['cuobjdump','--dump-sass','--function',fun,str(cubin)],text=True,stderr=subprocess.DEVNULL)
    m={}
    for line in out.splitlines():
        x=re.search(r'/\*([0-9a-fA-F]+)\*/\s+([^/;]+)',line)
        if x: m[int(x.group(1),16)]=x.group(2).strip()
    return m

def b16(x): return x.hex()

ap=argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('patched'); ap.add_argument('--function',default='sass_patch_target'); ap.add_argument('--out',required=True); a=ap.parse_args()
base=Path(a.base); pat=Path(a.patched); A=base.read_bytes(); B=pat.read_bytes()
if len(A)!=len(B): raise SystemExit('size_changed')
secname=f'.text.{a.function}'; off,size=section(base,secname); off2,size2=section(pat,secname)
if (off,size)!=(off2,size2): raise SystemExit('section_layout_changed')
mb=sass_map(base,a.function); mp=sass_map(pat,a.function)
changed_bytes=[i for i,(x,y) in enumerate(zip(A,B)) if x!=y]
changed_slots=sorted(set((i-off)//16 for i in changed_bytes))
if any(i<off or i>=off+size for i in changed_bytes): raise SystemExit('byte_changed_outside_text_section')
records=[]
for slot in changed_slots:
    pos=off+slot*16; addr=slot*16
    old=A[pos:pos+16]; new=B[pos:pos+16]
    oldop=mb.get(addr,''); newop=mp.get(addr,'')
    if not oldop.startswith('NOP'): raise SystemExit(f'non_NOP_replaced@{addr:x}:{oldop}')
    if not newop.startswith('OMMA.SF.SP.168128'): raise SystemExit(f'non_OMMA_inserted@{addr:x}:{newop}')
    records.append({'section_addr':f'0x{addr:x}','file_offset':f'0x{pos:x}','old_hex_le':b16(old),'new_hex_le':b16(new),'old_sass':oldop,'new_sass':newop})
# Every untouched byte must be identical; capmerc/info/ELF metadata therefore remain byte-identical.
changed_set=set(changed_bytes)
assert all(A[i]==B[i] for i in range(len(A)) if i not in changed_set)
base_omma=sum(v.startswith('OMMA.SF.SP.168128') for v in mb.values()); pat_omma=sum(v.startswith('OMMA.SF.SP.168128') for v in mp.values())
base_nop=sum(v.startswith('NOP') for v in mb.values()); pat_nop=sum(v.startswith('NOP') for v in mp.values())
if pat_omma-base_omma != len(records): raise SystemExit('OMMA_delta_mismatch')
if base_nop-pat_nop != len(records): raise SystemExit('NOP_delta_mismatch')
# Ensure all branch/control-flow instructions outside patched slots are raw-byte-identical (already implied by whole-file diff), record counts for audit.
ctrl_prefix=('BRA','BRX','JMP','CALL','RET','EXIT','BSSY','BSYNC','WARPSYNC','BAR','UCGABAR','CGAERRBAR','ERRBAR')
ctrl_base={k:v for k,v in mb.items() if v.startswith(ctrl_prefix)}; ctrl_pat={k:v for k,v in mp.items() if v.startswith(ctrl_prefix)}
if ctrl_base!=ctrl_pat: raise SystemExit('control_flow_changed')
obj={
 'schema':'agillm.sm121.sass-nop-fill-proof.v1','function':a.function,
 'base':{'path':str(base),'bytes':len(A),'sha256':sha(base),'text_offset':f'0x{off:x}','text_size':f'0x{size:x}','omma':base_omma,'nop':base_nop},
 'patched':{'path':str(pat),'bytes':len(B),'sha256':sha(pat),'omma':pat_omma,'nop':pat_nop},
 'proof':{'changed_byte_count':len(changed_bytes),'changed_instruction_count':len(records),'all_changes_inside_target_text':True,'all_replaced_instructions_were_nop':True,'all_inserted_instructions_are_omma_sparse_168128':True,'all_nonpatched_bytes_identical':True,'control_flow_identical':True,'section_layout_identical':True,'capmerc_and_other_metadata_byte_identical':True},
 'patches':records,
}
Path(a.out).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
print(json.dumps({'ok':True,'out':a.out,'base_omma':base_omma,'patched_omma':pat_omma,'base_nop':base_nop,'patched_nop':pat_nop,'changed_instructions':len(records),'changed_bytes':len(changed_bytes),'base_sha256':obj['base']['sha256'],'patched_sha256':obj['patched']['sha256']}))
