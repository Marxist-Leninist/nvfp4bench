#!/usr/bin/env python3
import argparse,hashlib,json,re,shutil,subprocess
from pathlib import Path
STALL_SHIFT=41
STALL_MASK=0xF<<STALL_SHIFT
YIELD_SHIFT=45

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def section(p,name):
 out=subprocess.check_output(['readelf','-SW',str(p)],text=True,stderr=subprocess.DEVNULL)
 for line in out.splitlines():
  if f' {name} ' in line:
   q=line.split(); i=q.index(name); return int(q[i+3],16),int(q[i+4],16)
 raise RuntimeError('missing '+name)
def disasm(p,fun):
 out=subprocess.check_output(['cuobjdump','--dump-sass','--function',fun,str(p)],text=True,stderr=subprocess.DEVNULL)
 rows=[]
 for i,line in enumerate(out.splitlines()):
  m=re.search(r'/\*([0-9a-fA-F]+)\*/\s+([^/;]+)',line)
  if m: rows.append((int(m.group(1),16),m.group(2).strip()))
 return rows

def control_at(data,pos): return int.from_bytes(data[pos+8:pos+16],'little')
def with_stall(ctrl,stall): return (ctrl & ~STALL_MASK) | ((stall & 0xF)<<STALL_SHIFT)
def stall(ctrl): return (ctrl>>STALL_SHIFT)&0xF
def yld(ctrl): return (ctrl>>YIELD_SHIFT)&1
ap=argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('--stall',type=int,required=True); ap.add_argument('--function',default='sass_patch_target_v2'); ap.add_argument('--from-stall',type=int,default=15); ap.add_argument('--manifest',required=True); a=ap.parse_args()
if not 0<=a.stall<=15: raise SystemExit('stall out of range')
src=Path(a.src); dst=Path(a.dst); shutil.copy2(src,dst)
A=src.read_bytes(); D=bytearray(A); off,size=section(src,'.text.'+a.function); before=disasm(src,a.function)
patch=[]
for addr,op in before:
 if not op.startswith('OMMA.SF.SP.168128'): continue
 pos=off+addr; c=control_at(D,pos); old=stall(c)
 if old!=a.from_stall: continue
 nc=with_stall(c,a.stall)
 D[pos+8:pos+16]=nc.to_bytes(8,'little')
 patch.append({'addr':f'0x{addr:x}','old_stall':old,'new_stall':a.stall,'yield':yld(c),'old_ctrl':f'0x{c:016x}','new_ctrl':f'0x{nc:016x}'})
dst.write_bytes(D); B=bytes(D); after=disasm(dst,a.function)
# Semantic disassembly text must remain byte-for-byte equivalent at every instruction address.
if before!=after:
 diff=[(hex(x[0]),x[1],y[1] if i<len(after) else None) for i,(x,y) in enumerate(zip(before,after)) if x!=y]
 raise SystemExit('semantic disasm changed '+repr(diff[:10]))
changed=[i for i,(x,y) in enumerate(zip(A,B)) if x!=y]
if any(i<off or i>=off+size for i in changed): raise SystemExit('change outside target text')
# Verify every OMMA selected now has exact requested stall and preserved yield/non-stall bits.
for rec in patch:
 addr=int(rec['addr'],16); pos=off+addr; old=int(rec['old_ctrl'],16); new=control_at(B,pos)
 assert stall(new)==a.stall and yld(new)==yld(old)
 assert (new & ~STALL_MASK)==(old & ~STALL_MASK)
# Ensure every changed bit belongs to the stall mask of a selected instruction.
allowed=set()
for rec in patch:
 pos=off+int(rec['addr'],16)+8
 old=int(rec['old_ctrl'],16); new=int(rec['new_ctrl'],16)
 for j,(x,y) in enumerate(zip(old.to_bytes(8,'little'),new.to_bytes(8,'little'))):
  if x!=y: allowed.add(pos+j)
if set(changed)!=allowed: raise SystemExit('unexpected changed bytes')
manifest={'schema':'agillm.sm121.exact-stall-patch.v2','function':a.function,'source':str(src),'output':str(dst),'source_sha256':sha(src),'output_sha256':sha(dst),'bytes':len(B),'from_stall':a.from_stall,'target_stall':a.stall,'patched_omma_count':len(patch),'patches':patch,'proof':{'semantic_disasm_identical':True,'all_changes_inside_target_text':True,'only_stall_bits_changed':True,'yield_preserved':True,'all_other_control_bits_preserved':True,'file_size_identical':len(A)==len(B),'changed_byte_count':len(changed)}}
Path(a.manifest).write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'ok':True,'target_stall':a.stall,'patched':len(patch),'changed_bytes':len(changed),'sha256':manifest['output_sha256']}))
