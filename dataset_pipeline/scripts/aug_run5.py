import os, json, glob, collections, math
import cv2, numpy as np
import sys; sys.path.insert(0,"/Users/bruceherwig/Claude_Code_Projects")
from modules.io_safe import robust_imread, robust_imwrite
from tools.masks_to_labelme import mask_to_shapes
ROOT="/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images"
OUT="/Volumes/T7 Shield/AI Projects/Star Trail CleanR/bridge_fix_tiles_2026_06/aug_run5_test"
os.makedirs(OUT,exist_ok=True)
SZ=640; HALF=SZ//2
TILTS=[0,15,30,45,60,75]
OFFS=[(-200,-200),(0,-200),(200,-200),(-200,0),(0,0),(200,0),(-200,200),(0,200),(200,200)]
EXTS=('.jpg','.jpeg','.JPG','.tif','.tiff','.TIF','.png')
# proven gather (same as build_bridge_fix_tiles)
recs={}
for lp in glob.glob(os.path.join(ROOT,"*","cleanr_workspace","run_log_*.jsonl")):
    ds=lp.split("/star trail images/")[1].split("/")[0]
    for line in open(lp,errors="ignore"):
        try: d=json.loads(line)
        except: continue
        if d.get("type")!="detect": continue
        fr=d.get("frame")
        for st in (d.get("detect_stages") or []):
            if isinstance(st,dict) and st.get("stage")=="seam_second_pass":
                for ev in (st.get("events",[]) or []):
                    if ev.get("reason")=="bridge_gap_miss":
                        k=(ds,fr,ev.get("cx"),ev.get("cy"))
                        if k not in recs: recs[k]={"ds":ds,"frame":fr,"cx":ev["cx"],"cy":ev["cy"]}
recs=list(recs.values())
print("bridge trails to augment:",len(recs),flush=True)
def find_src(ds,fr):
    for e in EXTS:
        p=os.path.join(ROOT,ds,fr+e)
        if os.path.exists(p): return p
    g=glob.glob(os.path.join(ROOT,ds,"*",fr+".*")); return g[0] if g else None
def comp_at(mask,cx,cy):
    n,lab=cv2.connectedComponents((mask>0).astype(np.uint8))
    H,W=mask.shape; cid=lab[min(cy,H-1),min(cx,W-1)]
    if cid==0:
        ys,xs=np.where(mask>0)
        if len(xs)==0: return None
        i=np.argmin((xs-cx)**2+(ys-cy)**2); cid=lab[ys[i],xs[i]]
    return (lab==cid).astype(np.uint8)*255
mask_cache={}
made=0; short_n=0; long_n=0; skipped=0
for ri_idx,rec in enumerate(recs[:5]):
    ds,fr,cx,cy=rec["ds"],rec["frame"],rec["cx"],rec["cy"]
    src=find_src(ds,fr)
    if not src: skipped+=1; continue
    img=robust_imread(src,cv2.IMREAD_COLOR)
    mp=os.path.join(ROOT,ds,"cleanr_workspace","masks",fr+".png")
    if mp not in mask_cache: mask_cache[mp]=cv2.imread(mp,0)
    fm=mask_cache[mp]
    if img is None or fm is None or fm.shape[:2]!=img.shape[:2]: skipped+=1; continue
    comp=comp_at(fm,cx,cy)
    if comp is None or comp.max()==0: skipped+=1; continue
    H,W=img.shape[:2]
    ys0,xs0=np.where(comp>0); ext=max(xs0.max()-xs0.min(), ys0.max()-ys0.min())
    is_long = ext > SZ-60
    out_ds=os.path.join(OUT,ds); os.makedirs(out_ds,exist_ok=True)
    ones=np.ones((H,W),np.uint8)
    for th in TILTS:
        if th:
            M=cv2.getRotationMatrix2D((W/2,H/2),th,1.0)
            ri=cv2.warpAffine(img,M,(W,H),flags=cv2.INTER_LINEAR)
            rc=cv2.warpAffine(comp,M,(W,H),flags=cv2.INTER_NEAREST)
            valid=cv2.warpAffine(ones,M,(W,H),flags=cv2.INTER_NEAREST)
        else:
            ri,rc,valid=img,comp,ones
        ys,xs=np.where(rc>0)
        if len(xs)==0: continue
        ccx,ccy=int(xs.mean()),int(ys.mean())
        if is_long:
            pts=np.column_stack([xs,ys]).astype(np.float32); m=pts.mean(0)
            _,_,vt=np.linalg.svd(pts-m); axis=vt[0]; t=(pts-m)@axis
            lo,hi=np.quantile(t,0.02),np.quantile(t,0.98)
            nseg=int(np.clip((hi-lo)//256,1,8))
            samples=[m+axis*(lo+(hi-lo)*i/max(nseg,1)) for i in range(nseg+1)]
            positions=[(int(p[0]),int(p[1])) for p in samples]
        else:
            positions=[(ccx+ (-dx), ccy+(-dy)) for (dx,dy) in OFFS]  # trail offset (dx,dy) in tile
        for (tcx,tcy) in positions:
            x=int(tcx-HALF); y=int(tcy-HALF)
            if x<0 or y<0 or x+SZ>W or y+SZ>H: continue
            if not valid[y:y+SZ,x:x+SZ].all(): continue   # real pixels only
            tm=rc[y:y+SZ,x:x+SZ]
            if tm.max()==0: continue
            shapes=mask_to_shapes(tm)
            if not shapes: continue
            ti=ri[y:y+SZ,x:x+SZ]
            base=f"{fr}_{cx}_{cy}_t{th}_x{x}_y{y}"
            robust_imwrite(os.path.join(out_ds,base+".jpg"),ti)
            json.dump({"version":"5.0.1","flags":{},"shapes":shapes,"imagePath":base+".jpg",
                       "imageData":None,"imageHeight":SZ,"imageWidth":SZ},open(os.path.join(out_ds,base+".json"),"w"))
            made+=1
            if is_long: long_n+=1
            else: short_n+=1
    if (ri_idx+1)%15==0: print(f"  {ri_idx+1}/{len(recs)} trails, {made} variants so far",flush=True)
print(f"DONE: {made} variants ({short_n} from short, {long_n} from long), {skipped} trails skipped",flush=True)
print("OUT:",OUT,flush=True)
