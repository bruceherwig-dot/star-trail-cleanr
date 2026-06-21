import os, json, glob, collections, math
import cv2, numpy as np
import sys; sys.path.insert(0,"/Users/bruceherwig/Claude_Code_Projects")
from modules.io_safe import robust_imread
ROOT="/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images"
OUT="/Users/bruceherwig/Claude_Code_Projects/runs/aug_bridge_test"; os.makedirs(OUT,exist_ok=True)
SZ=640; HALF=SZ//2; EXTS=('.jpg','.jpeg','.JPG','.tif','.tiff','.TIF','.png')
# gather bridge records
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
print("total bridge records:",len(recs))
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
def variants_for(rec):
    ds,fr,cx,cy=rec["ds"],rec["frame"],rec["cx"],rec["cy"]
    src=find_src(ds,fr); 
    if not src: return [],None
    img=robust_imread(src,cv2.IMREAD_COLOR)
    mp=os.path.join(ROOT,ds,"cleanr_workspace","masks",fr+".png")
    if mp not in mask_cache: mask_cache[mp]=cv2.imread(mp,0)
    fm=mask_cache[mp]
    if img is None or fm is None or fm.shape[:2]!=img.shape[:2]: return [],None
    comp=comp_at(fm,cx,cy)
    if comp is None or comp.max()==0: return [],None
    ys,xs=np.where(comp>0); H,W=img.shape[:2]
    ext=max(xs.max()-xs.min(), ys.max()-ys.min())
    out=[]
    def cut(cxp,cyp,ang):
        M=cv2.getRotationMatrix2D((cxp,cyp),ang,1.0)
        ri=cv2.warpAffine(img,M,(W,H),flags=cv2.INTER_LINEAR)
        rm=cv2.warpAffine(comp,M,(W,H),flags=cv2.INTER_NEAREST)
        x=int(cxp-HALF); y=int(cyp-HALF)
        if x<0 or y<0 or x+SZ>W or y+SZ>H: return None
        tm=rm[y:y+SZ,x:x+SZ]
        if tm.max()==0: return None
        return ri[y:y+SZ,x:x+SZ], tm
    if ext <= SZ-60:   # SHORT: slide + tilt around the trail centroid
        ccx,ccy=int(xs.mean()),int(ys.mean()); kind="short"
        for dx in (-200,0,200):
            for dy in (-200,0,200):
                r=cut(ccx+dx,ccy+dy,0); 
                if r: out.append(r)
        for a in (-40,-20,20,40):
            r=cut(ccx,ccy,a)
            if r: out.append(r)
    else:              # LONG: slide ALONG the trail axis + tilt
        kind="long"
        pts=np.column_stack([xs,ys]).astype(np.float32); m=pts.mean(0)
        u,s,vt=np.linalg.svd(pts-m); axis=vt[0]
        t=(pts-m)@axis; 
        for frac in (0.2,0.4,0.6,0.8):
            pt=m+axis*np.quantile(t,frac)
            for a in (0,-20,20):
                r=cut(float(pt[0]),float(pt[1]),a)
                if r: out.append(r)
    return out,kind
# pick 3 short + 3 long for the test
shortrecs=[]; longrecs=[]
for rec in recs:
    src=find_src(rec["ds"],rec["frame"])
    if not src: continue
    # cheap extent check via mask
    mp=os.path.join(ROOT,rec["ds"],"cleanr_workspace","masks",rec["frame"]+".png")
    fm=cv2.imread(mp,0)
    if fm is None: continue
    c=comp_at(fm,rec["cx"],rec["cy"])
    if c is None or c.max()==0: continue
    ys,xs=np.where(c>0); ext=max(xs.max()-xs.min(),ys.max()-ys.min())
    (shortrecs if ext<=SZ-60 else longrecs).append(rec)
    if len(shortrecs)>=3 and len(longrecs)>=3: break
print(f"test trails: {len(shortrecs)} short, {len(longrecs)} long")
panels=[]
for rec in (shortrecs[:3]+longrecs[:3]):
    vs,kind=variants_for(rec)
    for (ti,tm) in vs[:6]:
        ov=cv2.convertScaleAbs(ti,alpha=2.0)
        cnts,_=cv2.findContours(tm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov,cnts,-1,(0,0,255),2)
        cv2.putText(ov,kind,(6,24),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        panels.append(cv2.resize(ov,(300,300)))
    print(f'{rec["frame"]} ({kind}): {len(vs)} variants')
cols=6; rows=math.ceil(len(panels)/cols); PAD=5
W=cols*300+(cols+1)*PAD; H=rows*300+(rows+1)*PAD
cv=np.full((H,W,3),25,np.uint8)
for i,p in enumerate(panels):
    r,c=divmod(i,cols); cv[PAD+r*(300+PAD):PAD+r*(300+PAD)+300, PAD+c*(300+PAD):PAD+c*(300+PAD)+300]=p
cv2.imwrite(os.path.join(OUT,"sample.jpg"),cv,[cv2.IMWRITE_JPEG_QUALITY,88])
print("SHEET:",os.path.join(OUT,"sample.jpg"))
