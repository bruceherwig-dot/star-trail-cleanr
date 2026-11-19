import os, json, csv, glob
import cv2, numpy as np
import sys; sys.path.insert(0,"/Users/bruceherwig/Claude_Code_Projects")
from modules.io_safe import robust_imread, robust_imwrite
from tools.masks_to_labelme import mask_to_shapes

ROOT="/Volumes/T7 Shield/AI Projects/Star Trail CleanR/star trail images"
TILES="/Volumes/T7 Shield/AI Projects/Star Trail CleanR/bridge_fix_tiles_2026_06"
OUT=os.path.join(TILES,"aug_blind_source"); os.makedirs(OUT,exist_ok=True)
CSV=os.path.join(TILES,"detector_test.csv")
SZ=640; HALF=SZ//2
ANGLES=[-45,-30,-15,15,30,45]
OFFS=[(-220,-220),(0,-220),(220,-220),(-220,0),(0,0),(220,0),(-220,220),(0,220),(220,220)]
EXTS=('.jpg','.jpeg','.tif','.tiff','.png','.JPG','.JPEG','.TIF','.TIFF')

def find_source(ds,frame):
    for base in [os.path.join(ROOT,ds)]:
        for e in EXTS:
            p=os.path.join(base,frame+e)
            if os.path.exists(p): return p
        for sub in glob.glob(os.path.join(base,"*",frame+".*")):
            if sub.lower().endswith(tuple(e.lower() for e in EXTS)): return sub
    return None

# blind tiles
blind=[(r["dataset"],r["tile_image"]) for r in csv.DictReader(open(CSV)) if int(r["n_detections"])==0]
print("blind trails:",len(blind))

def parse(name):  # frame__tile_cx_cy.png
    stem=name[:-4]; p=stem.split("__"); frame=p[0]; rest=p[1].split("_")
    return frame, int(rest[-2]), int(rest[-1])

def win_ok(x,y,W,H): return 0<=x and 0<=y and x+SZ<=W and y+SZ<=H

made=0; skipped_big=0; skipped_edge=0; mask_cache={}
for ds,name in blind:
    frame,cx,cy=parse(name)
    src=find_source(ds,frame)
    if not src: continue
    img=robust_imread(src,cv2.IMREAD_COLOR)
    mp=os.path.join(ROOT,ds,"cleanr_workspace","masks",frame+".png")
    if mp not in mask_cache: mask_cache[mp]=cv2.imread(mp,cv2.IMREAD_GRAYSCALE)
    fmask=mask_cache[mp]
    if img is None or fmask is None or fmask.shape[:2]!=img.shape[:2]: continue
    H,W=img.shape[:2]
    # isolate the trail component nearest (cx,cy)
    num,lab=cv2.connectedComponents((fmask>0).astype(np.uint8))
    cid=lab[min(cy,H-1),min(cx,W-1)]
    if cid==0:
        ys,xs=np.where(fmask>0)
        if len(xs)==0: continue
        d=(xs-cx)**2+(ys-cy)**2; k=np.argmin(d); cid=lab[ys[k],xs[k]]
    comp=(lab==cid).astype(np.uint8)*255
    ys,xs=np.where(comp>0); 
    tw=xs.max()-xs.min(); th=ys.max()-ys.min()
    if max(tw,th) > SZ-40:   # bigger than a tile -> can't bury it
        skipped_big+=1; continue
    ccx=int(xs.mean()); ccy=int(ys.mean())
    out_ds=os.path.join(OUT,ds); os.makedirs(out_ds,exist_ok=True)
    # variant set: offsets at 0 deg + rotations at center
    variants=[(0,dx,dy) for (dx,dy) in OFFS]+[(a,0,0) for a in ANGLES]
    for (ang,dx,dy) in variants:
        if ang!=0:
            M=cv2.getRotationMatrix2D((ccx,ccy),ang,1.0)
            rimg=cv2.warpAffine(img,M,(W,H),flags=cv2.INTER_LINEAR,borderValue=0)
            rmask=cv2.warpAffine(comp,M,(W,H),flags=cv2.INTER_NEAREST,borderValue=0)
        else:
            rimg,rmask=img,comp
        x=ccx-HALF+dx; y=ccy-HALF+dy
        if not win_ok(x,y,W,H): skipped_edge+=1; continue
        ti=rimg[y:y+SZ,x:x+SZ]; tm=rmask[y:y+SZ,x:x+SZ]
        if tm.max()==0: skipped_edge+=1; continue
        shapes=mask_to_shapes((tm>0).astype(np.uint8)*255)
        if not shapes: continue
        base=f"{frame}_{cx}_{cy}_a{ang}_x{dx}_y{dy}"
        robust_imwrite(os.path.join(out_ds,base+".png"),ti)
        json.dump({"version":"5.0.1","flags":{},"shapes":shapes,"imagePath":base+".png",
                   "imageData":None,"imageHeight":SZ,"imageWidth":SZ},
                  open(os.path.join(out_ds,base+".json"),"w"),indent=2)
        made+=1

print(f"source-based aug tiles written: {made}")
print(f"skipped (trail bigger than a tile): {skipped_big}")
print(f"skipped variants (ran off edge / empty): {skipped_edge}")
print("folder:",OUT)
