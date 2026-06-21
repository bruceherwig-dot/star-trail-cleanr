import os, json, csv, glob
import cv2, numpy as np

TILES="/Volumes/T7 Shield/AI Projects/Star Trail CleanR/bridge_fix_tiles_2026_06"
OUT=os.path.join(TILES,"aug_blind_twilight"); os.makedirs(OUT,exist_ok=True)
CSV=os.path.join(TILES,"detector_test.csv")
ANGLES=[10,20,30,40]
SZ=640; CROP=320; OFF=(SZ-CROP)//2   # 160

# blank tiles = detector found nothing
blank=[]
for row in csv.DictReader(open(CSV)):
    if int(row["n_detections"])==0:
        blank.append((row["dataset"], row["tile_image"]))
print("blank tiles to augment:", len(blank))

def rot_pts(pts, M):
    p=np.array(pts,dtype=np.float64)
    ones=np.ones((len(p),1))
    return (np.hstack([p,ones]) @ M.T)

made=0; skipped=0
for ds, img_name in blank:
    ip=os.path.join(TILES, ds, img_name)
    jp=ip[:-4]+".json"
    img=cv2.imread(ip)
    if img is None or img.shape[:2]!=(SZ,SZ): 
        skipped+=1; continue
    shapes=[]
    if os.path.exists(jp): shapes=json.load(open(jp)).get("shapes",[])
    out_ds=os.path.join(OUT, ds); os.makedirs(out_ds, exist_ok=True)
    for ang in ANGLES:
        M=cv2.getRotationMatrix2D((SZ/2,SZ/2), ang, 1.0)
        rimg=cv2.warpAffine(img, M, (SZ,SZ), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
        crop=rimg[OFF:OFF+CROP, OFF:OFF+CROP]
        crop=cv2.resize(crop,(SZ,SZ),interpolation=cv2.INTER_LINEAR)
        new_shapes=[]
        for s in shapes:
            pr=rot_pts(s["points"], M)          # rotate
            pr=(pr-np.array([OFF,OFF]))*2.0      # crop-translate + resize x2
            # keep only if some of the poly stays inside the 640 crop
            inside=pr[(pr[:,0]>=0)&(pr[:,0]<SZ)&(pr[:,1]>=0)&(pr[:,1]<SZ)]
            if len(inside)<3: continue
            pr[:,0]=np.clip(pr[:,0],0,SZ-1); pr[:,1]=np.clip(pr[:,1],0,SZ-1)
            new_shapes.append({"label":s.get("label","trail"),
                               "points":[[float(x),float(y)] for x,y in pr],
                               "group_id":None,"shape_type":"polygon","flags":{}})
        base=img_name[:-4]+f"_rot{ang}"
        cv2.imwrite(os.path.join(out_ds,base+".png"), crop, [cv2.IMWRITE_JPEG_QUALITY,95])
        json.dump({"version":"5.0.1","flags":{},"shapes":new_shapes,
                   "imagePath":base+".png","imageData":None,
                   "imageHeight":SZ,"imageWidth":SZ},
                  open(os.path.join(out_ds,base+".json"),"w"), indent=2)
        made+=1

print(f"augmented tiles written: {made}  (skipped {skipped})")
print("folder:", OUT)
