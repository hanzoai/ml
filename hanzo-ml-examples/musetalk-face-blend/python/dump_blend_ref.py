#!/usr/bin/env python3
"""Dump PyTorch BiSeNet + get_image blend reference tensors for the Rust face_parse/blend port.

For a handful of demo frames, reproduce EXACTLY the blend step of musetalk_dub_native.py and
dump every intermediate so the Rust port can be verified stage-by-stage:
  bisenet_in_{i}.npy    [1,3,512,512] f32  normalized tensor fed to the net (ImageNet mean/std)
  parsing_{i}.npy       [512,512]     i64  argmax 19-class label map (raw network output)
  jawmask_{i}.npy       [512,512]     u8   binary {0,255} jaw mask after morphology (mode='jaw')
  alpha_{i}.npy         [H,W]         u8   feathered alpha at face_large size (post blur)  *for ref*
  facelarge_{i}.npy     [H,W,3]       u8   BGR face_large crop (input region)
  rf_{i}.npy            [h,w,3]       u8   BGR resized re-rendered mouth crop
  composite_{i}.npy     [768,576,3]   u8   BGR final get_image output (the proven blend)
  meta_{i}.json         box, crop_box, ori_shape, blur_kernel
"""
import os, sys, json, glob, copy
import numpy as np, cv2, torch
from PIL import Image

os.environ.setdefault('TORCHINDUCTOR_DISABLE','1'); os.environ.setdefault('TORCHDYNAMO_DISABLE','1')
_orig=torch.load; torch.load=lambda *a,**k:(k.setdefault('weights_only',False),_orig(*a,**k))[1]
REPO=os.path.expanduser('~/work/zen-dub-run/zen-dub'); sys.path.insert(0,REPO); os.chdir(REPO)
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.blending import get_crop_box, face_seg
import face_alignment

WN=os.path.expanduser('~/work/zen-dub-run/work_native')
OUT=os.path.expanduser('~/work/zen-dub-run/blendref'); os.makedirs(OUT,exist_ok=True)
FRAMES=sorted(glob.glob(os.path.join(WN,'frames','*.png')))
EXTRA_MARGIN=10
IDXS=[0,5,10,20,40]   # demo frame indices to dump

dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fp=FaceParsing(left_cheek_width=90,right_cheek_width=90)
fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,flip_input=False,device=str(dev))

def bbox(fr):
    rgb=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB); preds=fa.get_landmarks(rgb)
    if not preds: return None
    lm=preds[0].astype(np.int32); half_y=lm[29][1]; chin_y=int(np.max(lm[:,1])); hd=chin_y-half_y
    upper=max(0,half_y-hd); x1=int(np.min(lm[:,0])); x2=int(np.max(lm[:,0])); y1=upper; y2=chin_y
    if x2-x1<=0 or y2-y1<=0 or x1<0: return None
    return (x1,y1,x2,y2)

# Instrument fp.__call__ to also capture the normalized input tensor + parsing map
import torchvision.transforms as T
prep=T.Compose([T.ToTensor(),T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

def fp_instrumented(pil_img, mode='jaw'):
    width,height=pil_img.size
    with torch.no_grad():
        im=pil_img.resize((512,512),Image.BILINEAR)
        img=prep(im)
        inp=torch.unsqueeze(img,0)
        inp_dev=inp.cuda() if torch.cuda.is_available() else inp
        out=fp.net(inp_dev)[0]
        parsing=out.squeeze(0).cpu().numpy().argmax(0)   # 512x512 int
        parsing_raw=parsing.copy()
        # mode='jaw' morphology (verbatim from FaceParsing.__call__)
        face_region=np.isin(parsing,[1])*255; face_region=face_region.astype(np.uint8)
        original_dilated=cv2.dilate(face_region,fp.kernel,iterations=1)
        eroded=cv2.erode(original_dilated,fp.cheek_kernel,iterations=2)
        face_region=cv2.bitwise_and(eroded,fp.cheek_mask)
        face_region=cv2.bitwise_or(face_region,cv2.bitwise_and(original_dilated,~fp.cheek_mask))
        parsing[(face_region==255)&(~np.isin(parsing,[10]))]=255
        parsing[np.isin(parsing,[11,12,13])]=255
        parsing[np.where(parsing!=255)]=0
    jawmask=parsing.astype(np.uint8)   # 512x512 {0,255}
    return inp.cpu().numpy().astype(np.float32), parsing_raw.astype(np.int64), jawmask

# replicate get_image, capturing the alpha + composite
def get_image_dump(image, face, face_box, idx, upper_boundary_ratio=0.5, expand=1.5):
    body=Image.fromarray(image[:,:,::-1]); facep=Image.fromarray(face[:,:,::-1])
    x,y,x1,y1=face_box
    crop_box,s=get_crop_box(face_box,expand); x_s,y_s,x_e,y_e=crop_box
    face_large=body.crop(crop_box); ori_shape=face_large.size  # (W,H)
    # --- BiSeNet on face_large ---
    inp_np,parsing_raw,jawmask=fp_instrumented(face_large,mode='jaw')
    np.save(f'{OUT}/bisenet_in_{idx}.npy',inp_np)
    np.save(f'{OUT}/parsing_{idx}.npy',parsing_raw)
    np.save(f'{OUT}/jawmask_{idx}.npy',jawmask)
    np.save(f'{OUT}/facelarge_{idx}.npy',np.array(face_large)[:,:,::-1])  # BGR
    # mask_image = jawmask resized to face_large size (face_seg does .resize(image.size))
    mask_image=Image.fromarray(jawmask).resize(ori_shape)
    mask_small=mask_image.crop((x-x_s,y-y_s,x1-x_s,y1-y_s))
    mask_image=Image.new('L',ori_shape,0)
    mask_image.paste(mask_small,(x-x_s,y-y_s,x1-x_s,y1-y_s))
    width,height=mask_image.size; top_boundary=int(height*upper_boundary_ratio)
    modified=Image.new('L',ori_shape,0)
    modified.paste(mask_image.crop((0,top_boundary,width,height)),(0,top_boundary))
    blur_kernel_size=int(0.05*ori_shape[0]//2*2)+1
    mask_array=cv2.GaussianBlur(np.array(modified),(blur_kernel_size,blur_kernel_size),0)
    np.save(f'{OUT}/alpha_{idx}.npy',mask_array.astype(np.uint8))
    mask_imgf=Image.fromarray(mask_array)
    face_large.paste(facep,(x-x_s,y-y_s,x1-x_s,y1-y_s))
    body.paste(face_large,crop_box[:2],mask_imgf)
    comp=np.array(body)[:,:,::-1]  # BGR
    json.dump({'box':[int(v) for v in face_box],'crop_box':[int(v) for v in crop_box],
               'ori_shape':[int(ori_shape[0]),int(ori_shape[1])],'blur_kernel':int(blur_kernel_size),
               's':int(s)}, open(f'{OUT}/meta_{idx}.json','w'))
    return comp

for idx in IDXS:
    fr=cv2.imread(FRAMES[idx]); box=bbox(fr)
    if box is None: print(f'frame {idx}: no face, skip'); continue
    x1,y1,x2,y2=box; y2m=min(y2+EXTRA_MARGIN,fr.shape[0])
    m=np.load(os.path.join(WN,'dubout',f'mouth_{idx:06}.npy'))  # [3,256,256] RGB [0,1]
    rgb=np.transpose(m,(1,2,0)); bgr=(rgb[...,::-1]*255).round().astype(np.uint8)
    rf=cv2.resize(bgr,(x2-x1,y2m-y1))
    np.save(f'{OUT}/rf_{idx}.npy',rf)
    comp=get_image_dump(copy.deepcopy(fr),rf,[x1,y1,x2,y2m],idx)
    np.save(f'{OUT}/composite_{idx}.npy',comp)
    cv2.imwrite(f'{OUT}/composite_{idx}.png',comp)
    print(f'frame {idx}: box={box} y2m={y2m} composite={comp.shape}')
print('done ->',OUT)
