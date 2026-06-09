#!/usr/bin/env python3
"""Hybrid MuseTalk dub: Python CV pre/post + HANZO RUST MuseTalk model inference.

Pre  (Python): frames, face bbox (face_alignment), whisper-tiny features + PositionalEncoding,
               normalized half-mask-ready face crops -> dumped per-frame as .npy.
Model(RUST):   VAE-encode + UNet + VAE-decode run by `musetalk-bench dub` (real weights),
               output decoded mouths as .npy.
Post (Python): BiSeNet feathered blend (get_image) + ffmpeg encode/mux.
"""
import os, sys, glob, copy, math, argparse, subprocess
import numpy as np, cv2, torch

os.environ.setdefault("TORCHINDUCTOR_DISABLE","1"); os.environ.setdefault("TORCHDYNAMO_DISABLE","1")
_orig=torch.load; torch.load=lambda *a,**k:(k.setdefault("weights_only",False),_orig(*a,**k))[1]
REPO=os.path.expanduser("~/work/zen-dub-run/zen-dub"); sys.path.insert(0,REPO); os.chdir(REPO)
from musetalk.models.unet import PositionalEncoding
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.blending import get_image
from transformers import WhisperModel
import face_alignment

RUST_BIN=os.path.expanduser("~/work/sw-perf/musetalk-bench/target/release/musetalk-bench")
WF_BIN=os.path.expanduser("~/work/sw-perf/whisper-feats-verify/target/release/whisper-feats-verify")
WDIR=os.path.expanduser("~/work/zen-dub-run/rustweights")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video",required=True); ap.add_argument("--audio",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--ffmpeg",default=os.path.expanduser("~/work/zen-dub-run/bin/ffmpeg"))
    ap.add_argument("--fps",type=int,default=25); ap.add_argument("--extra_margin",type=int,default=10)
    ap.add_argument("--batch_size",type=int,default=8)
    ap.add_argument("--workdir",default=os.path.expanduser("~/work/zen-dub-run/work_native"))
    ap.add_argument("--dtype",default="f16")
    ap.add_argument("--native_whisper",type=int,default=1,help="1=Rust whisper feats (no Python whisper)")
    args=ap.parse_args()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.workdir,exist_ok=True)
    frames_dir=os.path.join(args.workdir,"frames"); out_dir=os.path.join(args.workdir,"out")
    dubin=os.path.join(args.workdir,"dubin"); dubout=os.path.join(args.workdir,"dubout")
    for d in (frames_dir,out_dir,dubin,dubout):
        if os.path.isdir(d):
            for f in glob.glob(os.path.join(d,"*")): os.remove(f)
        os.makedirs(d,exist_ok=True)

    # PositionalEncoding + whisper (auxiliary CV models, not the MuseTalk model)
    pe=PositionalEncoding(d_model=384).to(dev)
    M=os.path.join(REPO,"models")
    if not args.native_whisper:
        ap_=AudioProcessor(feature_extractor_path=os.path.join(M,"whisper"))
        whisper=WhisperModel.from_pretrained(os.path.join(M,"whisper")).to(device=dev,dtype=torch.float32).eval()
    fp=FaceParsing(left_cheek_width=90,right_cheek_width=90)
    fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,flip_input=False,device=str(dev))

    print("[native] extracting frames...",flush=True)
    subprocess.run([args.ffmpeg,"-v","fatal","-y","-i",args.video,"-start_number","0",f"{frames_dir}/%08d.png"],check=True)
    img_list=sorted(glob.glob(os.path.join(frames_dir,"*.png"))); frames=[cv2.imread(p) for p in img_list]
    print(f"[native] {len(frames)} frames",flush=True)

    whisper_chunks=None
    if args.native_whisper:
        # video_num is derived from audio length the same way as get_whisper_chunk.
        import wave
        with wave.open(args.audio) as wf_: lib_len=wf_.getnframes()
        video_num_audio=math.floor((lib_len/16000.0)*args.fps)
        print(f"[native] RUST whisper features ({video_num_audio} frames)...",flush=True)
    else:
        print("[native] whisper features...",flush=True)
        feats,lib_len=ap_.get_audio_feature(args.audio)
        whisper_chunks=ap_.get_whisper_chunk(feats,dev,torch.float32,whisper,lib_len,fps=args.fps,
                                             audio_padding_length_left=2,audio_padding_length_right=2)
        print(f"[native] whisper_chunks {tuple(whisper_chunks.shape)}",flush=True)

    print("[native] bboxes...",flush=True)
    coords=[]; last=None
    for fr in frames:
        rgb=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB); preds=fa.get_landmarks(rgb)
        if not preds: coords.append(last if last else (0,0,0,0)); continue
        lm=preds[0].astype(np.int32); half_y=lm[29][1]; chin_y=int(np.max(lm[:,1])); hd=chin_y-half_y
        upper=max(0,half_y-hd); x1=int(np.min(lm[:,0])); x2=int(np.max(lm[:,0])); y1=upper; y2=chin_y
        if x2-x1<=0 or y2-y1<=0 or x1<0: coords.append(last if last else (0,0,0,0))
        else: last=(x1,y1,x2,y2); coords.append(last)

    # cycle to cover audio length
    frames_cycle=frames+frames[::-1]; coords_cycle=coords+coords[::-1]
    video_num=(video_num_audio if args.native_whisper else whisper_chunks.shape[0])
    transform_mean=np.array([0.5,0.5,0.5],dtype=np.float32)

    def norm_crop(fr,box):
        x1,y1,x2,y2=box; y2m=min(y2+args.extra_margin,fr.shape[0])
        crop=cv2.resize(fr[y1:y2m,x1:x2],(256,256),interpolation=cv2.INTER_LANCZOS4)
        rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        rgb=(rgb-transform_mean)/transform_mean          # mean=std=0.5
        return np.transpose(rgb,(2,0,1))[None]           # [1,3,256,256]

    print(f"[native] dumping {video_num} per-frame model inputs...",flush=True)
    for i in range(video_num):
        fr=frames_cycle[i%len(frames_cycle)]; box=coords_cycle[i%len(coords_cycle)]
        np.save(os.path.join(dubin,f"face_{i:06}.npy"), norm_crop(fr,box).astype(np.float32))
        if not args.native_whisper:
            wb=whisper_chunks[i:i+1].to(dev).to(torch.float32)
            af=pe(wb).detach().cpu().float().numpy()         # [1,50,384]
            np.save(os.path.join(dubin,f"audio_{i:06}.npy"), af)
    if args.native_whisper:
        # RUST whisper-tiny audio2feature -> audio_*.npy (post-PositionalEncoding) directly.
        wf_env=dict(os.environ, WF_AUDIO=args.audio, WF_OUTDIR=dubin, WF_FPS=str(args.fps),
                    MUSETALK_DEV="cuda", MUSETALK_DTYPE=args.dtype, MUSETALK_WDIR=WDIR)
        print("[native] >>> RUST whisper-tiny audio features <<<",flush=True)
        subprocess.run([WF_BIN,"dump"],check=True,env=wf_env)

    # ---- RUST MuseTalk model inference ----
    print("[native] >>> HANZO RUST MuseTalk model inference <<<",flush=True)
    env=dict(os.environ, MUSETALK_DEV="cuda", MUSETALK_DTYPE=args.dtype,
             MUSETALK_DUBIN=dubin, MUSETALK_DUBOUT=dubout, MUSETALK_WDIR=WDIR,
             MUSETALK_NFRAMES=str(video_num), MUSETALK_BATCH=str(args.batch_size))
    subprocess.run([RUST_BIN,"dub"],check=True,env=env)

    # ---- read Rust mouths, blend (BiSeNet), write frames ----
    print("[native] blending...",flush=True)
    for i in range(video_num):
        m=np.load(os.path.join(dubout,f"mouth_{i:06}.npy"))   # [3,256,256] RGB [0,1]
        rgb=np.transpose(m,(1,2,0)); bgr=(rgb[...,::-1]*255).round().astype(np.uint8)  # -> BGR uint8
        x1,y1,x2,y2=coords_cycle[i%len(coords_cycle)]; ori=copy.deepcopy(frames_cycle[i%len(frames_cycle)])
        y2m=min(y2+args.extra_margin,ori.shape[0])
        try: rf=cv2.resize(bgr,(x2-x1,y2m-y1))
        except Exception as e:
            cv2.imwrite(os.path.join(out_dir,f"{i:08d}.png"),ori); continue
        combined=get_image(ori,rf,[x1,y1,x2,y2m],mode="jaw",fp=fp)
        cv2.imwrite(os.path.join(out_dir,f"{i:08d}.png"),combined)

    temp=os.path.join(args.workdir,"temp.mp4")
    print("[native] encoding video...",flush=True)
    subprocess.run([args.ffmpeg,"-y","-v","warning","-r",str(args.fps),"-f","image2","-i",f"{out_dir}/%08d.png",
                    "-vcodec","libx264","-vf","format=yuv420p","-crf","18",temp],check=True)
    subprocess.run([args.ffmpeg,"-y","-v","warning","-i",temp,"-i",args.audio,"-c:v","copy","-c:a","aac","-shortest",args.out],check=True)
    print(f"[native] DONE -> {args.out}  ({video_num} frames, model=RUST {args.dtype})",flush=True)

if __name__=="__main__": main()
