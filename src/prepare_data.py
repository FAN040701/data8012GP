"""Geometry-checked AMOS preprocessing and paired native-z degradation."""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

ROOT=Path(__file__).resolve().parents[1]
LABELS={1:1,2:2,3:3,6:4}
NAMES=['background','spleen','right_kidney','left_kidney','liver']
SPACING=3.0


def remap_labels(label):
    out=np.zeros(label.shape,dtype=np.uint8)
    for old,new in LABELS.items():
        out[label==old]=new
    return out


def resample(data,source_affine,target_affine,target_shape,order,cval):
    mapping=np.linalg.inv(source_affine)@target_affine
    return ndi.affine_transform(data,mapping[:3,:3],mapping[:3,3],output_shape=target_shape,order=order,mode='constant',cval=cval,prefilter=False)


def atomic_npy(path,array):
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('wb') as f: np.save(f,array)
    tmp.replace(path)


def prepare_one(item):
    split,member=item
    ip=ROOT/'data/raw'/member
    lp=ROOT/'data/raw'/member.replace('images','labels')
    case=ip.name.removesuffix('.nii.gz')
    dest=ROOT/'data/processed'/case
    factors=[1,2,3] if split in ['train','test'] else [1]
    oldmeta=json.loads((dest/'metadata.json').read_text()) if (dest/'metadata.json').exists() else {}
    if oldmeta.get('preprocessing_version')==3 and all((dest/f'image_f{f}.npy').exists() for f in factors):
        return oldmeta
    raw_i=nib.load(ip)
    raw_l=nib.load(lp)
    if raw_i.shape != raw_l.shape or not np.allclose(raw_i.affine,raw_l.affine,atol=1e-4):
        raise ValueError(f'Image/label geometry mismatch: {case}')
    if len(raw_i.shape)!=3:
        raise ValueError(f'Non-volume image: {case}')
    img=nib.as_closest_canonical(raw_i)
    lab=nib.as_closest_canonical(raw_l)
    affine=img.affine.copy()
    if not np.allclose(affine[:3,:3],np.diag(np.diag(affine[:3,:3])),atol=1e-4):
        raise ValueError(f'Oblique CT needs separate handling: {case}')
    spacing=np.array(nib.affines.voxel_sizes(affine))
    ct=np.asarray(img.dataobj,dtype=np.float32)
    original_label=np.asarray(lab.dataobj,dtype=np.uint8)
    if not np.isfinite(ct).all() or original_label.max()>15:
        raise ValueError(f'Invalid values: {case}')
    target_shape=tuple((np.ceil((np.array(ct.shape)-1)*spacing/SPACING).astype(int)+1).tolist())
    target_affine=np.diag([SPACING,SPACING,SPACING,1.0])
    target_affine[:3,3]=affine[:3,3]
    label=remap_labels(original_label)
    y=resample(label,affine,target_affine,target_shape,0,0).transpose(2,1,0).copy()
    dest.mkdir(parents=True,exist_ok=True)
    atomic_npy(dest/'label.npy',y)
    centers={}
    rng=np.random.default_rng(20261114+int(case.split('_')[1]))
    for k in range(1,5):
        coords=np.argwhere(y==k)
        if len(coords)>4096: coords=coords[rng.choice(len(coords),4096,replace=False)]
        centers[str(k)]=coords.astype(np.int16)
    np.savez(dest/'foreground.npz',**centers)
    for factor in factors:
        path=dest/f'image_f{factor}.npy'
        if path.exists() and (factor==1 or oldmeta.get('preprocessing_version')==3): continue
        if factor==1:
            degraded=ct
            degraded_affine=affine
        else:
            sigma=0.5*np.sqrt(factor*factor-1)
            degraded=ndi.gaussian_filter1d(ct,sigma=sigma,axis=2,mode='nearest',truncate=3.0)[:,:,::factor]
            # Extend the last retained plane through the scan end. This avoids
            # introducing an artificial air slab solely from decimation phase.
            degraded=np.concatenate([degraded,degraded[:,:,-1:]],axis=2)
            degraded_affine=affine.copy()
            degraded_affine[:3,2]*=factor
        common=resample(degraded,degraded_affine,target_affine,target_shape,1,-1000)
        common=(np.clip(common,-200,300)+200)/500
        common=common.transpose(2,1,0).astype(np.float16)
        atomic_npy(path,common)
    # Reference down/up sampling provides a representation diagnostic, not a model.
    roundtrip=resample(y.transpose(2,1,0),target_affine,affine,ct.shape,0,0)
    roundtrip_dice={}
    for k,name in enumerate(NAMES[1:],1):
        a=roundtrip==k; b=label==k
        roundtrip_dice[name]=float(2*np.count_nonzero(a&b)/(np.count_nonzero(a)+np.count_nonzero(b))) if b.any() else None
    info={'case':case,'split':split,'preprocessing_version':3,'raw_image':str(ip.relative_to(ROOT)),'raw_label':str(lp.relative_to(ROOT)),
          'original_shape_xyz':list(ct.shape),'original_affine':affine.tolist(),'original_spacing_xyz_mm':spacing.tolist(),
          'original_orientation':''.join(nib.aff2axcodes(raw_i.affine)),'canonical_orientation':'RAS',
          'model_shape_dhw':list(y.shape),'model_affine_xyz':target_affine.tolist(),'model_spacing_dhw_mm':[SPACING]*3,
          'factors':factors,'effective_z_sampling_mm':{str(f):float(spacing[2]*f) for f in factors},
          'original_labels_present':np.unique(original_label).tolist(),
          'foreground_voxels_native':{name:int(np.count_nonzero(label==k)) for k,name in enumerate(NAMES[1:],1)},
          'reference_roundtrip_dice':roundtrip_dice,
          'image_sha256':hashlib.sha256(ip.read_bytes()).hexdigest(),
          'label_sha256':hashlib.sha256(lp.read_bytes()).hexdigest()}
    (dest/'metadata.json').write_text(json.dumps(info,indent=2))
    return info


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--watch',action='store_true')
    args=ap.parse_args()
    manifest=json.loads((ROOT/'data/metadata/split.json').read_text())
    all_items=[(s,n) for s,ns in manifest['members'].items() for n in ns]
    completed={}
    with cf.ProcessPoolExecutor(args.workers) as pool:
        pending={}
        submitted=set()
        while len(completed)<len(all_items):
            for item in all_items:
                s,n=item
                if n in submitted: continue
                if (ROOT/'data/raw'/n).exists() and (ROOT/'data/raw'/n.replace('images','labels')).exists():
                    pending[pool.submit(prepare_one,item)]=n
                    submitted.add(n)
            ready=[f for f in pending if f.done()]
            for fut in ready:
                n=pending.pop(fut)
                info=fut.result()
                completed[n]=info
                print(f"Prepared {len(completed)}/{len(all_items)} {info['case']} {info['split']} {info['model_shape_dhw']} z={info['original_spacing_xyz_mm'][2]:.3f}",flush=True)
                (ROOT/'data/metadata/preprocessing_audit.json').write_text(json.dumps(list(completed.values()),indent=2))
            if not args.watch and not pending: break
            if not ready: time.sleep(2)
    if len(completed)==len(all_items):
        imgs=[v['image_sha256'] for v in completed.values()]
        if len(imgs)!=len(set(imgs)): raise ValueError('Duplicate image payload detected across selected patients')
        (ROOT/'data/metadata/preprocessing_complete.json').write_text(json.dumps({'cases':len(completed),'all_image_label_geometry_matched':True,'duplicate_image_hashes':False},indent=2))
    print('Preprocessing complete',len(completed),flush=True)


if __name__=='__main__': main()
