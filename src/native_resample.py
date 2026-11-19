"""Memory-bounded GPU implementation of the same physical trilinear inverse map."""
import numpy as np
import torch
from torch.nn import functional as F


@torch.inference_mode()
def restore_on_gpu(prob,meta,device,chunk_depth=16,return_prob=False):
    shape=tuple(meta['original_shape_xyz'])
    mapping=np.linalg.inv(np.asarray(meta['model_affine_xyz']))@np.asarray(meta['original_affine'])
    if not np.allclose(mapping[:3,:3],np.diag(np.diag(mapping[:3,:3])),atol=1e-8):
        raise ValueError('GPU inverse sampler requires the checked canonical axis-aligned grids')
    srcshape_xyz=np.array(prob.shape[1:][::-1])
    coordinates=[np.arange(shape[i],dtype=np.float64)*mapping[i,i]+mapping[i,3] for i in range(3)]
    valid=[(v>=0)&(v<=srcshape_xyz[i]-1) for i,v in enumerate(coordinates)]
    axes=[torch.as_tensor(2*v/(srcshape_xyz[i]-1)-1,dtype=torch.float32,device=device) for i,v in enumerate(coordinates)]
    x=axes[0][None,None,:]; y=axes[1][None,:,None]
    source=torch.as_tensor(np.ascontiguousarray(prob),dtype=torch.float32,device=device)[None]
    output=np.empty(shape,dtype=np.uint8)
    all_prob=np.empty((5,*shape),dtype=np.float32) if return_prob else None
    for start in range(0,shape[2],chunk_depth):
        stop=min(start+chunk_depth,shape[2]); depth=stop-start
        z=axes[2][start:stop,None,None]
        grid=torch.stack([x.expand(depth,shape[1],shape[0]),y.expand(depth,shape[1],shape[0]),z.expand(depth,shape[1],shape[0])],dim=-1)[None]
        p=F.grid_sample(source,grid,mode='bilinear',padding_mode='zeros',align_corners=True)[0]
        # scipy mode='constant' uses an all-zero result outside any source axis.
        ok=np.logical_and(valid[2][start:stop,None,None],np.logical_and(valid[1][None,:,None],valid[0][None,None,:]))
        if not ok.all(): p*=torch.as_tensor(ok,device=device)[None]
        output[:,:,start:stop]=p.argmax(0).permute(2,1,0).byte().cpu().numpy()
        if return_prob: all_prob[:,:,:,start:stop]=p.permute(0,3,2,1).cpu().numpy()
    return (output,all_prob) if return_prob else output
