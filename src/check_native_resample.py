"""Compare CPU and GPU resampling using synthetic geometry and a FITTING case."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage as ndi

from experiment import ROOT,PROC,init_runtime,predict_volume
from models import UNet
from evaluate import restore_probabilities,binary_metrics
from native_resample import restore_on_gpu
from prepare_data import resample


def main():
    init_runtime(11)
    device=torch.device('cuda:0')
    rng=np.random.default_rng(11)
    prob=rng.random((5,12,14,16),dtype=np.float32); prob/=prob.sum(0,keepdims=True)
    source=np.diag([3.,3.,3.,1.]); source[:3,3]=[10,-20,5]
    dest=np.diag([1.2,1.1,1.7,1.]); dest[:3,3]=source[:3,3]
    meta={'original_shape_xyz':[35,34,18],'model_affine_xyz':source.tolist(),'original_affine':dest.tolist()}
    cpu=np.stack([resample(p.transpose(2,1,0),source,dest,meta['original_shape_xyz'],1,0) for p in prob])
    gpulabel,gpu=restore_on_gpu(prob,meta,device,return_prob=True)
    max_error=float(np.max(np.abs(cpu-gpu)))
    assert max_error<5e-6
    assert np.array_equal(cpu.argmax(0),gpulabel)
    split=json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    case=split['train'][0]
    model=UNet('repeat').to(device).eval()
    checkpoint=torch.load(ROOT/'results/pilot/repeat_seed11_steps2000/best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    meta=json.loads((PROC/case/'metadata.json').read_text())
    prob=predict_volume(model,np.load(PROC/case/'image_f1.npy'),'repeat',device)
    before=time.time(); cpu_label=restore_probabilities(prob,meta); cpu_seconds=time.time()-before
    before=time.time(); gpu_label=restore_on_gpu(prob,meta,device); torch.cuda.synchronize(); gpu_seconds=time.time()-before
    disagreement=int(np.count_nonzero(cpu_label!=gpu_label))
    fraction=disagreement/cpu_label.size
    assert fraction<1e-4
    organ_dice=[]
    for k in range(1,5):
        a=cpu_label==k; b=gpu_label==k
        organ_dice.append(float(2*(a&b).sum()/(a.sum()+b.sum())))
    assert min(organ_dice)>0.9995
    result={'synthetic_probability_max_absolute_error':max_error,'synthetic_argmax_identical':True,'fitting_case':case,'test_data_used':False,'disagreeing_voxels':disagreement,'voxel_count':int(cpu_label.size),'disagreement_fraction':fraction,'cpu_gpu_prediction_agreement_dice':organ_dice,'cpu_seconds':cpu_seconds,'gpu_seconds':gpu_seconds,'same_physical_mapping':True,'decision':'Use GPU trilinear inverse map for every reported run; discard/recompute any partial CPU evaluations.'}
    (ROOT/'results/native_resample_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
