"""Small methodological checks: geometry, physical distance and matched controls."""
import json
from pathlib import Path
import numpy as np
import torch

from models import UNet,planar_input,volume_loss
from prepare_data import resample
from evaluate import binary_metrics

ROOT=Path(__file__).resolve().parents[1]


def main():
    results={}
    a=np.zeros((12,14,16),np.uint8); a[3:7,4:8,5:9]=1
    affine=np.diag([1.2,2.3,4.5,1.0]); affine[:3,3]=[10,-20,30]
    b=resample(a,affine,affine,a.shape,0,0)
    assert np.array_equal(a,b)
    results['identity_resampling']=True
    same=binary_metrics(a.astype(bool),a.astype(bool),[1.2,2.3,4.5])
    assert same['dice']==1.0 and same['hd95_mm']==0.0
    p=np.zeros((10,10,10),bool); t=p.copy(); p[3,3,3]=1; t[3,3,4]=1
    translated=binary_metrics(p,t,[1,1,3])
    assert translated['hd95_mm']==3.0
    results['physical_hd95_translation_mm']=3.0
    empty=binary_metrics(np.zeros_like(p),t,[1,1,3])
    assert empty['dice']==0 and np.isnan(empty['hd95_mm']) and empty['status']=='prediction_empty'
    results['empty_prediction_handling']=True
    absent=binary_metrics(t,np.zeros_like(t),[1,1,3])
    assert np.isnan(absent['dice']) and absent['absent_reference_false_positive_ml']==0.003
    results['reference_absence_scored_consistently']=True
    torch.manual_seed(11); repeat=UNet('repeat',width=4)
    torch.manual_seed(11); adjacent=UNet('2p5d',width=4)
    assert all(torch.equal(a,b) for a,b in zip(repeat.parameters(),adjacent.parameters()))
    # With identical slices, both input encodings and predictions must coincide.
    centre=torch.randn(1,32,32)
    halo=centre.expand(18,-1,-1).contiguous()
    xr=planar_input(halo,'repeat'); xa=planar_input(halo,'2p5d')
    assert torch.equal(xr,xa)
    repeat.eval(); adjacent.eval()
    with torch.no_grad(): assert torch.equal(repeat(xr),adjacent(xa))
    results['matched_control_initialization_and_encoding']=True
    logits=torch.randn(1,5,4,8,8,requires_grad=True)
    target=torch.randint(0,5,(1,4,8,8))
    loss=volume_loss(logits,target); loss.backward()
    assert torch.isfinite(loss) and logits.grad.abs().sum()>0
    results['volume_loss_nonzero_finite_gradient']=True
    split=json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    sets=[set(split[k]) for k in ['train','val','test']]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3))
    results['patient_split_disjoint']=True
    (ROOT/'results/pipeline_checks.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
