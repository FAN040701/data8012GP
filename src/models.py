"""Small matched-family U-Nets; repeat and adjacent inputs use the identical network."""
import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self,dim,cin,cout):
        super().__init__()
        Conv=nn.Conv3d if dim==3 else nn.Conv2d
        Norm=nn.InstanceNorm3d if dim==3 else nn.InstanceNorm2d
        self.layers=nn.Sequential(Conv(cin,cout,3,padding=1,bias=False),Norm(cout,affine=True),nn.LeakyReLU(0.01,inplace=True),
                                  Conv(cout,cout,3,padding=1,bias=False),Norm(cout,affine=True),nn.LeakyReLU(0.01,inplace=True))
    def forward(self,x): return self.layers(x)


class UNet(nn.Module):
    def __init__(self,kind,width=16,levels=5):
        super().__init__()
        self.kind=kind
        self.dim=3 if kind=='3d' else 2
        cin=3 if kind in ['repeat','2p5d','2p5d_aug'] else 1
        widths=[width*2**i for i in range(levels)]
        self.encoder=nn.ModuleList([ConvBlock(self.dim,cin if i==0 else widths[i-1],w) for i,w in enumerate(widths)])
        self.pool=(nn.MaxPool3d if self.dim==3 else nn.MaxPool2d)(2)
        self.decoder=nn.ModuleList([ConvBlock(self.dim,widths[i]+widths[i-1],widths[i-1]) for i in range(levels-1,0,-1)])
        self.head=(nn.Conv3d if self.dim==3 else nn.Conv2d)(widths[0],5,1)
    def forward(self,x):
        skips=[]
        for i,block in enumerate(self.encoder):
            x=block(self.pool(x) if i else x)
            skips.append(x)
        for block,skip in zip(self.decoder,reversed(skips[:-1])):
            x=F.interpolate(x,size=skip.shape[2:],mode='trilinear' if self.dim==3 else 'bilinear',align_corners=False)
            x=block(torch.cat([x,skip],dim=1))
        return self.head(x)


def volume_loss(logits,target):
    # logits: B,C,D,H,W for every model; target: B,D,H,W.
    ce=F.cross_entropy(logits.float(),target)
    p=logits.float().softmax(1)
    y=F.one_hot(target,num_classes=5).movedim(-1,1).float()
    axes=(0,2,3,4)
    dice=(2*(p*y).sum(axes)+1e-5)/(p.sum(axes)+y.sum(axes)+1e-5)
    return 0.5*ce+0.5*(1-dice[1:].mean())


def planar_input(halo,kind):
    # halo: (D+2),H,W, with nearest-edge replication at volume ends.
    centre=halo[1:-1]
    if kind=='2d': return centre[:,None]
    if kind=='repeat': return centre[:,None].expand(-1,3,-1,-1).contiguous()
    return torch.stack([halo[:-2],centre,halo[2:]],dim=1)
