import torch
from torch import nn
from torch.nn import functional as F
from .layers import Encoder,warp,image_demo


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;w=c.get('width',24);self.base=Encoder(6,w);self.base_decoder=nn.Conv2d(w,3,1)
        self.enhance=Encoder(5,w);self.enhance_decoder=nn.Conv2d(w,5,1)
    def quantize(self,z):
        step=self.c.get('step',.1);rounded=(z/step).round()*step
        return z+(rounded-z).detach() if self.training else rounded
    def forward(self,b):
        pair=b['x'].float();previous,current=pair[:,:3],pair[:,3:6];latent=self.quantize(self.base(pair))
        base=(previous+self.base_decoder(latent)).clamp(0,1);shifts=[(-1,0),(1,0),(0,-1),(0,1),(0,0)]
        errors=[];flows=[]
        for dx,dy in shifts:
            flow=torch.ones_like(pair[:,:2])*torch.tensor([dx,dy],device=pair.device)[None,:,None,None];flows.append(flow)
            errors.append(-(warp(previous,flow)-current).square().mean(1))
        probability=torch.stack(errors,1).mul(10).softmax(1);entropy=-(probability*probability.clamp_min(1e-8).log()).sum(1,keepdim=True)
        flow=sum(probability[:,k:k+1]*flows[k] for k in range(len(shifts)))
        residual=current-base;gate=entropy/1.609438
        if self.c['variant']=='no_entropy_gate':gate=torch.ones_like(gate)
        enhancement=self.quantize(self.enhance(torch.cat([residual,entropy,flow[:,:1]],1))*gate)
        refinement=self.enhance_decoder(enhancement);restored=(base+refinement[:,:3]).clamp(0,1);refined_flow=flow+refinement[:,3:]
        if self.c['variant'] in {'baseline','base_only'}:restored=base;refined_flow=flow
        step=self.c.get('step',.1)
        rate=lambda z:-(torch.sigmoid(z+step/2)-torch.sigmoid(z-step/2)).clamp_min(1e-8).log2().mean((1,2,3))
        return {'prediction':restored,'base_prediction':base,'flow':refined_flow,'dominant_flow':flow,'entropy':entropy,
                'base_latent':latent,'enhancement_latent':enhancement,'base_rate':rate(latent)[:,None],
                'enhancement_rate':rate(enhancement)[:,None],'transport':warp(previous,refined_flow)}
    def objective(self,b,p):
        target=b['y'].float();loss=F.mse_loss(p['base_prediction'],target)+F.mse_loss(p['prediction'],target)
        loss=loss+self.c.get('rate_weight',.01)*(p['base_rate'].mean()+p['enhancement_rate'].mean())
        if self.c['variant'] not in {'baseline','base_only'}:loss=loss+.2*F.l1_loss(p['transport'],target)
        return loss


def demo(seed,n,c):
    import numpy as np
    d=image_demo(seed,n,mask=False);previous=d['x'];current=np.roll(previous,1,axis=-1);d['x']=np.concatenate([previous,current],1);d['y']=current;return d


def measure(p,d,c):
    import numpy as np
    r={'base_bits_per_pixel':float(p['base_rate'].mean()*p['base_latent'].shape[1]),
       'enhancement_bits_per_pixel':float(p['enhancement_rate'].mean()*p['enhancement_latent'].shape[1]),
       'feature_transport_mae':float(np.mean(abs(p['transport']-d['y'])))}
    if 'flow' in d:r['flow_epe']=float(np.linalg.norm(p['flow']-d['flow'],axis=1).mean())
    return r
