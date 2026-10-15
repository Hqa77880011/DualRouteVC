import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from .method import Model


def array_bytes(value):
    stream=io.BytesIO()
    np.save(stream,value,allow_pickle=False)
    return stream.getvalue()


def read_array(archive,name):
    return np.load(io.BytesIO(archive.read(name)),allow_pickle=False)


def quantize(value,step):
    integer=np.rint(value.detach().cpu().numpy()/step)
    if np.max(abs(integer))>32767:
        raise ValueError('Quantized latent exceeds the signed 16-bit stream format')
    return integer.astype(np.int16)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['encode','decode'])
    parser.add_argument('--model',required=True)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--base-only',action='store_true')
    args=parser.parse_args()
    state=torch.load(args.model,map_location='cpu',weights_only=True)
    model=Model(state['config'])
    model.load_state_dict(state['state'])
    model.eval()
    step=state['config'].get('step',.1)
    if args.command=='encode':
        with np.load(args.input,allow_pickle=False) as data:
            frames=data['x'].astype(np.float32)
        if frames.ndim!=4 or frames.shape[1]!=3:
            raise ValueError('Video input is x[frames,3,height,width] in [0,1]')
        initial=np.rint(np.clip(frames[0],0,1)*255).astype(np.uint8)
        reference=torch.from_numpy(initial.astype(np.float32)/255)[None]
        with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('header.json',json.dumps({'format':1,'frames':len(frames),'step':step,'shape':list(frames.shape[1:]),'enhancement':not args.base_only}))
            archive.writestr('initial.npy',array_bytes(initial))
            for index in range(1,len(frames)):
                current=torch.from_numpy(frames[index:index+1])
                with torch.no_grad():
                    result=model({'x':torch.cat([reference,current],1)})
                    base=quantize(result['base_latent'],step)
                    enhancement=quantize(result['enhancement_latent'],step)
                    motion=quantize(result['dominant_flow'],step)
                    reference=(reference+model.base_decoder(torch.from_numpy(base.astype(np.float32)*step))).clamp(0,1)
                archive.writestr(f'base_{index}.npy',array_bytes(base))
                if not args.base_only:
                    archive.writestr(f'enhancement_{index}.npy',array_bytes(enhancement))
                archive.writestr(f'motion_{index}.npy',array_bytes(motion))
        pixels=frames.shape[0]*frames.shape[-2]*frames.shape[-1]
        Path(args.output+'.json').write_text(json.dumps({'stream_bytes':Path(args.output).stat().st_size,'actual_bits_per_pixel':8*Path(args.output).stat().st_size/pixels},indent=2))
    else:
        decoded=[]
        flows=[]
        with zipfile.ZipFile(args.input) as archive:
            header=json.loads(archive.read('header.json'))
            if header['format']!=1:
                raise ValueError('Unsupported stream version')
            step=header['step']
            reference=torch.from_numpy(read_array(archive,'initial.npy').astype(np.float32)/255)[None]
            decoded.append(reference[0].numpy())
            flows.append(np.zeros((2,*reference.shape[-2:]),np.float32))
            for index in range(1,header['frames']):
                base=torch.from_numpy(read_array(archive,f'base_{index}.npy').astype(np.float32)*step)
                with torch.no_grad():
                    reference=(reference+model.base_decoder(base)).clamp(0,1)
                    output=reference
                    motion=torch.from_numpy(read_array(archive,f'motion_{index}.npy').astype(np.float32)*step)
                    flow=motion
                    if not args.base_only and header.get('enhancement',True):
                        enhancement=torch.from_numpy(read_array(archive,f'enhancement_{index}.npy').astype(np.float32)*step)
                        refinement=model.enhance_decoder(enhancement)
                        output=(reference+refinement[:,:3]).clamp(0,1)
                        flow=motion+refinement[:,3:]
                decoded.append(output[0].numpy())
                flows.append(flow[0].numpy())
        np.savez_compressed(args.output,x=np.stack(decoded),flow=np.stack(flows))


if __name__=='__main__':main()
