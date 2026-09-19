"""One-command offline, CPU-only, bounded pipeline smoke check.

Each architecture performs one optimizer update on two 64x64 synthetic images,
one validation batch, a checkpoint read and one evaluation batch. This is not
an experiment or an attempt to reach the paper's reported accuracy.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=('both','unet_vgg','segformer_b0'),default='both')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    workdir=Path.cwd()
    output=(args.output_dir or workdir/'runs'/('smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('Output is not empty; choose a fresh directory')
    output.mkdir(parents=True,exist_ok=True)
    models=('unet_vgg','segformer_b0') if args.model=='both' else (args.model,)
    environment=os.environ.copy()
    environment.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',
                       OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                       MPLBACKEND='Agg',MPLCONFIGDIR=str(output/'matplotlib'))
    results=[]
    for name in models:
        train_dir=output/name/'train'
        eval_dir=output/name/'evaluation'
        commands=[
            [sys.executable,'-m','scripts.train','--dry-run','--model',name,'--output-dir',str(train_dir)],
            [sys.executable,'-m','scripts.evaluate','--dry-run','--model',name,'--checkpoint',str(train_dir/'checkpoint.pt'),'--output-dir',str(eval_dir)],
        ]
        for command in commands:
            result=subprocess.run(command,cwd=workdir,env=environment,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            log=output/name/('train.log' if command[2]=='scripts.train' else 'evaluation.log')
            log.parent.mkdir(parents=True,exist_ok=True);log.write_text(result.stdout)
            print(result.stdout,flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)
        results.append(eval_dir/'metrics.json')
    if len(results)>1:
        subprocess.run([sys.executable,'-m','scripts.compare',*map(str,results),'--output',str(output/'comparison.md')],
                       cwd=workdir,env=environment,check=True)
    (output/'SMOKE_STATUS.json').write_text(json.dumps({'status':'passed','synthetic_only':True,'device':'cpu',
        'models':list(models),'images_per_batch':2,'image_size':[64,64],'optimizer_updates_per_model':1},indent=2)+'\n')
    print(output)

if __name__=='__main__':
    main()
