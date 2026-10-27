import numpy as np, json
from neuroblock import run, summarize
from brain2 import Brain2
SEEDS=range(8)
TASKS={'medium':(dict(sigma=0.10,K=8),0.5),'hard':(dict(sigma=0.15,K=10),0.7)}
VARIANTS={'base':{},'rigid':dict(rigid=True),'expand':dict(expand=True),'rigid+expand':dict(rigid=True,expand=True)}
def batch(cfg,**kw):
    outs=[];exps=[];lrs=[]
    for s in SEEDS:
        rows,_,dw=run(cfg,s,cls=Brain2,**kw); o=summarize(rows,dw); outs.append(o)
        exps.append(rows[-1]['exp']); lrs.append(rows[-1]['lr_mean'])
    m={k:float(np.nanmean([o[k] for o in outs if np.isfinite(o[k])] or [np.nan])) for k in outs[0]}
    m['rec_never']=sum(1 for o in outs if not np.isfinite(o['recovery_windows']))
    m['exp']=float(np.mean(exps)); m['lr']=float(np.mean(lrs))
    m['acc_sd']=float(np.std([o['post_acc'] for o in outs]))
    return m
R={}
for task,(kw,rho) in TASKS.items():
  for dm in ['replace','shift']:
    for sel in ['fast','best']:
      key=f'{task}|{dm}|{sel}'
      R[key]={v:batch(dict(rho=rho,move=True,merge=True,prune=True,select=sel,**ex),drift_mode=dm,**kw) for v,ex in VARIANTS.items()}
# rigidity strength sweep (shift drift, hard, fast)
kw,rho=TASKS['hard']
R['c0 sweep|hard|shift']={f'c0={c}':batch(dict(rho=rho,move=True,merge=True,prune=True,rigid=True,c0=c),drift_mode='shift',**kw) for c in [3,10,30,100,300,1e9]}
json.dump(R,open('results2.json','w'),indent=1)
cols=['pre_acc','post_acc','acc_sd','pre_t_query','pre_n','post_n','pre_mean_edge','recovery_windows','rec_never','exp','lr']
for sec,d in R.items():
    print('\n##',sec); print('%-14s'%'',' '.join('%8s'%c[:8] for c in cols))
    for n,m in d.items(): print('%-14s'%n,' '.join('%8.3f'%m[c] for c in cols))
