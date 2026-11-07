import numpy as np
from neuroblock import run, summarize
from brain2 import Brain2
S=range(8); kw=dict(sigma=0.10,K=8); rho=0.5
def go(cfg,**k):
    o=[];rs=[]
    for s in S:
        rows,_,dw=run(cfg,s,cls=Brain2,drift_mode='shift',**kw,**k); o.append(summarize(rows,dw)); rs.append(rows)
    return o,rs
print('## condensing ablation (medium, shift), time = mean us to fire')
print('%-16s %8s %8s %8s %8s %8s'%('cond','t_first','t_final','edge','n','acc'))
for name,c in [('none',{}),('move',dict(move=True)),('merge',dict(merge=True)),('prune',dict(prune=True)),('full',dict(move=True,merge=True,prune=True))]:
    o,rs=go(dict(rho=rho,**c))
    t0=np.mean([r[0]['t_query'] for r in rs]); t1=np.mean([o_['post_t_query'] for o_ in o])
    print('%-16s %8.1f %8.1f %8.3f %8.1f %8.3f'%(name,t0,t1,np.mean([o_['post_mean_edge'] for o_ in o]),np.mean([o_['post_n'] for o_ in o]),np.mean([o_['post_acc'] for o_ in o])))
print('\n## churn check: expand drive with/without merge+prune (select=best)')
for name,c in [('expand, no merge/prune',dict(expand=True)),('expand, merge+prune',dict(expand=True,merge=True,prune=True)),('no expand, no merge/prune',{})]:
    o,rs=go(dict(rho=rho,select='best',**c))
    print('%-28s n_final=%.1f  accepted_expansions=%.1f'%(name,np.mean([r[-1]['n'] for r in rs]),np.mean([r[-1]['exp'] for r in rs])))
