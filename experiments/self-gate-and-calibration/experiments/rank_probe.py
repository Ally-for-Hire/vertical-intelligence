import sys; sys.path.insert(0,'../src')
import numpy as np, agent, ranked
env,enc,m,fh,log,snaps,rk=agent.train(0,8,64000,"hard",ck=(2000,4000,8000,16000,32000,64000))
names=env.names; S,A,R=log["hist"]["S"],log["hist"]["A"],log["hist"]["R"]
rng=np.random.default_rng(0)
for ck,s in snaps.items():
    sets={"ranked(greedy)":s["chosen"] or s["top"][:1],"ranked top3 by p":s["top"][:3],"oracle":[names.index(n) for n in("herr","lat","ontrack")],
          "bottom3 by p":[i for i in np.argsort(s["p"]) if names[i] not in ("drag","score")][:3],"random3":list(rng.choice([i for i,n in enumerate(names) if n!="score"],3,replace=False))}
    out={}
    for k,vs in sets.items():
        tab=ranked.Table(vs,S[:ck]); tab.fit(S[:ck],A[:ck],R[:ck])
        out[k]=round(float(np.mean([ranked.drive_table(env,tab,500,sd) for sd in range(3)])),2)
    print(ck,{k:[names[i] for i in v] for k,v in list(sets.items())[:2]},out,flush=True)
