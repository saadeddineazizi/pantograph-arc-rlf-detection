import json,glob,numpy as np
R={}
for f in glob.glob('out_v19/onset/*.json'):
    r=json.load(open(f)); R[(r['index'],r['D'])]=r
S={}
for D in (0,120,300):
    ks=[i for i in range(600,640) if (i,D) in R]
    S[D]={d:dict(prog=float(np.mean([R[(i,D)]['results'][d]['0.001']['eprog'] for i in ks])),
                 det=float(np.mean([R[(i,D)]['results'][d]['0.001']['det'] for i in ks])),
                 prog01=float(np.mean([R[(i,D)]['results'][d]['0.01']['eprog'] for i in ks]))) for d in ('LQE','R4','TLRmixSR','TLRmix','TLRsr','SRon')}
    S[D]['n']=len(ks)
    x=np.array([R[(i,D)]['results']['TLRmixSR']['0.001']['eprog'] for i in ks]); y=np.array([R[(i,D)]['results']['LQE']['0.001']['eprog'] for i in ks])
    d=x-y; rng=np.random.default_rng(19); bs=rng.choice(d,(10000,d.size)).mean(1)
    S[D]['mixSR_minus_LQE']=dict(mean=float(d.mean()), ci=np.percentile(bs,[2.5,97.5]).tolist(), mixSR_earlier=int((d<0).sum()))
json.dump(S,open('out_v19/onset_summary.json','w'),indent=1)
for D in S: print(D, S[D]['n'], {k:(round(v['prog'],3),round(v['det'],2)) for k,v in S[D].items() if isinstance(v,dict) and 'prog' in v}, S[D]['mixSR_minus_LQE'])
