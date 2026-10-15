#!/usr/bin/env python3
"""Offline exact regional elimination and f64/MP price/quote error separation.
No pool is removed from the objective. MP starts from exact binary64 inputs.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
import mpmath as mp
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
import laplacian_spectrum as spectral


def pool_quote(edge, price):
    p = edge['parameters']; a,b = edge['a'],edge['b']
    v0,v1 = price[a],price[b]; fee = mp.mpf(p['fee']); current = mp.mpf(p['current_price'])
    ticks=p['lower_prices']; liquidity=p['liquidity']; ct=p['current_tick']
    down=mp.sqrt(v0/(v1*fee)); up=mp.sqrt(v0*fee/v1)
    inputs=[mp.mpf(0),mp.mpf(0)]; outputs=inputs.copy(); value=mp.mpf(0); active=None
    if down < current:
        direction=1; target=down; walk=range(ct,len(liquidity))
    elif up > current and ct < len(ticks)-1:
        direction=-1; target=up; walk=range(ct,-1,-1)
    else:
        weight=mp.mpf(0) if fee!=1 else mp.mpf(liquidity[ct])*mp.sqrt(v0*v1)/2
        return inputs,outputs,value,weight
    initial=True
    for i in walk:
        k=mp.mpf(liquidity[i])
        if k<=0: initial=False; continue
        start=current if i==ct else mp.mpf(ticks[i] if direction==1 else ticks[i+1])
        end_bound=mp.mpf(ticks[i+1] if i+1<len(ticks) else 0) if direction==1 else mp.mpf(ticks[i])
        end=max(target,end_bound) if direction==1 else min(target,end_bound)
        distance=start-end if direction==1 else end-start
        if distance<=0:
            if initial: continue
            break
        token1=k*distance; token0=token1/start/end
        if direction==1: inputs[0]+=token0/fee; outputs[1]+=token1
        else: inputs[1]+=token1/fee; outputs[0]+=token0
        exhausted=(end==end_bound)
        boundary=abs(end-target)*(end+target)/end if exhausted else 0
        weight=v1 if direction==1 else v1/fee
        value+=k*weight*distance*(distance+boundary)/start
        if not exhausted or end==target: active=i
        initial=False
    curvature=mp.mpf(0) if active is None else mp.mpf(liquidity[active])*mp.sqrt(v0*v1)/mp.sqrt(fee)/2
    return inputs,outputs,value,curvature


def evaluate(state,price,tolerance):
    n=len(price); g=[mp.mpf(0) for _ in price]; turnover=g.copy(); weights=[]; cost=mp.mpf(0)
    for e in state['edges']:
        inp,out,value,w=pool_quote(e,price); weights.append(w);cost+=value
        for k,i in enumerate([e['a'],e['b']]): g[i]+=out[k]-inp[k];turnover[i]+=out[k]+inp[k]
    allowed=[mp.mpf(tolerance)+mp.mpf(1e-8)*turnover[i]*price[i] for i in range(n)]
    ratios=[abs(g[i])*price[i]/allowed[i] if i!=state['anchor'] else mp.mpf(0) for i in range(n)]
    return {'g':g,'turnover':turnover,'weights':weights,'allowed':allowed,'ratios':ratios,'cost':cost}


def graph_matrix(state,vertices,weights):
    copy=dict(state);copy['edges']=[dict(e,weight=w) for e,w in zip(state['edges'],weights)]
    return spectral.matrix(copy,vertices)


def eliminate(state,degree,rhs):
    anchor=state['anchor']; group=spectral.reachable(set(range(len(state['tokens']))),state['edges'],anchor)
    vertices=sorted(group-{anchor}); adjacency={i:{} for i in group}
    for e in state['edges']:
        a,b=e['a'],e['b'];w=mp.mpf(e['weight'])
        if w<=0 or a==b or a not in group: continue
        adjacency[a][b]=adjacency[a].get(b,mp.mpf(0))+w;adjacency[b][a]=adjacency[a][b]
    b={i:rhs[k] for k,i in enumerate(vertices)}; records=[]
    while True:
        candidates=[i for i in adjacency if i!=anchor and len(adjacency[i])<=degree]
        if not candidates: break
        i=min(candidates,key=lambda j:(len(adjacency[j]),j)); neighbors=list(adjacency[i].items());d=mp.fsum(w for _,w in neighbors)
        assert d>0
        records.append((i,neighbors,d,b[i]))
        for j,w in neighbors:
            if j!=anchor: b[j]+=w/d*b[i]
            del adjacency[j][i]
        for u,(j,wj) in enumerate(neighbors):
            for k,wk in neighbors[u+1:]:
                value=adjacency[j].get(k,mp.mpf(0))+wj*wk/d
                adjacency[j][k]=value;adjacency[k][j]=value
        del adjacency[i]
    core=sorted(set(adjacency)-{anchor});a=mp.matrix(len(core));local={i:k for k,i in enumerate(core)}
    for i in core:
        a[local[i],local[i]]=mp.fsum(adjacency[i].values())
        for j,w in adjacency[i].items():
            if j!=anchor: a[local[i],local[j]]=-w
    x={anchor:mp.mpf(0)}
    if core:
        solution=mp.lu_solve(a,mp.matrix([b[i] for i in core]));x.update({i:solution[k] for k,i in enumerate(core)})
    for i,neighbors,d,bi in reversed(records): x[i]=(bi+mp.fsum(w*x[j] for j,w in neighbors))/d
    original=spectral.matrix(state,vertices);sol=mp.matrix([x[i] for i in vertices]);dense=mp.lu_solve(original,mp.matrix(rhs))
    residual=max(abs(v) for v in original*sol-mp.matrix(rhs))/max(mp.mpf('1e-80'),max(abs(v) for v in rhs))
    difference=max(abs(v) for v in sol-dense)/max(mp.mpf('1e-80'),max(abs(v) for v in dense))
    assert residual<mp.mpf('1e-60') and difference<mp.mpf('1e-60')
    result={'max_eliminated_degree':degree,'original_dimension':len(vertices),'core_dimension':len(core),
            'core_tokens':[state['tokens'][i] for i in core],'eliminated_tokens':[state['tokens'][i] for i,_,_,_ in records],
            'reconstructed_relative_residual':float(residual),'relative_difference_from_dense_solve':float(difference)}
    if core:
        result['relative']=spectral.spectrum(a,False)[2];result['jacobi']=spectral.spectrum(spectral.normalized(a)[0],False)[2]
    return result


def metric(ev):
    return {'kkt':float(max(ev['ratios'])),'worst_token':int(max(range(len(ev['ratios'])),key=lambda i:ev['ratios'][i])),'cost':float(ev['cost'])}


def analyze_final(case,state,probes,run_grid=True):
    p=[mp.mpf(v) for v in state['price']];tolerance=case['tolerance']; base=evaluate(state,p,tolerance)
    native_g=[mp.mpf(v) for v in state['gradient']];native_quote_g=[mp.mpf(0) for _ in p]
    for e in state['edges']:
        for k,i in enumerate([e['a'],e['b']]): native_quote_g[i]+=mp.mpf(e['output'][k])-mp.mpf(e['input'][k])
    quote_error=[abs(native_quote_g[i]-base['g'][i])*p[i]/base['allowed'][i] for i in range(len(p))]
    sum_error=[abs(native_g[i]-native_quote_g[i])*p[i]/base['allowed'][i] for i in range(len(p))]
    component=spectral.reachable(set(range(len(p))),state['edges'],state['anchor']);vertices=sorted(component-{state['anchor']})
    a=graph_matrix(state,vertices,base['weights']);local={i:k for k,i in enumerate(vertices)}
    resolutions=[]
    for i in vertices:
        ulp=mp.mpf(math.nextafter(float(p[i]),math.inf))-p[i]
        rho=a[local[i],local[i]]*(ulp/p[i])/base['allowed'][i]
        resolutions.append({'token':state['tokens'][i],'index':i,'price':float(p[i]),'ulp':float(ulp),'diagonal_one_ulp_over_tolerance':float(rho),
            'native_gradient':float(native_g[i]),'mp_gradient':float(base['g'][i]),'mp_kkt':float(base['ratios'][i]),
            'quote_error_over_tolerance':float(quote_error[i]),'sum_error_over_tolerance':float(sum_error[i])})
    resolutions.sort(key=lambda r:r['diagonal_one_ulp_over_tolerance'],reverse=True)
    point=p.copy();iterations=[metric(base)]
    for _ in range(20):
        ev=evaluate(state,point,tolerance)
        if max(ev['ratios']) < mp.mpf('.05'): break
        graph=dict(state,edges=[dict(e,weight=w) for e,w in zip(state['edges'],ev['weights'])])
        live=spectral.reachable(set(range(len(p))),graph['edges'],state['anchor'])
        moving=sorted(live-{state['anchor']});aa=graph_matrix(state,moving,ev['weights'])
        y=mp.lu_solve(aa,mp.matrix([-point[i]*ev['g'][i] for i in moving]))
        dp={i:point[i]*y[k] for k,i in enumerate(moving)}
        slope=mp.fsum(ev['g'][i]*dp[i] for i in moving); assert slope<0
        alpha=mp.mpf(1)
        for _ in range(80):
            trial=[v+alpha*dp.get(i,0) for i,v in enumerate(point)]
            if all(v>0 for v in trial):
                candidate=evaluate(state,trial,tolerance)
                if candidate['cost']<=ev['cost']+mp.mpf('1e-4')*alpha*slope: break
            alpha/=2
        else: raise AssertionError('MP Armijo failed')
        point=trial;row=metric(candidate);row.update({'alpha':float(alpha),'curved_dimension':len(moving)});iterations.append(row)
    continuous=evaluate(state,point,tolerance)
    rounded=[mp.mpf(float(v)) for v in point];r_ev=evaluate(state,rounded,tolerance)
    probe_case={'snapshot':case['snapshot'],'cutoff':case['cutoff'],'tolerance':tolerance,'baseline':state['price'],'points':[
        {'label':'baseline','price':state['price'],'mp_kkt':metric(base)['kkt']},{'label':'rounded-mp-newton','price':[float(v) for v in rounded],'mp_kkt':metric(r_ev)['kkt']}]}
    result={'snapshot':case['snapshot'],'native_reported':state['economic'],'mp_same_prices':metric(base),
        'maximum_quote_error_over_tolerance':float(max(quote_error)), 'maximum_sum_error_over_tolerance':float(max(sum_error)),
        'resolution':resolutions,'mp_newton_iterations':iterations,'mp_prices':metric(continuous),'rounded_prices_mp_quotes':metric(r_ev),
        'mp_cost_change':mp.nstr(continuous['cost']-base['cost'],30), 'rounded_cost_change':mp.nstr(r_ev['cost']-base['cost'],30)}
    if case['snapshot']=='full-strict':
        result['price_bit_sweep']=[]
        for bits in [53,56,60,64,80,106]:
            with mp.workprec(bits): quantized=[+v for v in point]
            result['price_bit_sweep'].append({'price_bits':bits,**metric(evaluate(state,quantized,tolerance))})
        hi=[mp.mpf(float(v)) for v in point];lo=[float(v-h) for v,h in zip(point,hi)]
        twofold=[h+mp.mpf(l) for h,l in zip(hi,lo)]
        result['twofold_prices_mp_quotes']=metric(evaluate(state,twofold,tolerance))
        result['twofold_price_bits']=[{'hi':struct.pack('>d',float(h)).hex(),'lo':struct.pack('>d',l).hex()} for h,l in zip(hi,lo)]
        result['pair_cost_bound']=pair_cost_bound(state,point,rounded,continuous,base,tolerance,512)
    if case['snapshot']=='full-strict' and run_grid:
        # Bounded integer-grid diagnostic, not a global impossibility certificate.
        vertices=[i for i in range(len(p)) if i!=state['anchor']]
        aa=graph_matrix(state,vertices,r_ev['weights']);n=len(vertices)
        ulps=[mp.mpf(math.nextafter(float(rounded[i]),math.inf))-rounded[i] for i in vertices]
        model=np.array([[float(aa[k,j]*ulps[j]/rounded[vertices[j]]/r_ev['allowed'][i]) for j in range(n)] for k,i in enumerate(vertices)])
        r=np.array([float(r_ev['g'][i]*rounded[i]/r_ev['allowed'][i]) for i in vertices])
        constraints=np.vstack([np.column_stack([model,-np.ones(n)]),np.column_stack([-model,-np.ones(n)])])
        # Keep each pool inside its CURRENT ratio cell, including inactive pools.
        # A frozen active Hessian without these constraints gives false grid predictions.
        unconstrained_constraints=constraints.copy()
        cell_rows=[];cell_upper=[]
        variable={i:k for k,i in enumerate(vertices)}
        for e in state['edges']:
            pp=e['parameters'];fee=mp.mpf(pp['fee']);current=mp.mpf(pp['current_price']);u,v=e['a'],e['b']
            ratio=rounded[u]/rounded[v]
            breaks=[fee*current**2,current**2/fee]
            breaks += [fee*mp.mpf(t)**2 if t<=pp['current_price'] else mp.mpf(t)**2/fee for t in pp['lower_prices'] if t>0]
            lower=max([t for t in breaks if t<=ratio],default=mp.mpf(0));upper=min([t for t in breaks if t>ratio],default=mp.inf)
            for boundary,is_lower in [(lower,True),(upper,False)]:
                if not mp.isfinite(boundary) or boundary==0: continue
                coefficients=[mp.mpf(0) for _ in vertices]
                if u in variable: coefficients[variable[u]]+=ulps[variable[u]]
                if v in variable: coefficients[variable[v]]-=boundary*ulps[variable[v]]
                scale=max(abs(x) for x in coefficients)
                if scale==0: continue
                margin=rounded[u]-boundary*rounded[v]
                if abs(margin)>512*mp.fsum(abs(x) for x in coefficients): continue
                sign=-1 if is_lower else 1
                cell_rows.append([float(sign*x/scale) for x in coefficients]+[0.0]);cell_upper.append(float(-sign*margin/scale))
        if cell_rows:
            constraints=np.vstack([constraints,np.array(cell_rows)])
        result['bounded_grid_cell_constraints']=len(cell_rows)
        result['bounded_grid_search']=[]
        for same_region,radius in [(False,512),(True,32),(True,512)]:
            solution=milp(np.r_[np.zeros(n),1.0],integrality=np.r_[np.ones(n),0],
                bounds=Bounds(np.r_[-radius*np.ones(n),0.0],np.r_[radius*np.ones(n),np.inf]),
                constraints=LinearConstraint(constraints if same_region else unconstrained_constraints,-np.inf,np.r_[-r,r,cell_upper] if same_region else np.r_[-r,r]),options={'time_limit':15.0,'mip_rel_gap':1e-9})
            record={'same_region':same_region,'radius_ulps':radius,'status':int(solution.status),'message':str(solution.message)}
            if solution.x is not None:
                z=np.rint(solution.x[:n]).astype(int);trial=rounded.copy()
                for k,i in enumerate(vertices): trial[i]=mp.mpf(float(rounded[i]+int(z[k])*ulps[k]))
                actual=evaluate(state,trial,tolerance);prediction=r+model@z
                record.update({'linearized_kkt':float(max(abs(prediction))),'mp_oracle_kkt':metric(actual)['kkt'],
                    'mp_cost_change':mp.nstr(actual['cost']-base['cost'],30),
                    'dual_bound':float(solution.mip_dual_bound) if getattr(solution,'mip_dual_bound',None) is not None else None,
                    'offsets':{state['tokens'][i]:int(z[k]) for k,i in enumerate(vertices) if z[k]}})
                probe_case['points'].append({'label':f'grid-{same_region}-{radius}','price':[float(v) for v in trial],'mp_kkt':metric(actual)['kkt']})
            result['bounded_grid_search'].append(record)
    probes.append(probe_case)
    return result


def pair_cost_bound(state,point,center,continuous,baseline,tolerance,radius):
    """Conditional current-cell bound, not a global f64 impossibility claim."""
    root=state['tokens'].index('WETH');leaf=state['tokens'].index('wstETH');anchor=state['anchor']
    incident=[e for e in state['edges'] if leaf in [e['a'],e['b']]]
    weights=[pool_quote(e,center)[3] for e in incident]
    curved=[e for e,w in zip(incident,weights) if w>0]
    assert len(curved)==1 and root in [curved[0]['a'],curved[0]['b']]
    ground=next(e for e in state['edges'] if set([e['a'],e['b']])=={root,anchor} and pool_quote(e,point)[3]>0)
    ground_w=pool_quote(ground,point)[3]
    for e in incident:
        if e is not curved[0]:
            t0=pool_quote(e,point);t1=pool_quote(e,center)
            assert t0[0]==t1[0] and t0[1]==t1[1], 'constant-trade cell assumption failed'
    ulps=[mp.mpf(math.nextafter(float(v),math.inf))-v for v in center]
    budget=mp.fsum(abs(continuous['g'][i])*(abs(center[i]-point[i])+radius*ulps[i]) for i in range(len(point)) if i!=anchor)
    for k in [-radius,radius]:
        test=point.copy();test[root]=center[root]+k*ulps[root]
        expected=ground_w*mp.sqrt(test[root]/point[root])
        assert abs(pool_quote(ground,test)[3]/expected-1)<mp.mpf('1e-70'), 'ground pool left its region'
    candidates=[]
    for k in range(-radius,radius+1):
        pr=center[root]+k*ulps[root];ideal=point[leaf]*pr/point[root];near=float(ideal)
        for value in [math.nextafter(near,-math.inf),near,math.nextafter(near,math.inf)]:
            pl=mp.mpf(value)
            if abs(pl-center[leaf])>radius*ulps[leaf]: continue
            trial=point.copy();trial[root]=pr;trial[leaf]=pl
            g=mp.mpf(0);turnover=mp.mpf(0)
            for e in incident:
                inp,out,_,_=pool_quote(e,trial);j=int(e['b']==leaf)
                g+=out[j]-inp[j];turnover+=out[j]+inp[j]
            ratio=abs(g)*pl/(mp.mpf(tolerance)+mp.mpf(1e-8)*turnover*pl)
            if ratio<=1:
                # All other pool remainders are nonnegative by convexity.
                remainder=2*ground_w/point[root]*(mp.sqrt(pr)-mp.sqrt(point[root]))**2
                bound=continuous['cost']-baseline['cost']-budget+remainder
                candidates.append((bound,k,int(mp.nint((pl-center[leaf])/ulps[leaf])),ratio))
    best=min(candidates) if candidates else None
    return {'scope':'Current-cell +/- radius binary64 neighborhood, constant trades in zero-curvature wstETH pools; MP numerical bound, not interval-certified or global.',
        'radius_ulps':radius,'candidate_pair_count':len(candidates),'linear_term_budget':mp.nstr(budget,30),
        'minimum_objective_change_lower_bound':mp.nstr(best[0],30) if best else None,
        'minimizing_pair_offsets':[best[1],best[2]] if best else None,
        'leaf_kkt_at_pair':float(best[3]) if best else None}


def self_test():
    mp.mp.dps=90
    e={'a':0,'b':1,'parameters':{'current_price':1.0,'current_tick':0,'lower_prices':[2.0,0.5],'liquidity':[1000.0,0.0],'fee':1.0}}
    for v0 in [mp.mpf('.8'),mp.mpf('1.2')]:
        inp,out,value,w=pool_quote(e,[v0,mp.mpf(1)])
        assert abs(value-(v0*(out[0]-inp[0])+out[1]-inp[1]))<mp.mpf('1e-80')
        h=mp.mpf('1e-35');gp=pool_quote(e,[v0+h,mp.mpf(1)])[2];gm=pool_quote(e,[v0-h,mp.mpf(1)])[2]
        assert abs((gp-gm)/(2*h)-(out[0]-inp[0]))<mp.mpf('1e-30')
    s={'anchor':0,'tokens':['anchor','a','b'],'edges':[{'index':0,'a':0,'b':1,'weight':1e-12},{'index':1,'a':1,'b':2,'weight':1e12}]}
    assert eliminate(s,1,[mp.mpf(2),mp.mpf(3)])['core_dimension']==0
    s['edges'].append({'index':2,'a':0,'b':2,'weight':2.0})
    assert eliminate(s,2,[mp.mpf(2),mp.mpf(3)])['core_dimension']==0
    print('precision controls passed: quote value/gradient, weak-spring elimination, cycle elimination')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--probes',type=Path);parser.add_argument('--self-test',action='store_true');parser.add_argument('--precision',type=int,default=90);parser.add_argument('--snapshot');parser.add_argument('--no-grid',action='store_true')
    args=parser.parse_args()
    if args.self_test: self_test();return
    if not(args.input and args.output and args.probes): parser.error('--input, --output and --probes required')
    mp.mp.dps=args.precision;data=json.loads(args.input.read_text());results=[];eliminations=[];probes=[];cache={}
    for case in data:
        if args.snapshot and args.snapshot!=case['snapshot']: continue
        for state in case['states']:
            key=json.dumps([state['tokens'],state['free'],[(e['a'],e['b'],e['weight']) for e in state['edges']]])
            if key not in cache:
                group=spectral.reachable(set(range(len(state['tokens']))),state['edges'],state['anchor']);vertices=sorted(group-{state['anchor']})
                rhs=[-mp.mpf(state['price'][i])*mp.mpf(state['gradient'][i]) for i in vertices]
                cache[key]=[eliminate(state,d,rhs) for d in [1,2]]
            eliminations.append({'snapshot':case['snapshot'],'state':state['state'],'reductions':cache[key]})
            if state['state']=='final':
                print('precision',case['snapshot'],flush=True);result=analyze_final(case,state,probes,not args.no_grid);results.append(result)
                print(case['snapshot'],'native',state['economic']['kkt'],'MP',result['mp_same_prices']['kkt'],'continuous',result['mp_prices']['kkt'],'rounded',result['rounded_prices_mp_quotes']['kkt'],flush=True)
    args.output.write_text(json.dumps({'precision_decimal_digits':args.precision,'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'eliminations':eliminations,'precision':results},indent=2,allow_nan=False)+'\n')
    for case in probes:
        case['baseline_bits']=[struct.pack('>d',float(v)).hex() for v in case['baseline']]
        for point in case['points']: point['price_bits']=[struct.pack('>d',float(v)).hex() for v in point['price']]
    args.probes.write_text(json.dumps(probes,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
