#!/usr/bin/env python3
"""High-precision, fixed-price regional spectrum and per-pool attribution.
Run using the locked Nix Python environment; this never filters production pools.
"""
import argparse, hashlib, json, math
from pathlib import Path
import mpmath as mp
import numpy as np


def reachable(vertices, edges, root):
    adjacency = {i: [] for i in vertices}
    for e in edges:
        if e['weight'] > 0 and e['a'] != e['b'] and e['a'] in adjacency and e['b'] in adjacency:
            adjacency[e['a']].append(e['b']); adjacency[e['b']].append(e['a'])
    seen = {root}; todo = [root]
    while todo:
        for j in adjacency[todo.pop()]:
            if j not in seen: seen.add(j); todo.append(j)
    return seen


def graph_groups(n, edges):
    remaining = set(range(n)); groups = []
    while remaining:
        group = reachable(set(range(n)), edges, min(remaining)); groups.append(group); remaining -= group
    return groups


def matrix(state, vertices, omit=None):
    local = {v: i for i, v in enumerate(vertices)}; a = mp.matrix(len(vertices))
    for e in state['edges']:
        if e['index'] == omit or e['weight'] == 0 or e['a'] == e['b']: continue
        u, v, w = e['a'], e['b'], mp.mpf(e['weight'])
        if u in local: a[local[u], local[u]] += w
        if v in local: a[local[v], local[v]] += w
        if u in local and v in local:
            a[local[u], local[v]] -= w; a[local[v], local[u]] -= w
    return a


def spectrum(a, vectors=True):
    vals, q = mp.eigsy(a) if vectors else (mp.eigsy(a, eigvals_only=True), None)
    assert vals[0] > 0, 'nonpositive eigenvalue: check grounding or increase precision'
    answer = {'min': float(vals[0]), 'max': float(vals[-1]), 'condition': float(vals[-1]/vals[0])}
    if vectors:
        scale = max(mp.fsum(abs(a[i,j]) for j in range(a.cols)) for i in range(a.rows))
        residual = max(max(abs(v) for v in a*q[:,k]-vals[k]*q[:,k]) for k in [0,a.rows-1])
        assert residual / vals[0] < mp.mpf('1e-25'), 'insufficient relative accuracy for smallest eigenvalue'
        answer['extremal_absolute_residual_over_lambda_min'] = float(residual/vals[0])
        answer['extremal_residual_over_matrix_norm'] = float(residual/scale)
    return vals, q, answer


def normalized(a):
    d = [mp.sqrt(a[i,i]) for i in range(a.rows)]
    return mp.matrix([[a[i,j]/d[i]/d[j] for j in range(a.cols)] for i in range(a.rows)]), d


def analyze(case, state, deletions):
    n = len(state['tokens']); anchor = state['anchor']; edges = state['edges']
    assert [i for i, free in enumerate(state['free']) if not free] == [anchor], 'this diagnostic expects one authorized anchor'
    groups = graph_groups(n, edges); component = next(g for g in groups if anchor in g)
    vertices = sorted(component-{anchor}); local = {v:i for i,v in enumerate(vertices)}
    print('spectrum',case['snapshot'],state['state'],'n',len(vertices),flush=True)
    a = matrix(state,vertices); b,d = normalized(a)
    val,q,raw = spectrum(a); nv,nq,norm = spectrum(b)
    raw_double = np.linalg.eigvalsh(np.array(a.tolist(),dtype=float))
    rows = []
    def difference(vec,e):
        return (vec[local[e['a']]] if e['a'] in local else 0)-(vec[local[e['b']]] if e['b'] in local else 0)
    for e in edges:
        if e['weight'] <= 0 or e['a'] not in component or e['b'] not in component: continue
        w = mp.mpf(e['weight']); rdiff = [difference(q[:,k],e) for k in range(a.rows)]
        raw_score = w*(rdiff[-1]**2/val[-1]-rdiff[0]**2/val[0])
        leverage = w*mp.fsum(rdiff[k]**2/val[k] for k in range(a.rows))
        assert -mp.mpf('1e-30') <= leverage <= 1+mp.mpf('1e-30')
        def log_derivative(k):
            v = [nq[i,k]/d[i] for i in range(a.rows)]
            diag = mp.fsum(v[local[t]]**2 for t in [e['a'],e['b']] if t in local)
            return w*(difference(v,e)**2/nv[k]-diag)
        norm_score = log_derivative(a.rows-1)-log_derivative(0)
        without = [other for other in edges if other['index'] != e['index']]
        bridge = not component <= reachable(set(range(n)),without,anchor)
        rows.append({'index':e['index'],'pair':e['pair'],'weight':e['weight'],'support':e['support'],
            'raw_log_condition_derivative':float(raw_score),'jacobi_log_condition_derivative':float(norm_score),
            'leverage':float(leverage),'bridge_in_current_curvature_graph':bridge})
    assert abs(math.fsum(r['leverage'] for r in rows)-len(vertices)) < 1e-10
    assert abs(math.fsum(r['raw_log_condition_derivative'] for r in rows)) < 1e-10
    assert abs(math.fsum(r['jacobi_log_condition_derivative'] for r in rows)) < 1e-10
    rows.sort(key=lambda r:r['jacobi_log_condition_derivative'],reverse=True)
    active = [e for e in edges if e['weight']>0 and e['a']!=e['b']]
    no_trade = lambda e: all(v==0 for v in e['input']+e['output'])
    nullity = sum(not any(not state['free'][i] for i in g) for g in groups)
    result = {'snapshot':case['snapshot'],'state':state['state'],'all_tokens':n,'active_edges':len(active),
        'free_dimension':sum(state['free']),'structural_nullity':nullity,
        'anchor_component_tokens':[state['tokens'][i] for i in sorted(component)],'grounded_dimension':len(vertices),
        'relative':raw,'jacobi':norm,'double_precision_lambda_min':float(raw_double[0]),
        'zero_curvature_pools':sum(e['weight']==0 for e in edges),
        'zero_curvature_pool_ids':[e['index'] for e in edges if e['weight']==0],
        'floating_curved_components':[{'tokens':[state['tokens'][i] for i in sorted(g)],'edges':[e['index'] for e in active if e['a'] in g]} for g in groups if anchor not in g and len(g)>1],
        'zero_curvature_but_trading':sum(e['weight']==0 and not no_trade(e) for e in edges),
        'no_trade_pools':sum(no_trade(e) for e in edges),'pools':rows}
    if deletions:
        selected = []
        categories = [sorted(rows,key=lambda r:r['weight'],reverse=True)[:3],rows[:3],
            sorted(rows,key=lambda r:r['jacobi_log_condition_derivative'])[:3],
            sorted([r for r in rows if r['bridge_in_current_curvature_graph']],key=lambda r:r['weight'])[:2]]
        for group in categories:
            for r in group:
                if r['index'] not in selected: selected.append(r['index'])
        result['single_pool_deletions'] = []
        for idx in selected:
            row = next(r for r in rows if r['index']==idx); out = {'index':idx,'pair':row['pair']}
            if row['bridge_in_current_curvature_graph']:
                out['status']='singular_on_original_variables'
            else:
                removed = matrix(state,vertices,omit=idx)
                _,_,out['relative']=spectrum(removed,False)
                _,_,out['jacobi']=spectrum(normalized(removed)[0],False)
                out['status']='positive_definite'
            result['single_pool_deletions'].append(out)
    return result


def self_test():
    mp.mp.dps=90
    state={'edges':[{'index':0,'a':0,'b':1,'weight':1e-12}, {'index':1,'a':1,'b':2,'weight':1e12}]}
    a=matrix(state,[1,2]); _,_,r=spectrum(a); _,_,s=spectrum(normalized(a)[0])
    assert abs(r['condition']/4e24-1)<1e-14
    assert abs(s['condition']/4e24-1)<1e-14
    assert reachable({0,1,2},state['edges'],0)=={0,1,2}
    assert reachable({0,1,2},state['edges'][:1],0)=={0,1}
    parallel={'edges':[{'index':0,'a':0,'b':1,'weight':2.0},{'index':1,'a':0,'b':1,'weight':3.0}]}
    assert matrix(parallel,[1])[0,0]==5
    assert matrix(parallel,[1],omit=0)[0,0]==3
    print('spectral controls passed: weak grounding, parallel edges, deletion rank')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path);p.add_argument('--output',type=Path);p.add_argument('--precision',type=int,default=90);p.add_argument('--self-test',action='store_true');p.add_argument('--pruning',type=Path);p.add_argument('--precision-check',type=Path)
    args=p.parse_args()
    if args.self_test: self_test(); return
    if not args.input or not args.output: p.error('--input and --output are required')
    mp.mp.dps=args.precision
    data=json.loads(args.input.read_text());results=[];cache={}
    for case in data:
        for state in case['states']:
            key=json.dumps([state['tokens'],state['price'],state['free'],state['anchor'],[(e['a'],e['b'],e['weight']) for e in state['edges']]])
            if key in cache:
                result=dict(cache[key]);result['snapshot']=case['snapshot'];result['state']=state['state']
            else:
                result=analyze(case,state,case['snapshot']=='full' and state['state']=='final');cache[key]=result
            results.append(result)
    output={'precision_decimal_digits':args.precision,'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'scope':'Fixed f64 regional edge weights; relative and Jacobi spectra of the authorized-anchor component; structural nullity reported on ALL original free variables. Not a global convergence or pruning certificate.',
        'results':results}
    output['fixture_sha256']=hashlib.sha256((Path(__file__).parent.parent/'fixtures/uniswap_v3_pools.json').read_bytes()).hexdigest()
    if args.pruning:
        output['pruning']=json.loads(args.pruning.read_text())
        output['pruning_scope']='One run per configuration; cold or oracle-assisted warm start; diagnostic times are not speedup evidence. Full-dual rises at pruned prices are NOT primal-profit losses.'
    if args.precision_check:
        check=json.loads(args.precision_check.read_text()); base=next(r for r in results if r['snapshot']=='full' and r['state']=='final')
        for k in ['relative','jacobi']: assert abs(check[k]['condition']/base[k]['condition']-1)<1e-14
        output['precision_validation']={'digits':110,'relative':check['relative'],'jacobi':check['jacobi']}
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    for r in results:
        print(r['snapshot'],r['state'],'nullity',r['structural_nullity'],'relative',r['relative']['condition'],'Jacobi',r['jacobi']['condition'],flush=True)
    full=next(r for r in results if r['snapshot']=='full' and r['state']=='final')
    print('deletions',json.dumps(full['single_pool_deletions']),flush=True)
if __name__=='__main__': main()
