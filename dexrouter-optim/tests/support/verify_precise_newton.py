#!/usr/bin/env python3
"""Independent all-pool MP check of production twofold solver outputs (offline)."""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import mpmath as mp
import newton_precision_analysis as oracle


def word(value):
    return mp.mpf(struct.unpack('>d', bytes.fromhex(value))[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision', type=int, default=90)
    args = parser.parse_args()
    mp.mp.dps = args.precision
    data = json.loads(args.input.read_text())
    saved = json.loads(args.results.read_text())
    if isinstance(saved, dict):
        witnesses = {w['id']: w for w in saved['witnesses']}
        rows = []
        for base in saved['rows']:
            row = dict(base)
            if 'witness' in row:
                w = witnesses[row['witness']]
                row.update(prices=w['prices'], warm_prices=w['warm_prices'])
            rows.append(row)
    else:
        rows = saved
    verified = []
    seen = set()
    for row in rows:
        if not row['mode'].endswith('twofold'):
            continue
        key = (row['snapshot'], row['mode'], json.dumps(row['prices']), json.dumps(row['warm_prices']))
        if key in seen:
            continue
        seen.add(key)
        case = next(c for c in data if c['snapshot'] == row['snapshot'])
        state = next(s for s in case['states'] if s['state'] == 'final')
        p = [word(v['hi']) + word(v['lo']) for v in row['prices']]
        warm = [word(v) for v in row['warm_prices']]
        assert len(p) == len(state['price']) and all(v > 0 for v in p)
        assert p[state['anchor']] == 1
        # Original economic tolerances, every original pool, original binary64 inputs.
        actual = oracle.evaluate(state, p, case['tolerance'])
        reference = oracle.evaluate(state, warm, case['tolerance'])
        ratio = max(actual['ratios'])
        delta = actual['cost'] - reference['cost']
        assert ratio <= 1 and abs(float(ratio) - row['kkt']) < 1e-6
        assert delta <= 0
        item = {'snapshot': row['snapshot'], 'mode': row['mode'], 'pools': len(state['edges']),
                'mp_kkt': float(ratio), 'native_kkt': row['kkt'],
                'mp_refinement_delta': mp.nstr(delta, 35),
                'native_refinement_delta': row['refinement_delta']}
        print(item)
        verified.append(item)
    assert len(verified) >= 3
    args.output.write_text(json.dumps({'precision_digits': args.precision,
        'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'results_sha256': hashlib.sha256(args.results.read_bytes()).hexdigest(),
        'verified': verified}, indent=2) + '\n')


if __name__ == '__main__':
    main()
