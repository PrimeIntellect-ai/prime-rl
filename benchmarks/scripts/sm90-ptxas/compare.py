import json, sys, glob, collections
out = sys.argv[1]
arms = {}
for a in ['p129', 'p133', 'noslp']:
    d = {}
    for fn in glob.glob(f'{out}/res-{a}-*.jsonl'):
        for l in open(fn):
            r = json.loads(l); d[r['file']] = r
    arms[a] = d
stats = collections.Counter(); rows = []
for f, r in arms['p129'].items():
    o = arms['p133'].get(f); n = arms['noslp'].get(f)
    if o is None: stats['missing133'] += 1; continue
    for tag, rr in [('p129', r), ('p133', o), ('noslp', n or {})]:
        if 'crash' in rr: stats[f'crash_{tag}'] += 1; rows.append((tag+'-crash', f, rr.get('name'), '', ''))
        if 'error' in rr: stats[f'error_{tag}'] += 1
    for k, c in r.get('configs', {}).items():
        stats['configs'] += 1
        c3 = o.get('configs', {}).get(k); cn = (n or {}).get('configs', {}).get(k)
        if c3 is None: stats['cfg_missing133'] += 1; continue
        dets = (c['det'], c3['det'], None if cn is None else cn['det'])
        if not (c['det'] and c3['det']):
            if c3['det'] and cn is not None and cn['det'] and cn['d'] == c3['d']:
                stats['DIFF_nondet129'] += 1; rows.append(('DIFF_nondet129', f, r['name'], k, 'noslp==133')); continue
            stats['nondet'] += 1; rows.append(('nondet', f, r['name'], k, f'det129/133/noslp={dets}')); continue
        if c['d'] == c3['d']: stats['equal'] += 1; continue
        noslp = None if cn is None else ('noslp==133' if cn['d'] == c3['d'] else ('noslp==129' if cn['d'] == c['d'] else 'noslp_differs'))
        stats['DIFF'] += 1; rows.append(('DIFF', f, r['name'], k, noslp))
print(dict(stats))
for row in sorted(rows): print(*row, sep='\t')
agree = collections.Counter()
for f, o in arms['p133'].items():
    n = arms['noslp'].get(f)
    if n is None: continue
    for k, c3 in o.get('configs', {}).items():
        cn = n.get('configs', {}).get(k)
        if cn is None or not (c3['det'] and cn['det']): continue
        agree['noslp==133' if cn['d'] == c3['d'] else 'noslp!=133'] += 1
        if cn['d'] != c3['d']: print('NOSLP_VS_133', f, o.get('name'), k, sep='\t')
print(dict(agree))
