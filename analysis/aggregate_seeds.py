#!/usr/bin/env python3
"""
Aggregate multi-seed head results.

Expects one directory per seed, each containing variant_B.txt and variant_C.txt
in TEMPURA's evaluator output format, e.g.

    ~/results_seed0/variant_B.txt
    ~/results_seed0/variant_C.txt
    ~/results_seed1/variant_B.txt
    ...

Reports mean and standard deviation per variant, and a paired t-test of C - B
across shared seeds (the correct test for this comparison, since both variants
see the same seed and the paired difference removes the shared component of
variance).

Usage:
    python3 aggregate_seeds.py ~/results_seed*
"""
import sys, os, re, math, glob

FINE = ['drinking_from', 'eating', 'wiping', 'writing_on', 'twisting']

# two-sided critical values of Student's t at alpha = 0.05
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def parse(path, mode='with constraint', k=20):
    """Return (macro, {predicate: value}) for the requested mode and K."""
    if not os.path.exists(path):
        return None, None
    text = open(path).read()
    blocks = re.split(r'-+\s*(with constraint|no constraint)\s*-+', text)
    # blocks = ['', mode1, body1, mode2, body2, ...]
    body = None
    for i in range(1, len(blocks) - 1, 2):
        if blocks[i].strip() == mode:
            body = blocks[i + 1]
            break
    if body is None:
        return None, None
    macro, per = None, {}
    current = None
    for line in body.splitlines():
        m = re.match(r'\s*mR@(\d+):\s*([0-9.]+)', line)
        if m:
            current = int(m.group(1))
            if current == k:
                macro = float(m.group(2))
            continue
        m = re.match(r'\s+(\w+):\s*([0-9.]+)', line)
        if m and current == k:
            per[m.group(1)] = float(m.group(2))
    return macro, per


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    if len(xs) < 2:
        return float('nan')
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def paired_t(a, b):
    """Paired t statistic and 95% verdict for a - b."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return None
    s = sd(d)
    if s == 0:
        return dict(diff=mean(d), t=float('inf'), n=n, sig=True)
    t = mean(d) / (s / math.sqrt(n))
    crit = T_CRIT_95.get(n - 1)
    return dict(diff=mean(d), sd=s, t=t, n=n,
                crit=crit, sig=(crit is not None and abs(t) > crit))


def main():
    dirs = sys.argv[1:]
    if not dirs:
        dirs = sorted(glob.glob(os.path.expanduser('~/results_seed*')))
    if not dirs:
        print('no seed directories given or found')
        sys.exit(1)

    data = {'B': [], 'C': []}
    fine = {'B': [], 'C': []}
    used = []
    for d in dirs:
        ok = True
        row = {}
        for v in ('B', 'C'):
            macro, per = parse(os.path.join(d, f'variant_{v}.txt'))
            if macro is None:
                ok = False
                break
            row[v] = (macro, per)
        if not ok:
            print(f'  skipping {d} (incomplete)')
            continue
        used.append(d)
        for v in ('B', 'C'):
            macro, per = row[v]
            data[v].append(macro)
            vals = [per[f] for f in FINE if f in per]
            fine[v].append(sum(vals) / len(vals) if vals else float('nan'))

    n = len(used)
    print(f'\n{n} seeds aggregated: ' + ', '.join(os.path.basename(d) for d in used))
    if n == 0:
        sys.exit(1)

    print('\nmacro mR@20 (with constraint)')
    for v in ('B', 'C'):
        print(f'  variant {v}: {mean(data[v]):.4f} +/- {sd(data[v]):.4f}')
    print('\nfine-5 mean mR@20')
    for v in ('B', 'C'):
        print(f'  variant {v}: {mean(fine[v]):.4f} +/- {sd(fine[v]):.4f}')

    if n >= 2:
        for name, a, b in (('macro', data['C'], data['B']),
                           ('fine-5', fine['C'], fine['B'])):
            r = paired_t(a, b)
            print(f'\npaired t-test on {name}, C - B, n={r["n"]}')
            print(f'  mean difference : {r["diff"]:+.4f}')
            print(f'  sd of differences: {r.get("sd", float("nan")):.4f}')
            print(f'  t = {r["t"]:.3f}   (|t| > {r["crit"]} needed at alpha=0.05, df={r["n"]-1})')
            print('  verdict: ' + ('difference is significant'
                                   if r['sig'] else
                                   'not distinguishable from zero'))
    else:
        print('\nneed at least two seeds for the paired test')


if __name__ == '__main__':
    main()
