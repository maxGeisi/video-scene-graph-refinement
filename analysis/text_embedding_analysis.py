#!/usr/bin/env python3
"""
Scale-free comparison of child-to-parent proximity in two embedding spaces.

Reproduces Figure 9b and Table 5 of the report.

Raw cosine distances from different embedding spaces are not commensurable, so
this script reports two statistics that are invariant to the scale of a space:

  1. the RANK of the coarse parent among the other predicates, ordered by
     distance from the fine predicate;
  2. the child-parent distance DIVIDED BY the mean pairwise distance of the
     same space.

Inputs
------
  --visual  CSV or JSON with the pairwise distance matrix of the relation
            features. If omitted, the matrix measured in this project is used.
  --out     output directory (default: current)

Requires gensim for the pretrained GloVe vectors (downloaded on first run).

Usage
-----
    python3 text_embedding_analysis.py --out results/
"""
import argparse, itertools, json, os
import numpy as np

PREDICATES = ['drinking_from', 'eating', 'wiping', 'writing_on', 'twisting',
              'holding', 'touching']

# strongest coarse parent per fine predicate, as measured by co-occurrence on
# the training split (see motion_diagnostics.py / coarse_to_fine_mapping.json)
PARENT = {'drinking_from': 'holding',
          'eating':        'holding',
          'wiping':        'holding',
          'writing_on':    'touching',
          'twisting':      'holding'}

# Pairwise cosine distances between mean TEMPURA relation features, measured in
# this project. Order follows PREDICATES.
VISUAL_DEFAULT = np.array([
    [0.00, 0.20, 0.30, 0.41, 0.28, 0.17, 0.32],
    [0.20, 0.00, 0.46, 0.53, 0.29, 0.18, 0.40],
    [0.30, 0.46, 0.00, 0.23, 0.15, 0.16, 0.10],
    [0.41, 0.53, 0.23, 0.00, 0.23, 0.31, 0.21],
    [0.28, 0.29, 0.15, 0.23, 0.00, 0.08, 0.12],
    [0.17, 0.18, 0.16, 0.31, 0.08, 0.00, 0.10],
    [0.32, 0.40, 0.10, 0.21, 0.12, 0.10, 0.00],
])


def text_distance_matrix(names, model_name='glove-wiki-gigaword-100'):
    """Cosine distance matrix of predicate names in a pretrained text space."""
    import gensim.downloader as api
    model = api.load(model_name)

    def embed(name):
        words = [w for w in name.split('_') if w in model]
        if not words:
            raise ValueError(f'no vocabulary entry for any token of "{name}"')
        v = np.mean([model[w] for w in words], axis=0)
        return v / np.linalg.norm(v)

    emb = {n: embed(n) for n in names}
    return np.array([[1.0 - float(np.dot(emb[a], emb[b])) for b in names]
                     for a in names])


def mean_pairwise(matrix):
    n = len(matrix)
    return float(np.mean([matrix[i, j] for i, j in itertools.combinations(range(n), 2)]))


def summarise(matrix, names, parent_of):
    """Rank of the parent and normalised child-parent distance, per fine predicate."""
    idx = {p: i for i, p in enumerate(names)}
    mp = mean_pairwise(matrix)
    out = {}
    for fine, parent in parent_of.items():
        i = idx[fine]
        ordering = [names[j] for j in np.argsort(matrix[i]) if j != i]
        out[fine] = {
            'parent': parent,
            'rank': ordering.index(parent) + 1,
            'distance': float(matrix[i, idx[parent]]),
            'normalised': float(matrix[i, idx[parent]] / mp),
        }
    return out, mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--visual', default=None,
                    help='JSON file with a pairwise distance matrix; uses the '
                         'measured matrix if omitted')
    ap.add_argument('--out', default='.')
    ap.add_argument('--model', default='glove-wiki-gigaword-100')
    args = ap.parse_args()

    V = VISUAL_DEFAULT
    if args.visual:
        V = np.array(json.load(open(args.visual)))

    T = text_distance_matrix(PREDICATES, args.model)

    vis, mv = summarise(V, PREDICATES, PARENT)
    txt, mt = summarise(T, PREDICATES, PARENT)

    print(f'mean pairwise distance   visual={mv:.3f}   text={mt:.3f}\n')
    header = f'{"fine":15s} {"parent":9s} | {"rank":>4} {"d/mean":>7} | {"rank":>4} {"d/mean":>7}'
    print(header)
    print('-' * len(header))
    for f in PARENT:
        print(f'{f:15s} {PARENT[f]:9s} | {vis[f]["rank"]:4d} {vis[f]["normalised"]:7.2f} '
              f'| {txt[f]["rank"]:4d} {txt[f]["normalised"]:7.2f}')

    nv = [vis[f]['normalised'] for f in PARENT]
    nt = [txt[f]['normalised'] for f in PARENT]
    n_first_v = sum(1 for f in PARENT if vis[f]['rank'] == 1)
    n_first_t = sum(1 for f in PARENT if txt[f]['rank'] == 1)
    print(f'\nmean normalised child-parent distance: visual={np.mean(nv):.2f}  text={np.mean(nt):.2f}')
    print(f'parent is nearest neighbour: visual {n_first_v}/{len(PARENT)}   '
          f'text {n_first_t}/{len(PARENT)}')

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'embedding_comparison.json')
    json.dump({'predicates': PREDICATES,
               'visual_matrix': V.tolist(),
               'text_matrix': T.tolist(),
               'visual': vis, 'text': txt,
               'mean_pairwise': {'visual': mv, 'text': mt}},
              open(path, 'w'), indent=1)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
