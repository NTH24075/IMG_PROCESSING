"""Benchmark B: write robustness_dev.csv / robustness_test.csv. Images are generated in memory at eval time."""
from object_counting.io import ROOT, read_csv, write_csv
from object_counting.degrade import GRID, seed_for


def expand(rows):
    out = []
    for r in rows:
        for kind, levels in [('none', [0])]+list(GRID.items()):
            for level in levels:
                image_id = r['image_id'] if kind == 'none' else f"{r['image_id']}__{kind}_{level}"
                # Degraded copies keep the split of their source image.
                out.append(dict(r, image_id=image_id, source_image_id=r['image_id'], kind=kind, level=level,
                                seed=seed_for(image_id), condition=f'{kind}_{level}' if kind != 'none' else r.get('condition', 'original')))
    return out


def main(root=ROOT):
    folder = root/'data/week1/manifests'
    for name, sources in [('robustness_dev', ['dev']), ('robustness_test', ['test', 'legacy_holdout'])]:
        rows = expand([r for s in sources for r in read_csv(folder/f'{s}.csv')])
        write_csv(folder/f'{name}.csv', rows, rows[0].keys())
        print(name, len(rows))


if __name__ == '__main__':
    main()
