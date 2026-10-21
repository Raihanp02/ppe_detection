"""Remove pixel-identical PPE crops with identical labels, preferring held-out splits."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import cv2
import yaml

if __package__:
    from .filter_person_dataset import relative_file
else:
    from filter_person_dataset import relative_file


def fingerprint(root, record):
    path = relative_file(root, record['image'])
    image = cv2.imread(str(root/path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'Unreadable image: {path}')
    if record.get('mask'):
        raise ValueError('This tool is for background-retaining crops without masks')
    label = path.parent.parent/'labels'/path.with_suffix('.txt').name
    relative_file(root, label)
    labels = frozenset(tuple(map(float, row.split())) for row in (root/label).read_text().splitlines() if row.strip())
    digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
    return digest, record, label, labels


def select_duplicates(entries):
    groups = defaultdict(list)
    for entry in entries:
        groups[entry[0]].append(entry)
    duplicates, conflicts = [], []
    priority = {'test': 0, 'val': 1, 'train': 2}
    for digest, group in groups.items():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda e: (priority[Path(e[1]['image']).parts[0]], e[1]['image']))
        keep = ordered[0]
        if any(entry[3] != keep[3] for entry in ordered[1:]):
            conflicts.append(dict(sha256=digest, images=[e[1]['image'] for e in ordered]))
            continue
        for remove in ordered[1:]:
            duplicates.append((keep, remove))
    return duplicates, conflicts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    root = args.dataset.resolve()
    report_path = root/'image_deduplication.json'
    if report_path.exists():
        raise FileExistsError(report_path)
    records = [json.loads(line) for line in (root/'persons.jsonl').read_text().splitlines() if line.strip()]
    if len({r['image'] for r in records}) != len(records):
        raise ValueError('Duplicate manifest paths')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        entries = list(pool.map(lambda r: fingerprint(root, r), records))
    duplicates, conflicts = select_duplicates(entries)
    if conflicts:
        (root/'duplicate_annotation_conflicts.json').write_text(json.dumps(conflicts, indent=2)+'\n')
        raise ValueError('Identical crops have different labels; review duplicate_annotation_conflicts.json before removal')
    removed_paths = {remove[1]['image'] for keep, remove in duplicates}
    by_image = {r['image']: r for r in records}
    removed_counts = Counter()
    removed_classes = defaultdict(Counter)
    names = yaml.safe_load((root/'data.yaml').read_text())['names']
    log = []
    for keep, remove in duplicates:
        # Hash equality is checked against decoded pixel equality before any deletion.
        first = cv2.imread(str(root/keep[1]['image']), cv2.IMREAD_UNCHANGED)
        second = cv2.imread(str(root/remove[1]['image']), cv2.IMREAD_UNCHANGED)
        if first.shape != second.shape or not (first == second).all():
            raise ValueError('Fingerprint collision or concurrently changed image')
        split = Path(remove[1]['image']).parts[0]
        removed_counts[split] += 1
        removed_classes[split].update(names[int(row[0])] for row in remove[3])
        log.append(dict(sha256=keep[0], kept_image=keep[1]['image'], removed_record=remove[1]))
        by_image[keep[1]['image']].setdefault('duplicate_origins', []).append(remove[1])
    retained = [r for r in records if r['image'] not in removed_paths]
    report = dict(status='prepared', rule='Exact decoded pixels AND identical numeric label sets; prefer test, then val, then train',
                  before=len(records), kept=len(retained), removed=len(duplicates), removed_by_split=dict(removed_counts), duplicates=log)
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    # All paths were validated beneath this dataset; only duplicate files are removed.
    for keep, remove in duplicates:
        (root/remove[1]['image']).unlink()
        (root/remove[2]).unlink()
    temporary = root/'persons.jsonl.dedup.tmp'
    temporary.write_text(''.join(json.dumps(r)+'\n' for r in retained), encoding='utf-8')
    temporary.replace(root/'persons.jsonl')
    summary_path = root/'filter_summary.json'
    summary = json.loads(summary_path.read_text())
    summary['kept_before_image_deduplication'] = summary['kept']
    summary['kept'] = len(retained)
    summary['duplicate_images_removed'] = len(duplicates)
    for split, stats in summary['counts'].items():
        stats['duplicate_images_removed'] = removed_counts[split]
        stats['kept'] -= removed_counts[split]
        for name, count in removed_classes[split].items():
            summary['class_boxes'][split]['kept'][name] -= count
        summary['class_boxes'][split]['duplicate_images_removed'] = dict(removed_classes[split])
    summary_path.write_text(json.dumps(summary, indent=2)+'\n')
    if (root/'verification.json').exists():
        (root/'verification.json').rename(root/'verification_before_image_deduplication.json')
    for split in ('train', 'val', 'test'):
        cache = root/split/'labels.cache'
        if cache.is_file():
            cache.unlink()
        expected = sum(Path(r['image']).parts[0] == split for r in retained)
        for directory in ('images', 'labels'):
            assert len(list((root/split/directory).iterdir())) == expected
    report['status'] = 'completed'
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    with (root/'README.md').open('a') as readme:
        readme.write('\nExact duplicate crops with identical labels were removed across splits. '
                      'Test copies take precedence over validation copies, then training copies. '
                      'image_deduplication.json preserves removed records and keeper mappings. '
                      'This does not detect resized, shifted, recompressed, or other near-duplicate images.\n')
    print(json.dumps({k:v for k,v in report.items() if k!='duplicates'}, indent=2))


if __name__ == '__main__':
    main()
