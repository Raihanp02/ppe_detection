"""Filter PPE person crops using native mask sizes or rectangular image sizes."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

import cv2
import yaml


def relative_file(root, value):
    relative = Path(value)
    if relative.is_absolute() or not (root / relative).resolve().is_relative_to(root):
        raise ValueError(f'Path must stay within source dataset: {value}')
    if not (root / relative).is_file():
        raise FileNotFoundError(root / relative)
    return relative


def measure(root, record, size_mode='mask'):
    image_path = relative_file(root, record['image'])
    mask_path = relative_file(root, record['mask']) if record.get('mask') else None
    label_path = image_path.parent.parent / 'labels' / image_path.with_suffix('.txt').name
    relative_file(root, label_path)
    if size_mode == 'crop':
        image = cv2.imread(str(root / image_path))
        if image is None:
            raise ValueError(f'Cannot read crop: {image_path}')
        height, width = image.shape[:2]
        metrics = dict(width=width, height=height, crop_pixels=width*height)
    else:
        if mask_path is None:
            raise ValueError('Mask missing; use --size-mode crop for background-retaining crops')
        mask = cv2.imread(str(root / mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f'Cannot read mask: {mask_path}')
        _, _, width, height = cv2.boundingRect(mask)
        metrics = dict(width=width, height=height, foreground_pixels=cv2.countNonZero(mask))
    return record, image_path, mask_path, label_path, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing folders are not overwritten')
    parser.add_argument('--min-width', type=int, default=32)
    parser.add_argument('--min-height', type=int, default=96)
    parser.add_argument('--min-foreground', type=int, default=2048)
    parser.add_argument('--size-mode', choices=['mask', 'crop'], default='mask')
    parser.add_argument('--min-area', type=int, default=2048, help='Minimum rectangular pixel area in crop mode')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if min(args.min_width, args.min_height, args.min_foreground, args.min_area, args.workers) < 1:
        parser.error('Thresholds and workers must be positive')
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(source):
        parser.error('Output must be a new directory outside the source dataset')
    config = yaml.safe_load((source / 'data.yaml').read_text())
    records = [json.loads(line) for line in (source / 'persons.jsonl').read_text().splitlines() if line.strip()]
    exported = [r for r in records if r.get('image')]
    thresholds = dict(width=args.min_width, height=args.min_height)
    thresholds.update(dict(crop_pixels=args.min_area) if args.size_mode == 'crop' else dict(foreground_pixels=args.min_foreground))
    counts, class_counts = {}, {}
    # Complete source validation/measurement before creating the destination.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        measured = list(pool.map(lambda r: measure(source, r, args.size_mode), exported))
    has_masks = any(entry[2] is not None for entry in measured)
    output.mkdir(parents=True)
    for split in ('train', 'val', 'test'):
        counts[split] = Counter()
        class_counts[split] = dict(before=Counter(), kept=Counter(), removed=Counter())
        for directory in (('images', 'labels', 'masks') if has_masks else ('images', 'labels')):
            (output / split / directory).mkdir(parents=True)
    seen = set()
    with (output / 'persons.jsonl').open('w', encoding='utf-8') as kept_file, (output / 'removed.jsonl').open('w', encoding='utf-8') as removed_file:
        for record, image_path, mask_path, label_path, metrics in measured:
            if image_path in seen:
                raise ValueError(f'Duplicate image: {image_path}')
            seen.add(image_path)
            split = image_path.parts[0]
            if split not in counts:
                raise ValueError(f'Unsupported split: {split}')
            reasons = [f'{key}_below_{limit}' for key, limit in thresholds.items() if metrics[key] < limit]
            status = 'removed' if reasons else 'kept'
            counts[split]['before'] += 1
            counts[split][status] += 1
            classes = Counter(int(row.split()[0]) for row in (source / label_path).read_text().splitlines() if row.strip())
            class_counts[split]['before'].update(classes)
            class_counts[split][status].update(classes)
            if reasons:
                removed_file.write(json.dumps({**record, 'size_filter': dict(metrics=metrics, reasons=reasons)}) + '\n')
            else:
                for relative in (image_path, mask_path, label_path):
                    if relative is not None:
                        shutil.copy2(source / relative, output / relative)
                kept_file.write(json.dumps({**record, 'size_filter': dict(metrics=metrics, status='kept')}) + '\n')
    names = config['names']
    result_config = dict(path=output.as_posix(), train='train/images', val='val/images', test='test/images', nc=len(names), names=names)
    (output / 'data.yaml').write_text(yaml.safe_dump(result_config, sort_keys=False))
    summary = dict(source=source.as_posix(), thresholds=thresholds, size_mode=args.size_mode,
                   rule='Remove if ANY native-resolution size measurement is below its threshold',
                   counts={s: dict(c) for s, c in counts.items()},
                   class_boxes={s: {status: {names[k]: v for k, v in sorted(c.items())} for status, c in stats.items()} for s, stats in class_counts.items()},
                   source_records_without_images=len(records)-len(exported),
                   kept=sum(c['kept'] for c in counts.values()), removed=sum(c['removed'] for c in counts.values()))
    # Verify copied bytes, pairing, and counts. Labels and masks must stay unchanged.
    for record, image_path, mask_path, label_path, metrics in measured:
        if any(metrics[key] < limit for key, limit in thresholds.items()):
            assert not (output / image_path).exists()
            continue
        for relative in (image_path, mask_path, label_path):
            if relative is not None and (source / relative).read_bytes() != (output / relative).read_bytes():
                raise ValueError(f'Copy verification failed: {relative}')
    for split, count in counts.items():
        for directory in ('images', 'labels'):
            assert sum(1 for p in (output / split / directory).iterdir() if p.is_file()) == count['kept']
        if has_masks:
            expected = sum(1 for _, image, mask, _, metrics in measured if image.parts[0] == split and mask is not None and all(metrics[k] >= v for k, v in thresholds.items()))
            assert sum(1 for p in (output / split / 'masks').iterdir() if p.is_file()) == expected
    summary['copy_verification'] = 'passed: all retained images, masks and labels are byte-identical to source'
    (output / 'filter_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for original, retained in [('summary.json', 'conversion_summary.json'), ('review.jsonl', 'conversion_review.jsonl')]:
        if (source / original).is_file():
            shutil.copy2(source / original, output / retained)
    with (output / 'unmatched_persons.jsonl').open('w', encoding='utf-8') as unmatched:
        for record in records:
            if not record.get('image'):
                unmatched.write(json.dumps(record) + '\n')
    (output / 'README.md').write_text(
        '# Size-filtered PPE person dataset\n\n'
        f'Source: {source.as_posix()}\n\n'
        f'Measurement mode: {args.size_mode}. Thresholds: {thresholds}. A person must pass all three. '
        'Crop mode measures image dimensions and rectangular area, not foreground pixels.\n\n'
        'Images, masks, class IDs and PPE boxes are unchanged. Original splits are preserved. '
        'persons.jsonl contains retained people; removed.jsonl records rejected people and reasons, '
        'with image and mask paths relative to the SOURCE dataset. '
        'filter_summary.json includes class counts before and after filtering. '
        'Conversion provenance and review records are copied to conversion_summary.json and '
        'conversion_review.jsonl when available. unmatched_persons.jsonl records people not exported '
        'during conversion. If the intermediate source dataset is removed, rejected crop/mask paths '
        'are historical references; source_image and source_label still identify original full-frame data.\n\n'
        'These thresholds are heuristics, not validated accuracy cutoffs. '
        'Scores on the filtered validation/test splits only describe people passing this size filter; '
        'also evaluate on the original test set when tiny people occur in deployment. '
        'Existing model-generated mask and PPE-assignment limitations still apply.\n'
    )
    print(json.dumps(dict(counts=summary['counts'], kept=summary['kept'], removed=summary['removed'], verification=summary['copy_verification']), indent=2))


if __name__ == '__main__':
    main()
