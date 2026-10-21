"""Convert full-frame YOLO PPE annotations to person crops and attributes."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import cv2
import yaml


def deduplicate_labels(rows, source_lines):
    """Keep the first identical numeric class/box row and its provenance."""
    if len(rows) != len(source_lines):
        raise ValueError('Label rows and source line references must align')
    seen, unique_rows, unique_sources, removed_sources = set(), [], [], []
    for row, source_line in zip(rows, source_lines):
        key = tuple(map(float, row.split()))
        if key in seen:
            removed_sources.append(source_line)
        else:
            seen.add(key)
            unique_rows.append(row)
            unique_sources.append(source_line)
    return unique_rows, unique_sources, removed_sources


def resolve_device(requested):
    if str(requested).lower() != 'auto':
        return requested
    import torch
    return '0' if torch.cuda.is_available() else 'cpu'


def segmented_image(crop, mask):
    """Clear RGB as well as alpha so RGB-only training sees black background."""
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[mask == 0, :3] = 0
    rgba[:, :, 3] = mask
    return rgba


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return intersection / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-intersection)


def match_masks(result, people, width, height, min_iou):
    """Conservative one-to-one matching; retain competing matches for review."""
    if result.masks is None:
        return {}, set()
    boxes = result.boxes.xyxy.cpu().numpy()
    pairs = sorted(((iou(p['box'], (b[0]/width, b[1]/height, b[2]/width, b[3]/height)), i, j)
                    for i, p in enumerate(people) for j, b in enumerate(boxes)), reverse=True)
    matches, used, uncertain = {}, set(), set()
    for score, i, j in pairs:
        if score < min_iou:
            break
        if i in matches or j in used:
            uncertain.add(i)
            uncertain.update(k for k, value in matches.items() if value[0] == j)
            continue
        matches[i] = (j, float(score))
        used.add(j)
    return matches, uncertain


def read_labels(path, names):
    labels = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = [float(v) for v in line.split()]
        cls, coords = values[0], values[1:]
        if not all(math.isfinite(v) for v in values) or cls != int(cls) or not 0 <= cls < len(names):
            raise ValueError(f"Invalid class/coordinate: {path}:{line_number}")
        polygon = len(coords) != 4
        if not polygon:
            x, y, w, h = coords
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid box: {path}:{line_number}")
            box = (x-w/2, y-h/2, x+w/2, y+h/2)
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            box = (min(coords[::2]), min(coords[1::2]), max(coords[::2]), max(coords[1::2]))
        else:
            raise ValueError(f"Invalid annotation: {path}:{line_number}")
        box = tuple(max(0.0, min(1.0, v)) for v in box)
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Empty box: {path}:{line_number}")
        labels.append(dict(line=line_number, class_id=int(cls), name=names[int(cls)], box=box,
                           polygon_to_box=polygon))
    return labels


def coverage(item, person):
    x1, y1, x2, y2 = item
    a, b, c, d = person
    intersection = max(0, min(x2, c)-max(x1, a))*max(0, min(y2, d)-max(y1, b))
    return intersection / ((x2-x1)*(y2-y1))


def assign_items(items, people, threshold=0.7, margin=0.15):
    assigned = [[] for _ in people]
    unresolved = []
    for item in items:
        scores = sorted(((coverage(item['box'], p['box']), i) for i, p in enumerate(people)), reverse=True)
        candidates = [i for score, i in scores if score > 0]
        if not scores or scores[0][0] < threshold:
            unresolved.append(dict(**item, reason='insufficient_person_overlap', candidates=candidates))
        elif len(scores) > 1 and scores[1][0] > 0 and scores[0][0]-scores[1][0] < margin:
            unresolved.append(dict(**item, reason='ambiguous_person', candidates=candidates))
        else:
            assigned[scores[0][1]].append(item)
    return assigned, unresolved


def attributes(items):
    names = {item['name'] for item in items}
    result = {}
    for attr in ('boots', 'gloves', 'hardhat', 'vest'):
        positive, negative = attr in names, f'no_{attr}' in names
        result[attr] = 'conflict' if positive and negative else 'present' if positive else 'absent' if negative else 'unknown'
    return result


def crop_labels(items, crop_xyxy, image_shape, class_map, min_visibility=0.0):
    """Include PPE when intersection/original PPE area meets min_visibility."""
    left, top, right, bottom = crop_xyxy
    height, width = image_shape[:2]
    cw, ch = right-left, bottom-top
    rows, kept = [], []
    for item in items:
        a, b, c, d = item['box']
        original_area = (c-a)*width*(d-b)*height
        a, b, c, d = max(0, a*width-left), max(0, b*height-top), min(cw, c*width-left), min(ch, d*height-top)
        # Exclude empty intersections and numerical noise at a touching edge.
        if c-a <= 1e-6 or d-b <= 1e-6:
            continue
        visible_fraction = (c-a)*(d-b)/original_area
        if visible_fraction + 1e-12 < min_visibility:
            continue
        coords = [(a+c)/2/cw, (b+d)/2/ch, (c-a)/cw, (d-b)/ch]
        if round(coords[2], 8) == 0 or round(coords[3], 8) == 0:
            continue
        rows.append(f"{class_map[item['class_id']]} " + ' '.join(f'{v:.8f}' for v in coords))
        kept.append(item)
    return rows, kept


def resolve_images(config_path, config, key):
    raw = Path(config[key])
    base = Path(config.get('path', '.'))
    if not base.is_absolute():
        base = config_path.parent / base
    standard = (base / raw).resolve()
    if standard.is_dir():
        return standard
    # Roboflow exports sometimes use ../train although train is beside data.yaml.
    local = config_path.parent / raw.parent.name / raw.name
    if local.is_dir():
        return local.resolve()
    raise FileNotFoundError(f"Cannot resolve {key}: {raw}")


def convert(args):
    source = args.data.resolve()
    config = yaml.safe_load(source.read_text())
    names = config['names']
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    person_id = names.index('person')
    ppe_names = [n for n in names if n != 'person']
    class_map = {i: ppe_names.index(n) for i, n in enumerate(names) if n != 'person'}
    splits = {key: resolve_images(source, config, key) for key in ('train', 'val', 'test') if config.get(key)}
    model = None
    label_policy = args.label_policy
    if label_policy == 'auto':
        label_policy = 'assigned' if args.seg_model else 'visible'
    if args.seg_model and label_policy == 'visible':
        raise ValueError('Visible-box export requires background-retaining crops; omit --seg-model')
    device = resolve_device(args.device) if args.seg_model else None
    if args.seg_model:
        from ultralytics import YOLO
        if not args.seg_model.is_file():
            raise FileNotFoundError(args.seg_model)
        model = YOLO(str(args.seg_model), task='segment')
        if model.task != 'segment':
            raise ValueError('A segmentation model is required')
        model_person_id = next(i for i, name in model.names.items() if name == 'person')
        print(f'Segmentation device: {device} (requested: {args.device})', flush=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = Counter()
    with (output/'persons.jsonl').open('w', encoding='utf-8') as manifest, (output/'review.jsonl').open('w', encoding='utf-8') as review:
        for split, images_dir in splits.items():
            (output/split/'images').mkdir(parents=True)
            (output/split/'labels').mkdir()
            if model:
                (output/split/'masks').mkdir()
            paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
            if args.limit:
                paths = paths[:args.limit]
            for image_path in paths:
                label_path = images_dir.parent/'labels'/f'{image_path.stem}.txt'
                if not label_path.exists():
                    raise FileNotFoundError(label_path)
                labels = read_labels(label_path, names)
                people = [a for a in labels if a['class_id'] == person_id]
                items = [a for a in labels if a['class_id'] != person_id]
                if label_policy == 'assigned':
                    assigned, unresolved = assign_items(items, people, args.min_overlap, args.ambiguity_margin)
                else:
                    assigned, unresolved = [[] for _ in people], []
                summary[f'{split}_frames'] += 1
                summary['polygon_labels_converted_to_boxes'] += sum(a['polygon_to_box'] for a in labels)
                summary['frames_without_person'] += not people
                summary['unresolved_ppe_labels'] += len(unresolved)
                if unresolved or not people:
                    review.write(json.dumps(dict(split=split, source_image=str(image_path), no_person=not people, unresolved=unresolved))+'\n')
                if not people:
                    continue
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f'Cannot read image: {image_path}')
                h, w = image.shape[:2]
                matches, uncertain = {}, set()
                result = None
                if model:
                    result = model.predict(image, classes=[model_person_id], conf=args.seg_conf,
                                           device=device, retina_masks=True, verbose=False)[0]
                    matches, uncertain = match_masks(result, people, w, h, args.mask_min_iou)
                for index, (person, owned) in enumerate(zip(people, assigned)):
                    x1, y1, x2, y2 = person['box']
                    dx, dy = (x2-x1)*args.padding, (y2-y1)*args.padding
                    left, top = max(0, math.floor((x1-dx)*w)), max(0, math.floor((y1-dy)*h))
                    right, bottom = min(w, math.ceil((x2+dx)*w)), min(h, math.ceil((y2+dy)*h))
                    cw, ch = right-left, bottom-top
                    stem = f'{image_path.name}__person_{index:03d}'
                    rel_image = Path(split)/'images'/f'{stem}.png'
                    mask = None
                    rows, kept = crop_labels(items if label_policy == 'visible' else owned,
                                             [left, top, right, bottom], image.shape, class_map,
                                             args.min_ppe_visibility if label_policy == 'visible' else 0.0)
                    below_visibility = []
                    if label_policy == 'visible' and args.min_ppe_visibility > 0:
                        _, intersecting = crop_labels(items, [left, top, right, bottom], image.shape, class_map)
                        included_lines = {item['line'] for item in kept}
                        below_visibility = [item['line'] for item in intersecting if item['line'] not in included_lines]
                        summary['ppe_labels_below_visibility'] += len(below_visibility)
                    attr = attributes(kept)
                    issues = [u for u in unresolved if index in u['candidates']]
                    record = dict(image=rel_image.as_posix(), source_image=str(image_path), source_label=str(label_path),
                                  person_index=index, person_label_line=person['line'], crop_xyxy=[left, top, right, bottom],
                                  attributes=attr, attribute_scope='crop' if label_policy == 'visible' else 'person',
                                  label_policy=label_policy, assigned_label_lines=[i['line'] for i in kept],
                                  visibility_rejected_label_lines=below_visibility,
                                  needs_review=bool(issues) or 'conflict' in attr.values(),
                                  unresolved_label_lines=[i['line'] for i in issues])
                    if model:
                        record.update(mask=None, mask_origin='model_prediction', mask_status='unmatched',
                                      mask_model=str(args.seg_model.resolve()))
                        if index in matches:
                            mask_index, score = matches[index]
                            mask = result.masks.data[mask_index].cpu().numpy()
                            if mask.shape != (h, w):
                                raise ValueError(f'Mask dimensions {mask.shape} do not match image {(h, w)}')
                            mask = (mask[top:bottom, left:right] > 0.5).astype('uint8') * 255
                            if mask.any():
                                rel_mask = Path(split)/'masks'/f'{stem}.png'
                                if not cv2.imwrite(str(output/rel_mask), mask):
                                    raise OSError(f'Cannot write {rel_mask}')
                                record.update(mask=rel_mask.as_posix(), mask_status='needs_review',
                                              mask_match_iou=score, mask_confidence=float(result.boxes.conf[mask_index]),
                                              mask_ambiguous=index in uncertain)
                                summary['persons_with_masks'] += 1
                        if record['mask'] is None:
                            summary['persons_without_masks'] += 1
                            review.write(json.dumps(dict(split=split, source_image=str(image_path), person_index=index,
                                                         reason='no_matching_person_mask'))+'\n')
                        # Predicted masks are never presented as human-verified segmentation labels.
                        record['needs_review'] = True
                    if model and record['mask'] is None:
                        record['image'] = None
                        record['export_status'] = 'skipped_no_mask'
                    else:
                        crop = image[top:bottom, left:right]
                        if model:
                            crop = segmented_image(crop, mask)
                            # Do not train on a PPE box whose pixels were entirely erased.
                            erased, kept_rows, visible_items = [], [], []
                            for item, row in zip(kept, rows):
                                a, b, c, d = item['box']
                                a, b = max(0, math.floor(a*w-left)), max(0, math.floor(b*h-top))
                                c, d = min(cw, math.ceil(c*w-left)), min(ch, math.ceil(d*h-top))
                                if mask[b:d, a:c].any():
                                    kept_rows.append(row)
                                    visible_items.append(item)
                                else:
                                    erased.append(item['line'])
                            rows = kept_rows
                            record.update(source_attributes=attr, attributes=attributes(visible_items),
                                          assigned_label_lines=[i['line'] for i in visible_items],
                                          erased_ppe_label_lines=erased)
                            summary['ppe_labels_erased_by_mask'] += len(erased)
                            if erased:
                                review.write(json.dumps(dict(split=split, source_image=str(image_path), person_index=index,
                                                             reason='ppe_erased_by_mask', label_lines=erased))+'\n')
                        rows, source_lines, duplicates = deduplicate_labels(rows, record['assigned_label_lines'])
                        record['assigned_label_lines'] = source_lines
                        record['duplicate_label_lines_removed'] = duplicates
                        summary['duplicate_ppe_labels_removed'] += len(duplicates)
                        if not cv2.imwrite(str(output/rel_image), crop):
                            raise OSError(f'Cannot write {rel_image}')
                        (output/split/'labels'/f'{stem}.txt').write_text('\n'.join(rows)+ ('\n' if rows else ''))
                        record['export_status'] = 'exported'
                        summary['exported_images'] += 1
                        summary['exported_ppe_labels'] += len(rows)
                    manifest.write(json.dumps(record)+'\n')
                    summary[f'{split}_persons'] += 1
                    summary['assigned_ppe_labels'] += len(kept)
                    summary['persons_needing_review'] += record['needs_review']
                if summary[f'{split}_frames'] % 100 == 0:
                    print(f"{split}: {summary[f'{split}_frames']}/{len(paths)} frames", flush=True)
            print(f"{split}: {summary[f'{split}_persons']} person crops", flush=True)
    export = dict(path=output.as_posix(), **{s: f'{s}/images' for s in splits}, nc=len(ppe_names), names=ppe_names)
    (output/'data.yaml').write_text(yaml.safe_dump(export, sort_keys=False))
    report = dict(source=str(source), settings=dict(padding=args.padding, min_overlap=args.min_overlap,
                  ambiguity_margin=args.ambiguity_margin, limit_per_split=args.limit,
                  seg_model=str(args.seg_model.resolve()) if args.seg_model else None,
                  mask_min_iou=args.mask_min_iou, seg_conf=args.seg_conf,
                  requested_device=args.device, resolved_device=device, label_policy=label_policy,
                  min_ppe_visibility=args.min_ppe_visibility if label_policy == 'visible' else None), counts=dict(summary),
                  source_metadata=config.get('roboflow', {}))
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['counts'], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing output is never overwritten')
    parser.add_argument('--padding', type=float, default=0.05, help='Fraction of person box added on each side')
    parser.add_argument('--min-overlap', type=float, default=0.7, help='Assigned policy only: minimum fraction of PPE area inside person')
    parser.add_argument('--ambiguity-margin', type=float, default=0.15, help='Assigned policy only: minimum gap between best and runner-up person overlap')
    parser.add_argument('--label-policy', choices=['auto', 'visible', 'assigned'], default='auto',
                        help='auto: PPE meeting visibility threshold for background crops, exclusive assignment for segmentation')
    parser.add_argument('--min-ppe-visibility', type=float, default=0.4,
                        help='Visible policy: minimum fraction of original PPE box area inside crop (default 0.4); 0 includes any positive intersection')
    parser.add_argument('--limit', type=int, default=0, help='Preview: maximum frames per split; zero means all')
    parser.add_argument('--seg-model', type=Path, help='Local YOLO segmentation weights; exports binary person masks')
    parser.add_argument('--device', default='auto', help='auto (default): CUDA GPU 0 if available, otherwise CPU; explicit cpu or GPU index also supported')
    parser.add_argument('--seg-conf', type=float, default=0.25)
    parser.add_argument('--mask-min-iou', type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.padding <= 1 or not 0 < args.min_overlap <= 1 or not 0 <= args.ambiguity_margin <= 1 or args.limit < 0:
        parser.error('Invalid padding, overlap, ambiguity margin, or limit')
    if not 0 < args.seg_conf <= 1 or not 0 < args.mask_min_iou <= 1:
        parser.error('Segmentation confidence and mask IoU must be in (0, 1]')
    if not 0 <= args.min_ppe_visibility <= 1:
        parser.error('PPE visibility must be between 0 and 1')
    convert(args)


if __name__ == '__main__':
    main()
