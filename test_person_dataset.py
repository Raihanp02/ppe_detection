import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2
from script.filter_person_dataset import measure
from script.deduplicate_crop_images import select_duplicates

from script.convert_person_dataset import assign_items, attributes, iou, match_masks, read_labels, segmented_image, resolve_device, deduplicate_labels, crop_labels


class ConversionTests(unittest.TestCase):
    def test_visibility_threshold_is_inclusive_and_uses_original_ppe_area(self):
        item = dict(class_id=8, box=(.1, .1, .6, .6), line=1)
        args = ((100, 100, 3), {8: 7})
        rows, kept = crop_labels([item], [40, 0, 80, 100], *args, min_visibility=.4)
        self.assertEqual(kept, [item])  # Exactly 20*50 / (50*50) = 40%.
        self.assertEqual(rows, ['7 0.25000000 0.35000000 0.50000000 0.50000000'])
        self.assertEqual(crop_labels([item], [41, 0, 80, 100], *args, min_visibility=.4), ([], []))
        # Half the width AND half the height means 25% area, not 50%.
        self.assertEqual(crop_labels([item], [35, 35, 60, 60], *args, min_visibility=.4), ([], []))
        self.assertEqual(crop_labels([item], [0, 0, 100, 100], *args, min_visibility=1)[1], [item])
        self.assertEqual(crop_labels([item], [41, 0, 80, 100], *args, min_visibility=0)[1], [item])

    def test_duplicate_crops_prefer_heldout_split_and_require_identical_labels(self):
        labels = frozenset({(7, .5, .5, .2, .2)})
        train = ('same', {'image': 'train/images/a.png'}, Path('train/labels/a.txt'), labels)
        val = ('same', {'image': 'val/images/b.png'}, Path('val/labels/b.txt'), labels)
        test = ('same', {'image': 'test/images/c.png'}, Path('test/labels/c.txt'), labels)
        duplicates, conflicts = select_duplicates([train, val, test])
        self.assertFalse(conflicts)
        self.assertEqual(len(duplicates), 2)
        self.assertTrue(all(keep == test for keep, remove in duplicates))
        changed = ('same', test[1], test[2], frozenset())
        duplicates, conflicts = select_duplicates([train, changed])
        self.assertFalse(duplicates)
        self.assertEqual(len(conflicts), 1)

    def test_visible_boxes_are_included_in_both_overlapping_person_crops(self):
        item = dict(class_id=8, box=(.45, .2, .55, .7), line=10)
        for crop in ([0, 0, 60, 100], [40, 0, 100, 100]):
            rows, kept = crop_labels([item], crop, (100, 100, 3), {8: 7})
            self.assertEqual(kept, [item])
            self.assertEqual(len(rows), 1)

    def test_visible_boxes_are_clipped_but_touching_or_outside_boxes_are_excluded(self):
        items = [dict(class_id=8, box=(.1, .1, .3, .5), line=1),
                 dict(class_id=8, box=(0, 0, .2, .3), line=2),
                 dict(class_id=8, box=(.8, .8, .9, .9), line=3)]
        rows, kept = crop_labels(items, [20, 20, 60, 80], (100, 100, 3), {8: 7})
        self.assertEqual(kept, [items[0]])
        self.assertEqual(rows, ['7 0.12500000 0.25000000 0.25000000 0.50000000'])

    def test_filter_measures_background_crop_without_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root/'train/images').mkdir(parents=True)
            (root/'train/labels').mkdir()
            cv2.imwrite(str(root/'train/images/person.png'), np.full((100, 40, 3), 123, dtype=np.uint8))
            (root/'train/labels/person.txt').write_text('')
            record = dict(image='train/images/person.png')
            measured = measure(root, record, 'crop')
            self.assertIsNone(measured[2])
            self.assertEqual(measured[-1], dict(width=40, height=100, crop_pixels=4000))
            with self.assertRaisesRegex(ValueError, 'Mask missing'):
                measure(root, record, 'mask')

    def test_deduplication_preserves_different_boxes_classes_and_provenance(self):
        rows = ['7 0.5 0.5 0.2 0.2', '7 0.50 0.5 0.20 0.2',
                '6 0.5 0.5 0.2 0.2', '7 0.51 0.5 0.2 0.2']
        self.assertEqual(deduplicate_labels(rows, [9, 17, 18, 19]),
                         ([rows[0], rows[2], rows[3]], [9, 18, 19], [17]))

    def test_auto_device_uses_cuda_when_available_and_cpu_otherwise(self):
        with patch('torch.cuda.is_available', return_value=True):
            self.assertEqual(resolve_device('auto'), '0')
            self.assertEqual(resolve_device('cpu'), 'cpu')
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(resolve_device('auto'), 'cpu')

    def test_segmented_image_has_no_hidden_background(self):
        crop = np.full((2, 2, 3), 123, dtype=np.uint8)
        mask = np.array([[255, 0], [0, 255]], dtype=np.uint8)
        result = segmented_image(crop, mask)
        self.assertTrue((result[mask == 0] == 0).all())
        self.assertTrue((result[mask != 0, :3] == 123).all())
        self.assertTrue((result[:, :, 3] == mask).all())

    def test_assignment_and_ambiguous_overlap(self):
        people = [dict(box=(0, 0, .6, 1)), dict(box=(.4, 0, 1, 1))]
        unique = dict(name='hardhat', box=(.1, .1, .2, .2))
        ambiguous = dict(name='vest', box=(.45, .2, .55, .7))
        assigned, unresolved = assign_items([unique, ambiguous], people)
        self.assertEqual(assigned, [[unique], []])
        self.assertEqual(unresolved[0]['reason'], 'ambiguous_person')
        self.assertEqual(set(unresolved[0]['candidates']), {0, 1})

    def test_unlabelled_is_unknown_and_conflict_is_preserved(self):
        result = attributes([dict(name='hardhat'), dict(name='no_hardhat'), dict(name='no_vest')])
        self.assertEqual(result, dict(boots='unknown', gloves='unknown', hardhat='conflict', vest='absent'))

    def test_polygon_converted_and_bad_box_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'labels.txt'
            path.write_text('0 0.1 0.2 0.6 0.2 0.6 0.8\n')
            self.assertEqual(read_labels(path, ['vest'])[0]['box'], (.1, .2, .6, .8))
            path.write_text('0 0.5 0.5 -0.2 0.2\n')
            with self.assertRaises(ValueError):
                read_labels(path, ['vest'])

    def test_mask_matching_geometry(self):
        self.assertEqual(iou((0, 0, 1, 1), (0, 0, 1, 1)), 1)
        self.assertEqual(iou((0, 0, .2, .2), (.4, .4, 1, 1)), 0)

    def test_one_mask_cannot_be_assigned_to_two_people(self):
        coordinates = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array([[0, 0, 100, 100]])))
        result = SimpleNamespace(masks=True, boxes=SimpleNamespace(xyxy=coordinates))
        matches, uncertain = match_masks(result, [dict(box=(0, 0, 1, 1)), dict(box=(.1, .1, .9, .9))], 100, 100, .5)
        self.assertEqual(matches, {0: (0, 1.0)})
        self.assertEqual(uncertain, {0, 1})

    def test_missing_masks_remain_unmatched(self):
        self.assertEqual(match_masks(SimpleNamespace(masks=None), [dict(box=(0, 0, 1, 1))], 100, 100, .5), ({}, set()))


if __name__ == '__main__':
    unittest.main()
