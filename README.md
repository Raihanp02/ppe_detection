# PPE detection datasets and scripts

Convert the full-frame dataset to person masks with PPE attributes:

```powershell
.venv/Scripts/python.exe script/convert_person_dataset.py --data yoloformat_dataset/PPE_detection_5400/data.yaml --output yoloformat_dataset/PPE_segmented_people --seg-model yolo26n-seg.pt --device 0
```

Use a new output directory for each run. Add `--limit 5` for a small preview
(five frames per split). Use `--device cpu` to force CPU execution. Omit
`--seg-model` to export only crops and attributes.

The default device is `auto`: CUDA GPU 0 when available, otherwise CPU. You can
also pass `--device auto` explicitly. The requested and selected devices are
recorded in the conversion summary. For medium-model conversion and filtering:

```powershell
.venv/Scripts/python.exe script/convert_person_dataset.py --data yoloformat_dataset/PPE_detection_5400/data.yaml --output yoloformat_dataset/_PPE_yolo26m_conversion --seg-model yolo26m-seg.pt --device auto
.venv/Scripts/python.exe script/filter_person_dataset.py --source yoloformat_dataset/_PPE_yolo26m_conversion --output yoloformat_dataset/PPE_segmented_people_yolo26m_filtered
```

After successful filtering and verification, the intermediate conversion folder
can be deleted. The filtered folder retains conversion settings, review records,
and unmatched-person records in addition to the filtered images, labels, masks,
and manifests. This run uses the same 32px width, 96px height, and 2,048 foreground
pixel thresholds as the earlier filtered dataset.

The converter uses existing person boxes as the source of person identities.
It runs the local segmentation model on the full image and matches predicted
person boxes to annotated people with one-to-one greedy IoU matching
(minimum 0.5). Masks are predictions, **not human-verified ground truth**.
Unmatched people remain in the manifest with `mask: null`, `image: null`, and
`export_status: skipped_no_mask`; no image or detection label file is exported
for them. Competing matches are marked `mask_ambiguous`.
People absent from the original person annotations are not added automatically.

Output:

- `train/`, `val/`, `test/`: original split membership is preserved.
- `<split>/images/`: segmented person PNGs with transparent background. RGB
  pixels outside the mask are also zeroed, so RGB-only training loaders see
  a black background. The canvas is the person box with 5% padding per side.
- `<split>/masks/`: matching crop-sized PNG masks, foreground 255 and background 0.
- `<split>/labels/`: crop-relative YOLO PPE boxes, with the person class removed.
- `persons.jsonl`: one record per annotated person, including image/mask paths,
  source image and annotation references, crop coordinates, mask provenance,
  match IoU, review flags, and `boots`, `gloves`, `hardhat`, `vest` attributes.
- `review.jsonl`: unassigned/ambiguous PPE labels, frames without annotated
  people, and people without matched masks. All predicted masks additionally
  have `mask_status: needs_review` in `persons.jsonl`.
- `summary.json`: counts, conversion settings, and original Roboflow metadata.
- `data.yaml`: YOLO **PPE detection** configuration for the crop images. This is
  not a YOLO segmentation configuration; segmentation plus attributes is
  represented by the PNG masks and JSONL manifest.

For segmentation exports (or explicit `--label-policy assigned`), PPE labels
are assigned using the fraction of their bounding-box area inside
each person's original box. At least 70% must overlap the best person; the best
score must exceed the runner-up by at least 0.15. Ambiguous labels are retained
for review instead of being assigned arbitrarily. These geometric assignments
still need review in crowded scenes. Polygon PPE labels are converted to their
enclosing boxes. Boxes are clipped to crop boundaries.
Exact duplicate class/box rows are removed after cropping and mask filtering;
the first row and its source reference are retained. Removed source line
references are recorded in `duplicate_label_lines_removed` in the manifest.

Attribute values are `present`, `absent` (only from explicit `no_*` labels),
`unknown` (no assigned evidence), or `conflict` (both positive and negative
labels). A positive glove/boot annotation indicates labelled evidence; it does
not establish that both hands/feet are compliant.

The segmentation mask is applied before exporting the image. PPE boxes remain
crop-relative; labels whose boxes have no remaining foreground pixels are
omitted and recorded as `ppe_erased_by_mask` in the review file. The manifest
retains `source_attributes` as well as attributes from the surviving labels.
Inspect PPE near mask boundaries because predicted masks can partially erase
equipment or include another person. Retained foreground within a PPE box does
not prove that the equipment itself was preserved. Review labels before
training; ambiguous/unassigned PPE can otherwise become unlabelled objects.

`PPE_person_masks_preview` and the interrupted `PPE_person_masks` output, if
present, are earlier exports with background retained. Use
`PPE_segmented_people` for background-removed images. `PPE_segmented_preview`
contains a small sample of the current format.

To reconstruct a mask in the original frame, place its pixels within
`crop_xyxy = [left, top, right, bottom]`. Coordinates use exclusive right/bottom
bounds. Paths within the manifest are relative to the output directory except
for source and model paths.

Validation:

```powershell
.venv/Scripts/python.exe -m unittest test_person_dataset -v
```

Filter tiny segmented people into a separate dataset:

```powershell
.venv/Scripts/python.exe script/filter_person_dataset.py --source yoloformat_dataset/PPE_segmented_people --output yoloformat_dataset/PPE_segmented_people_filtered --min-width 32 --min-height 96 --min-foreground 2048
```

All three limits apply to the actual person mask at its original resolution,
excluding transparent padding. People below any limit are excluded together
with their label and mask files. These are conservative starting heuristics,
not thresholds proven to improve training accuracy. Original split membership,
retained pixels, and retained PPE labels are preserved. The script verifies
all copied files against the source and writes counts by split and PPE class
to `filter_summary.json`. `removed.jsonl` records rejected people, measurements,
and reasons; its paths refer to the source dataset. Existing directories are
never overwritten.

A stricter experiment can use `--min-width 48 --min-height 128
--min-foreground 4096` with a different output directory. Filtering validation
and test changes the evaluation population: also evaluate against the original
test set if tiny people appear in deployment. Small people can provide useful
training examples, so compare model results before adopting a stricter cutoff.

For person crops that retain background inside the annotated person box:

```powershell
.venv/Scripts/python.exe script/convert_person_dataset.py --data yoloformat_dataset/PPE_detection_5400/data.yaml --output yoloformat_dataset/_PPE_person_crops_v40_conversion --padding 0 --label-policy visible --min-ppe-visibility 0.4
.venv/Scripts/python.exe script/filter_person_dataset.py --source yoloformat_dataset/_PPE_person_crops_v40_conversion --output yoloformat_dataset/PPE_person_crops_v40_filtered --size-mode crop --min-width 32 --min-height 96 --min-area 2048
.venv/Scripts/python.exe script/deduplicate_crop_images.py --dataset yoloformat_dataset/PPE_person_crops_v40_filtered
```

No segmentation model is used. `--padding 0` crops the annotated person box
without extra surrounding padding. Crop filtering uses rectangular image width,
height and pixel area; it does not measure person foreground. By default, a PPE
box must have at least 40% of its original area inside a crop to be included:
`intersection_area / original_PPE_box_area >= 0.4`. This is not IoU or the
percentage of the person box covered. Included boxes are clipped to the crop
boundary, with exact duplicate rows removed. Boxes on neighboring people can
qualify too; no unique-owner or ambiguity threshold excludes them.
`--label-policy auto` (the default) selects this visible-box policy for crops
without a segmentation model; segmentation exports retain exclusive assignment.
The manifest uses `attribute_scope: crop` for visible-box exports: its attributes
summarize all visible PPE labels, not verified compliance of one person. After
verification, the intermediate conversion directory can be removed. Background
and neighboring people remain visible. Change `--min-ppe-visibility` to any value
from 0 to 1; `0` restores all positive intersections and `1` requires full box
visibility. The cutoff is inclusive. Rejected annotation references are saved
in `visibility_rejected_label_lines`. Segmentation/exclusive-assignment mode is
unaffected by this setting. This exports existing annotations and cannot restore
missing source labels.

The deduplication command removes exact pixel-identical crops only when their
numeric label sets also match. It keeps test copies over validation copies,
then training copies, and writes `image_deduplication.json` with removed source
records and retained-image mappings. Conflicting annotations stop deletion and
are reported for review. Similar-looking, shifted, resized or recompressed
images are not automatically removed. The earlier `PPE_person_crops_filtered`
folder used exclusive assignment; `PPE_person_crops_visible_filtered` used all
positive intersections. Use `PPE_person_crops_v40_filtered` for the 40% threshold.
