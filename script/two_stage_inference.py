"""Two-stage inference: Stage-1 person localization -> Stage-2 per-person detection.

Pipeline:
    full frame
      -> Stage-1 model (person DETECTION or SEGMENTATION, swappable)
      -> crop each person (optionally mask background using seg mask)
      -> Stage-2 model (e.g. PPE DETECTION, swappable) on every crop
      -> map crop-relative boxes back to full-frame coordinates

Modularity:
    * ``BaseStage1`` — anything that localizes people (detect or segment).
      Implement ``extract(frame) -> list[PersonProposal]``.
    * ``BaseStage2`` — anything that detects objects inside a crop.
      Implement ``detect(crop) -> list[CropDetection]``.
    * ``TwoStageInference`` — orchestrator, agnostic to model backends.

    Provided backends (both are Ultralytics YOLO, task auto-detected):
    * ``YoloStage1`` — works with a detection ``*.pt`` OR a segmentation
      ``*-seg.pt``. Set ``task_hint="detect"/"segment"/"auto"``.
    * ``YoloStage2`` — works with any YOLO detection/segmentation ``*.pt``
      (boxes are used; masks ignored). Supports true batched inference via
      ``detect_batch`` (pass a list of crops in one ``predict`` call).

    Batching:
    * Stage-2 crops are processed in batches, not one-by-one. Control with
      ``TwoStageInference(stage1, stage2, stage2_batch=...)``:
      an int (fixed batch size), ``"auto"`` (estimate from crop sizes +
      free CUDA memory), or override per call
      ``process_frame(frame, batch_size=...)``.
    * ``iter_crop_batches`` is a generator (``yield``) so crops are streamed
      batch-by-batch without materializing all results at once.
      ``process_frames`` streams whole frames the same way (good for video).

Example:
    >>> from two_stage_inference import YoloStage1, YoloStage2, TwoStageInference
    >>> # Stage-1 as segmentation, Stage-2 as PPE detection
    >>> stage1 = YoloStage1("yolo11n-seg.pt", classes=[0], conf=0.5)
    >>> stage2 = YoloStage2("best (1).pt", conf=0.5)
    >>> pipe = TwoStageInference(stage1, stage2, padding=10, stage2_batch="auto")
    >>> result = pipe.process_frame(cv2.imread("img.jpg"))          # auto batch
    >>> result = pipe.process_frame(cv2.imread("img.jpg"), batch_size=8)  # fixed
    >>> annotated = pipe.annotate(result.frame, result)

    Swap Stage-1 to detection with one line:
    >>> stage1 = YoloStage1("yolo26n.pt", task_hint="detect")
"""

from __future__ import annotations

import argparse
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def _close_preview() -> None:
    """Cleanup must not mask an inference error on headless OpenCV builds."""
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def _show_preview(frame: np.ndarray, delay: int) -> int | None:
    """Return a key, or None when this OpenCV build lacks GUI support."""
    try:
        cv2.imshow("two-stage", frame)
        return cv2.waitKey(delay)
    except cv2.error as exc:
        if exc.code != -2 or "The function is not implemented" not in str(exc):
            raise
        warnings.warn(
            "OpenCV GUI support is unavailable; continuing without preview. "
            "Use --no-show to disable preview and --output to save annotations.",
            RuntimeWarning, stacklevel=2,
        )
        return None


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PersonProposal:
    """One person found by Stage-1 (coordinates in full-frame space)."""

    x1: int
    y1: int
    x2: int
    y2: int
    conf: float = 1.0
    # Full-frame binary mask (H x W bool) or None (detection models).
    mask: Optional[np.ndarray] = None

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


@dataclass
class CropDetection:
    """One object found by Stage-2 (coordinates mapped to full-frame space)."""

    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    label: str
    conf: float
    person_id: int = -1  # index of the parent PersonProposal


@dataclass
class TwoStageResult:
    """Output of :meth:`TwoStageInference.process_frame`."""

    frame: np.ndarray                     # original frame (BGR, untouched)
    persons: List[PersonProposal] = field(default_factory=list)
    detections: List[CropDetection] = field(default_factory=list)
    stage1_time_ms: float = 0.0
    stage2_time_ms: float = 0.0
    stage2_batches: int = 0               # how many stage-2 batches ran
    stage2_batch_size: int = 0            # resolved batch size for this frame


@dataclass
class _CropJob:
    """Internal unit of Stage-2 work: one crop + where it came from."""

    crop: np.ndarray
    x1: int           # crop origin in full-frame coords
    y1: int
    person_id: int


# ---------------------------------------------------------------------------
# Stage-1 interface (person localization: detection OR segmentation)
# ---------------------------------------------------------------------------

class BaseStage1(ABC):
    """Interface for Stage-1 person localizers."""

    @abstractmethod
    def extract(self, frame: np.ndarray) -> List[PersonProposal]:
        """Return person proposals in full-frame pixel coordinates."""
        raise NotImplementedError

    @property
    def task(self) -> str:
        return "unknown"


class YoloStage1(BaseStage1):
    """Ultralytics YOLO Stage-1: accepts a DETECTION or SEGMENTATION model.

    Args:
        model_path: path to ``*.pt`` (detect) or ``*-seg.pt`` (segment).
        classes: class ids treated as person (default ``[0]`` = COCO person).
        conf: confidence threshold.
        iou: NMS IoU threshold.
        imgsz: inference image size.
        task_hint: ``"auto"`` (read from model), ``"detect"`` or ``"segment"``.
        bbox_from_mask: if True and masks exist, recompute a tight bbox from
            the mask instead of using the detector box.
        min_width / min_height: drop proposals smaller than this (pixels).
        verbose: pass-through to ultralytics.
    """

    def __init__(
        self,
        model_path: str | Path,
        classes: Sequence[int] | None = (0,),
        conf: float = 0.5,
        iou: float = 0.5,
        imgsz: int = 640,
        task_hint: str = "auto",
        bbox_from_mask: bool = False,
        min_width: int = 8,
        min_height: int = 8,
        verbose: bool = False,
    ) -> None:
        self.model = YOLO(str(model_path))
        self.classes = list(classes) if classes is not None else None
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.bbox_from_mask = bbox_from_mask
        self.min_width = min_width
        self.min_height = min_height
        self.verbose = verbose

        detected_task = str(getattr(self.model, "task", "detect") or "detect").lower()
        self._task = task_hint.lower() if task_hint.lower() in ("detect", "segment") else detected_task

    @property
    def task(self) -> str:
        return self._task

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    def extract(self, frame: np.ndarray) -> List[PersonProposal]:
        h, w = frame.shape[:2]
        res = self.model.predict(
            source=frame, classes=self.classes, conf=self.conf,
            iou=self.iou, imgsz=self.imgsz, verbose=self.verbose,
        )[0]

        if res.boxes is None or len(res.boxes) == 0:
            return []

        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else np.ones(len(boxes))

        # Masks are (N, H, W) in full-frame coordinates (if seg model).
        masks: Optional[np.ndarray] = None
        if getattr(res, "masks", None) is not None:
            try:
                masks = res.masks.data.cpu().numpy() > 0.5  # (N,H,W) bool
                # Resize to frame size if ultralytics returned a different res.
                if masks.shape[1] != h or masks.shape[2] != w:
                    import cv2 as _cv2
                    resized = np.zeros((masks.shape[0], h, w), dtype=bool)
                    for i in range(masks.shape[0]):
                        resized[i] = _cv2.resize(
                            masks[i].astype(np.uint8), (w, h),
                            interpolation=_cv2.INTER_NEAREST,
                        ).astype(bool)
                    masks = resized
            except Exception:
                masks = None

        proposals: List[PersonProposal] = []
        for i, (x1, y1, x2, y2) in enumerate(boxes.tolist()):
            mask_i: Optional[np.ndarray] = None
            if masks is not None and i < len(masks):
                mask_i = masks[i]
                if self.bbox_from_mask:
                    tight = self._mask_to_bbox(mask_i)
                    if tight is None:
                        continue
                    x1, y1, x2, y2 = tight

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < self.min_width or y2 - y1 < self.min_height:
                continue
            proposals.append(PersonProposal(x1, y1, x2, y2, float(confs[i]), mask_i))
        return proposals


# ---------------------------------------------------------------------------
# Stage-2 interface (per-crop detection)
# ---------------------------------------------------------------------------

class BaseStage2(ABC):
    """Interface for Stage-2 per-crop detectors.

    Implement :meth:`detect` (single crop). Override :meth:`detect_batch`
    if your backend supports true batched inference (faster); the default
    loops over :meth:`detect` so old subclasses keep working.
    """

    @abstractmethod
    def detect(self, crop: np.ndarray) -> List[CropDetection]:
        """Return detections in CROP-relative pixel coordinates (x1,y1,x2,y2)."""
        raise NotImplementedError

    def detect_batch(self, crops: Sequence[np.ndarray]) -> List[List[CropDetection]]:
        """Batched version: ``crops[i]`` -> detections for that crop.

        Default implementation loops over :meth:`detect`. Backends with
        native batch support (e.g. :class:`YoloStage2`) override this.
        """
        return [self.detect(c) for c in crops]

    @property
    def names(self) -> Dict[int, str]:
        return dict(getattr(self, "_names", {}))


class YoloStage2(BaseStage2):
    """Ultralytics YOLO Stage-2 (any detection/pose/seg ``*.pt``; boxes used).

    The returned boxes are crop-relative; :class:`TwoStageInference` maps
    them back to full-frame coordinates automatically.

    Args:
        model_path: path to detection weights (e.g. PPE model).
        conf / iou / imgsz: inference hyper-parameters.
        classes: optional class filter (None = all classes).
        verbose: pass-through to ultralytics.
        batch: default ``batch`` passthrough for ultralytics dataloader
            (rarely needed since :class:`TwoStageInference` chunks the
            crops itself; kept for manual ``detect_batch`` calls).
    """

    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.5,
        iou: float = 0.5,
        imgsz: int = 640,
        classes: Sequence[int] | None = None,
        verbose: bool = False,
        batch: int = 1,
    ) -> None:
        self.model = YOLO(str(model_path))
        self._names: Dict[int, str] = dict(getattr(self.model, "names", {}))
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = list(classes) if classes is not None else None
        self.verbose = verbose
        self.batch = batch

    @staticmethod
    def _parse_result(res, names: Dict[int, str]) -> List[CropDetection]:
        if res.boxes is None or len(res.boxes) == 0:
            return []
        out: List[CropDetection] = []
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, cf in zip(boxes.tolist(), cls.tolist(), conf.tolist()):
            out.append(CropDetection(
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                cls_id=int(c), label=str(names.get(int(c), int(c))),
                conf=float(cf),
            ))
        return out

    def detect(self, crop: np.ndarray) -> List[CropDetection]:
        if crop.size == 0:
            return []
        res = self.model.predict(
            source=crop, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            classes=self.classes, verbose=self.verbose,
        )[0]
        return self._parse_result(res, self.names)

    def detect_batch(self, crops: Sequence[np.ndarray]) -> List[List[CropDetection]]:
        """Run one batched ``predict`` over all crops (much faster than loop)."""
        crops = [c for c in crops if c.size > 0]
        if not crops:
            return []
        results = self.model.predict(
            source=list(crops), conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            classes=self.classes, verbose=self.verbose, batch=len(crops),
        )
        return [self._parse_result(r, self.names) for r in results]


# ---------------------------------------------------------------------------
# Auto batch sizing helpers
# ---------------------------------------------------------------------------

def _free_cuda_bytes() -> Optional[int]:
    """Free CUDA bytes on device 0, or None if CUDA is unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info(0)
        return int(free)
    except Exception:
        return None


def estimate_auto_batch(
    crops: Sequence[np.ndarray],
    imgsz: int = 640,
    max_batch: int = 32,
    pixel_budget: int = 8 * 640 * 640,
    cuda_safety: float = 0.4,
    bytes_per_item_at_640: int = 200 * 1024 * 1024,
) -> int:
    """Heuristic batch size from crop sizes + free GPU memory.

    * Pixel budget: ``batch ~= pixel_budget / mean_crop_pixels`` (RAM friendly).
    * CUDA refinement: ``batch ~= free_cuda * safety / per_item_bytes`` where
      per-item cost scales with ``(imgsz/640)^2``.
    Returns at least 1, at most ``min(max_batch, len(crops))``.
    """
    n = len(crops)
    if n == 0:
        return 0
    if n == 1:
        return 1
    mean_px = float(np.mean([max(1, c.shape[0] * c.shape[1]) for c in crops]))
    pixel_batch = max(1, int(pixel_budget // max(1, mean_px)))
    batch = min(n, max_batch, pixel_batch)
    free = _free_cuda_bytes()
    if free is not None:
        per_item = bytes_per_item_at_640 * (imgsz / 640.0) ** 2
        cuda_batch = max(1, int(free * cuda_safety / max(1, per_item)))
        batch = min(batch, max(1, cuda_batch))
    return max(1, batch)


def _is_oom(err: BaseException) -> bool:
    msg = f"{type(err).__name__}: {err}".lower()
    return "out of memory" in msg or "cudnn" in msg and "memory" in msg


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TwoStageInference:
    """Run Stage-1 (person) then Stage-2 (per-crop, batched) and fuse results.

    Args:
        stage1: any :class:`BaseStage1` (detection or segmentation backend).
        stage2: any :class:`BaseStage2` (per-crop detector, ideally with
            native ``detect_batch`` like :class:`YoloStage2`).
        padding: pixels to expand each person box before cropping.
        apply_mask: if True and Stage-1 provides masks, zero-out background
            outside the person mask inside the crop (helps PPE models focus
            on the person). If False, plain bbox crop is used.
        min_crop_size: skip crops smaller than this (w or h in pixels).
        keep_person_boxes: keep Stage-1 boxes in the result even when Stage-2
            finds nothing (useful for visualization/debug).
        stage2_batch: fixed int batch size for Stage-2, or ``"auto"`` to
            estimate per frame from crop sizes + free CUDA memory.
        max_batch: upper clamp for ``"auto"`` mode.
        pixel_budget: total native crop pixels per auto batch (RAM friendly).
    """

    def __init__(
        self,
        stage1: BaseStage1,
        stage2: BaseStage2,
        padding: int = 10,
        apply_mask: bool = False,
        min_crop_size: int = 16,
        keep_person_boxes: bool = True,
        stage2_batch: int | str = "auto",
        max_batch: int = 32,
        pixel_budget: int = 8 * 640 * 640,
    ) -> None:
        self.stage1 = stage1
        self.stage2 = stage2
        self.padding = padding
        self.apply_mask = apply_mask
        self.min_crop_size = min_crop_size
        self.keep_person_boxes = keep_person_boxes
        self.stage2_batch = stage2_batch
        self.max_batch = max_batch
        self.pixel_budget = pixel_budget

    # -- core --------------------------------------------------------------
    def _expand_clip(self, p: PersonProposal, w: int, h: int) -> Tuple[int, int, int, int]:
        return (
            max(0, p.x1 - self.padding),
            max(0, p.y1 - self.padding),
            min(w, p.x2 + self.padding),
            min(h, p.y2 + self.padding),
        )

    def _make_crop(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                   mask: Optional[np.ndarray]) -> np.ndarray:
        crop = frame[y1:y2, x1:x2].copy()
        if self.apply_mask and mask is not None:
            m = mask[y1:y2, x1:x2]
            if m.shape[:2] == crop.shape[:2]:
                crop[~m] = 0  # black-out background
        return crop

    # -- batch plumbing -----------------------------------------------------
    def _prepare_jobs(self, frame: np.ndarray,
                      persons: List[PersonProposal]) -> List[_CropJob]:
        """Build one crop job per valid person box."""
        h, w = frame.shape[:2]
        jobs: List[_CropJob] = []
        for pid, p in enumerate(persons):
            x1, y1, x2, y2 = self._expand_clip(p, w, h)
            if x2 - x1 < self.min_crop_size or y2 - y1 < self.min_crop_size:
                continue
            crop = self._make_crop(frame, x1, y1, x2, y2, p.mask)
            if crop.size == 0:
                continue
            jobs.append(_CropJob(crop=crop, x1=x1, y1=y1, person_id=pid))
        return jobs

    def _resolve_batch_size(self, jobs: List[_CropJob],
                            override: int | str | None = None) -> int:
        """Resolve fixed int vs ``"auto"`` batch size for this frame."""
        if not jobs:
            return 0
        setting = override if override is not None else self.stage2_batch
        if isinstance(setting, str):
            if setting.lower() != "auto":
                raise ValueError(f'stage2_batch must be int or "auto", got {setting!r}')
            imgsz = int(getattr(self.stage2, "imgsz", 640) or 640)
            return estimate_auto_batch(
                [j.crop for j in jobs], imgsz=imgsz,
                max_batch=self.max_batch, pixel_budget=self.pixel_budget,
            )
        return max(1, min(int(setting), len(jobs)))

    def iter_crop_batches(
        self,
        jobs: List[_CropJob],
        batch_size: int | str | None = None,
    ) -> Iterator[List[_CropJob]]:
        """Yield successive crop batches (generator — memory friendly).

        Args:
            jobs: output of :meth:`_prepare_jobs`.
            batch_size: int to force a size, ``"auto"`` to estimate,
                or None to use the constructor's ``stage2_batch`` setting.
        """
        bs = self._resolve_batch_size(jobs, override=batch_size)
        if bs <= 0:
            return
        for i in range(0, len(jobs), bs):
            yield jobs[i:i + bs]

    def _detect_jobs(self, batch_jobs: List[_CropJob]) -> List[List[CropDetection]]:
        """Run Stage-2 on one batch, halving on OOM until it fits."""
        crops = [j.crop for j in batch_jobs]
        bs = len(crops)
        while True:
            try:
                out = self.stage2.detect_batch(crops)
                # Normalize: backends must return one list per crop.
                if len(out) != len(crops):
                    raise RuntimeError(
                        f"detect_batch returned {len(out)} results for {len(crops)} crops")
                return out
            except Exception as e:  # noqa: BLE001 — need OOM fallback
                if not _is_oom(e) or bs <= 1:
                    raise
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                bs = max(1, bs // 2)
                # Retry as two halves (recursion keeps halving on repeat OOM).
                mid = len(crops) // 2
                first = self._detect_jobs(
                    [_CropJob(crop=c, x1=j.x1, y1=j.y1, person_id=j.person_id)
                     for c, j in zip(crops[:mid], batch_jobs[:mid])])
                second = self._detect_jobs(
                    [_CropJob(crop=c, x1=j.x1, y1=j.y1, person_id=j.person_id)
                     for c, j in zip(crops[mid:], batch_jobs[mid:])])
                return first + second

    def _fuse(self, batch_jobs: List[_CropJob],
              batch_dets: List[List[CropDetection]],
              img_w: int, img_h: int) -> List[CropDetection]:
        """Map crop-relative boxes back to full-frame coordinates."""
        fused: List[CropDetection] = []
        for job, dets in zip(batch_jobs, batch_dets):
            for d in dets:
                gx1 = min(max(0, job.x1 + d.x1), img_w)
                gy1 = min(max(0, job.y1 + d.y1), img_h)
                gx2 = min(max(0, job.x1 + d.x2), img_w)
                gy2 = min(max(0, job.y1 + d.y2), img_h)
                if gx2 <= gx1 or gy2 <= gy1:
                    continue
                d.x1, d.y1, d.x2, d.y2 = gx1, gy1, gx2, gy2
                d.person_id = job.person_id
                fused.append(d)
        return fused

    def process_frame(self, frame: np.ndarray,
                      batch_size: int | str | None = None) -> TwoStageResult:
        """Run the full two-stage pipeline on one BGR frame.

        Args:
            frame: BGR image.
            batch_size: override Stage-2 batching for this call — int for a
                fixed size, ``"auto"`` to estimate, None = constructor default.
        """
        h, w = frame.shape[:2]

        t0 = time.perf_counter()
        persons = self.stage1.extract(frame)
        t1 = time.perf_counter()

        jobs = self._prepare_jobs(frame, persons)
        detections: List[CropDetection] = []
        n_batches = 0
        resolved_bs = self._resolve_batch_size(jobs, override=batch_size)
        for chunk in self.iter_crop_batches(jobs, batch_size=batch_size):
            n_batches += 1
            batch_dets = self._detect_jobs(chunk)
            detections.extend(self._fuse(chunk, batch_dets, w, h))
        t2 = time.perf_counter()

        return TwoStageResult(
            frame=frame, persons=persons, detections=detections,
            stage1_time_ms=(t1 - t0) * 1000.0, stage2_time_ms=(t2 - t1) * 1000.0,
            stage2_batches=n_batches, stage2_batch_size=resolved_bs,
        )

    def process_frames(self, frames: Iterable[np.ndarray],
                       batch_size: int | str | None = None) -> Iterator[TwoStageResult]:
        """Stream results frame-by-frame (generator — use for video).

        Example:
            >>> for result in pipe.process_frames(video_frames):
            ...     vis = pipe.annotate(result.frame, result)
        """
        for frame in frames:
            yield self.process_frame(frame, batch_size=batch_size)

    # -- visualization ------------------------------------------------------
    def annotate(
        self,
        frame: np.ndarray,
        result: TwoStageResult,
        draw_persons: bool = True,
        draw_masks: bool = False,
        person_color: Tuple[int, int, int] = (255, 0, 0),
        mask_color: Tuple[int, int, int] = (255, 0, 0),
    ) -> np.ndarray:
        """Draw fused Stage-2 boxes (+ optionally Stage-1 boxes/masks)."""
        vis = frame.copy()
        if draw_masks:
            overlay = vis.copy()
            for p in result.persons:
                if p.mask is not None:
                    overlay[p.mask] = mask_color
            vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
        if draw_persons:
            for i, p in enumerate(result.persons):
                cv2.rectangle(vis, (p.x1, p.y1), (p.x2, p.y2), person_color, 1)
                cv2.putText(vis, f"person#{i} {p.conf:.2f}", (p.x1, max(0, p.y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, person_color, 1)
        for d in result.detections:
            cv2.rectangle(vis, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{d.label} {d.conf:.2f} [p{d.person_id}]",
                        (d.x1, max(0, d.y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return vis

    # -- I/O helpers ---------------------------------------------------------
    def run_image(self, image_path: str | Path, save_path: str | Path | None = None,
                  show: bool = False, draw_persons: bool = True,
                  draw_masks: bool = False,
                  batch_size: int | str | None = None) -> Tuple[TwoStageResult, np.ndarray]:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        result = self.process_frame(frame, batch_size=batch_size)
        vis = self.annotate(frame, result, draw_persons=draw_persons, draw_masks=draw_masks)
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), vis)
        if show:
            try:
                _show_preview(vis, 0)
            finally:
                _close_preview()
        return result, vis

    def _frame_stream(self, cap: "cv2.VideoCapture") -> Iterator[np.ndarray]:
        """Yield frames from an open capture (generator)."""
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame

    def run_video(self, source: str | Path | int, save_path: str | Path | None = None,
                  show: bool = True, draw_persons: bool = True,
                  draw_masks: bool = False, max_frames: int = -1,
                  batch_size: int | str | None = None) -> None:
        """Run on a video file / webcam index / stream URL."""
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        writer = None
        preview_requested = show
        try:
            n = 0
            for frame in self._frame_stream(cap):
                result = self.process_frame(frame, batch_size=batch_size)
                vis = self.annotate(frame, result, draw_persons=draw_persons,
                                    draw_masks=draw_masks)
                if save_path is not None and writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    vw, vh = vis.shape[1], vis.shape[0]
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(save_path), fourcc, fps, (vw, vh))
                if writer is not None:
                    writer.write(vis)
                if show:
                    key = _show_preview(vis, 1)
                    if key is None:
                        show = False
                    elif key & 0xFF == ord("q"):
                        break
                n += 1
                if 0 < max_frames <= n:
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if preview_requested:
                _close_preview()


# ---------------------------------------------------------------------------
# Factory (one-line backend swapping)
# ---------------------------------------------------------------------------

def build_stage1(model_path: str | Path, task_hint: str = "auto", **kwargs) -> YoloStage1:
    """Build Stage-1 backend. Same call works for detect and segment models.

    Example:
        build_stage1("yolo26n.pt")        # person detection
        build_stage1("yolo11n-seg.pt")    # person segmentation
    """
    return YoloStage1(model_path, task_hint=task_hint, **kwargs)


def build_stage2(model_path: str | Path, **kwargs) -> YoloStage2:
    """Build Stage-2 per-crop detector (e.g. PPE model)."""
    return YoloStage2(model_path, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Two-stage inference: person (detect/seg) -> per-crop detection.")
    ap.add_argument("--stage1", help="Stage-1 weights (person detect *.pt or segment *-seg.pt).", default="yolo26m-seg.pt")
    ap.add_argument("--stage2", help="Stage-2 weights (e.g. PPE detection *.pt).", default=r"yoloformat_dataset/PPE_person_crops_v40_filtered/output/train/weights/best.pt")
    ap.add_argument("--source", required=True, help="Image / video file / folder / webcam index (e.g. 0) / stream URL.")
    ap.add_argument("--output", default=None, help="Save path (image or video). For folders, treated as output dir.")
    ap.add_argument("--task-hint", default="auto", choices=["auto", "detect", "segment"])
    ap.add_argument("--stage1-conf", type=float, default=0.5)
    ap.add_argument("--stage2-conf", type=float, default=0.5)
    ap.add_argument("--imgsz1", type=int, default=640)
    ap.add_argument("--imgsz2", type=int, default=160)
    ap.add_argument("--padding", type=int, default=10, help="Pixels to expand person box before cropping.")
    ap.add_argument("--apply-mask", action="store_true", help="Black-out background outside seg mask in crops.")
    ap.add_argument("--bbox-from-mask", action="store_true", help="Recompute tight bbox from seg mask.")
    ap.add_argument("--stage2-batch", default="auto",
                    help='Stage-2 batch size: int (e.g. 8) or "auto" (default).')
    ap.add_argument("--max-batch", type=int, default=32, help="Upper clamp for --stage2-batch auto.")
    ap.add_argument("--no-person-box", action="store_true", help="Hide Stage-1 person boxes in visualization.")
    ap.add_argument("--draw-masks", action="store_true", help="Overlay Stage-1 segmentation masks.")
    ap.add_argument("--no-show", action="store_true")
    return ap.parse_args()


def _parse_batch(value: str) -> int | str:
    v = str(value).strip().lower()
    if v == "auto":
        return "auto"
    iv = int(v)
    if iv < 1:
        raise ValueError("--stage2-batch must be >= 1 or 'auto'")
    return iv


def main() -> None:
    args = _parse_args()

    stage1 = build_stage1(
        args.stage1, task_hint=args.task_hint,
        classes=[0], conf=args.stage1_conf, imgsz=args.imgsz1,
        bbox_from_mask=args.bbox_from_mask,
    )
    stage2 = build_stage2(args.stage2, conf=args.stage2_conf, imgsz=args.imgsz2)
    batch_setting = _parse_batch(args.stage2_batch)
    pipe = TwoStageInference(stage1, stage2, padding=args.padding, apply_mask=args.apply_mask,
                             stage2_batch=batch_setting, max_batch=args.max_batch)

    print(f"[stage1] task={stage1.task} weights={args.stage1}")
    print(f"[stage2] classes={stage2.names} weights={args.stage2} batch={batch_setting}")

    src = args.source
    draw_persons = not args.no_person_box
    show = not args.no_show

    # Webcam index / stream
    if str(src).isdigit() or str(src).startswith(("rtsp://", "rtmp://", "http://", "https://")):
        pipe.run_video(src, save_path=args.output, show=show,
                       draw_persons=draw_persons, draw_masks=args.draw_masks)
        return

    p = Path(src)
    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        result, _ = pipe.run_image(p, save_path=args.output, show=show,
                                   draw_persons=draw_persons, draw_masks=args.draw_masks)
        print(f"persons={len(result.persons)} detections={len(result.detections)} "
              f"({result.stage1_time_ms:.1f}ms + {result.stage2_time_ms:.1f}ms "
              f"in {result.stage2_batches}x{result.stage2_batch_size})")
    elif p.is_file():
        pipe.run_video(p, save_path=args.output, show=show,
                       draw_persons=draw_persons, draw_masks=args.draw_masks)
    elif p.is_dir():
        outdir = Path(args.output) if args.output else p / "two_stage_out"
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        imgs = [f for f in sorted(p.iterdir()) if f.suffix.lower() in exts]
        for img in imgs:
            result, _ = pipe.run_image(img, save_path=outdir / img.name, show=False,
                                       draw_persons=draw_persons, draw_masks=args.draw_masks)
            print(f"{img.name}: persons={len(result.persons)} detections={len(result.detections)}")
    else:
        raise FileNotFoundError(f"Source not found: {src}")


if __name__ == "__main__":
    main()
