import json
import os
import time
from functools import lru_cache

import cv2
import numpy as np
import onnxruntime as ort


DEFAULT_CONFIG = {
    "seg_input_size": (1024, 1024),
    "det_input_size": 960,
    "num_seg_classes": 8,
    "road_class_id": 2,
    "person_seg_class_id": 6,
    "det_conf_threshold": 0.35,
    "det_iou_threshold": 0.45,
    "overlap_ratio_threshold": 0.15,
    "road_dilate_px": 25,
    "seg_class_names": [
        "Clutter", "Building", "Road", "Static_Car",
        "Tree", "Vegetation", "Human", "Moving_Car"
    ],
    # TTA (Test-Time Augmentation) - matching PaddleSeg aug_pred
    "use_tta": True,
    "tta_scales": [1.0],
    "tta_flip_h": True,
    "tta_flip_v": True,
}


def get_color_map(num_classes=256):
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]
    return np.array(color_map).reshape(-1, 3)


COLOR_MAP = get_color_map()
COLORS = {
    "safe": (0, 220, 100),
    "danger": (0, 0, 255),
    "warning": (0, 165, 255),
    "info": (255, 200, 50),
    "road": (255, 100, 50),
}


def create_onnx_session(model_path, use_gpu=True):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    available = ort.get_available_providers()
    if use_gpu and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)


class SegmentationEngine:
    """
    PaddleSeg-compatible ONNX Segmentation Engine.

    Tối ưu hoá matching PaddleSeg predict.py:
    - Letterbox resize (giữ aspect ratio, padding) thay vì stretch
    - Test-Time Augmentation (TTA): multi-scale + flip
    - Normalize khớp PaddleSeg (mean=0.5, std=0.5)
    - Postprocess đúng: bilinear cho logits, nearest cho argmax mask
    """

    def __init__(self, model_path, input_size=(1024, 1024),
                 use_tta=True, tta_scales=None, tta_flip_h=True, tta_flip_v=True):
        self.session = create_onnx_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size  # (H, W)

        # Kiểm tra output format (logits hay argmax)
        output_info = self.session.get_outputs()[0]
        self.output_dtype = output_info.type
        self.is_argmax_output = 'int' in self.output_dtype

        # TTA config
        self.use_tta = use_tta
        self.tta_scales = tta_scales or [1.0]
        self.tta_flip_h = tta_flip_h
        self.tta_flip_v = tta_flip_v

    def _normalize(self, img_rgb):
        """PaddleSeg Normalize: img/255.0, (img - mean) / std, mean=0.5, std=0.5"""
        img_float = img_rgb.astype(np.float32, copy=False) / 255.0
        img_float -= 0.5
        img_float /= 0.5
        return img_float

    def _letterbox_resize(self, img, target_size):
        """Letterbox resize giữ nguyên tỷ lệ, padding bằng 128."""
        th, tw = target_size
        h, w = img.shape[:2]
        scale = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_top = (th - new_h) // 2
        pad_left = (tw - new_w) // 2
        canvas = np.full((th, tw, 3), 128, dtype=np.uint8)
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = img_resized
        return canvas, scale, pad_top, pad_left, new_h, new_w

    def preprocess(self, img):
        """BGR→RGB → Letterbox resize → Normalize(0.5, 0.5) → CHW → batch"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_padded, scale, pad_top, pad_left, new_h, new_w = self._letterbox_resize(
            img_rgb, self.input_size)
        img_float = self._normalize(img_padded)
        tensor = np.expand_dims(img_float.transpose((2, 0, 1)), axis=0).astype(np.float32)
        return tensor, scale, pad_top, pad_left, new_h, new_w

    def _unpad_mask(self, mask, pad_top, pad_left, new_h, new_w, orig_h, orig_w):
        """Loại bỏ padding và resize mask về kích thước gốc (nearest)."""
        mask_cropped = mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
        return cv2.resize(mask_cropped.astype(np.uint8), (orig_w, orig_h),
                          interpolation=cv2.INTER_NEAREST)

    def _unpad_logits(self, logits, pad_top, pad_left, new_h, new_w, orig_h, orig_w):
        """Loại bỏ padding và resize logits bằng bilinear (chất lượng cao)."""
        num_classes = logits.shape[0]
        logits_cropped = logits[:, pad_top:pad_top + new_h, pad_left:pad_left + new_w]
        logits_resized = np.zeros((num_classes, orig_h, orig_w), dtype=np.float32)
        for c in range(num_classes):
            logits_resized[c] = cv2.resize(logits_cropped[c], (orig_w, orig_h),
                                           interpolation=cv2.INTER_LINEAR)
        return logits_resized

    def _run_single(self, tensor):
        return self.session.run(None, {self.input_name: tensor})[0]

    def _flip_tensor(self, tensor, flip_h, flip_v):
        if flip_h:
            tensor = tensor[:, :, :, ::-1].copy()
        if flip_v:
            tensor = tensor[:, :, ::-1, :].copy()
        return tensor

    def _flip_output(self, output, flip_h, flip_v):
        output = np.squeeze(output)
        if output.ndim > 2:
            if flip_h:
                output = output[:, :, ::-1].copy()
            if flip_v:
                output = output[:, ::-1, :].copy()
        else:
            if flip_h:
                output = output[:, ::-1].copy()
            if flip_v:
                output = output[::-1, :].copy()
        return output

    def postprocess(self, pred, pad_info, original_shape):
        """Unpad + resize: bilinear cho logits, nearest cho argmax mask."""
        scale, pad_top, pad_left, new_h, new_w = pad_info
        h_orig, w_orig = original_shape
        pred = np.squeeze(pred)
        if pred.ndim > 2:
            logits_resized = self._unpad_logits(
                pred, pad_top, pad_left, new_h, new_w, h_orig, w_orig)
            mask = np.argmax(logits_resized, axis=0).astype(np.uint8)
        else:
            mask = self._unpad_mask(pred, pad_top, pad_left, new_h, new_w, h_orig, w_orig)
        return mask

    def _get_flip_combinations(self):
        combos = [(False, False)]
        if self.tta_flip_h:
            combos.append((True, False))
        if self.tta_flip_v:
            combos.append((False, True))
            if self.tta_flip_h:
                combos.append((True, True))
        return combos

    def infer(self, frame):
        """Inference với TTA matching PaddleSeg aug_inference."""
        h, w = frame.shape[:2]

        if not self.use_tta or (len(self.tta_scales) == 1 and self.tta_scales[0] == 1.0
                                 and not self.tta_flip_h and not self.tta_flip_v):
            tensor, scale, pad_top, pad_left, new_h, new_w = self.preprocess(frame)
            output = self._run_single(tensor)
            return self.postprocess(output, (scale, pad_top, pad_left, new_h, new_w), (h, w))

        flip_combos = self._get_flip_combinations()
        if self.is_argmax_output:
            return self._tta_majority_voting(frame, flip_combos)
        else:
            return self._tta_logits_averaging(frame, flip_combos)

    def _tta_majority_voting(self, frame, flip_combos):
        h, w = frame.shape[:2]
        num_classes = 8
        vote_map = np.zeros((num_classes, h, w), dtype=np.float32)
        for scale in self.tta_scales:
            scaled_size = (int(self.input_size[0] * scale), int(self.input_size[1] * scale))
            for flip_h, flip_v in flip_combos:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_padded, s, pt, pl, nh, nw = self._letterbox_resize(img_rgb, scaled_size)
                img_float = self._normalize(img_padded)
                tensor = np.expand_dims(img_float.transpose((2, 0, 1)), axis=0).astype(np.float32)
                tensor = self._flip_tensor(tensor, flip_h, flip_v)
                output = self._run_single(tensor)
                output = self._flip_output(output, flip_h, flip_v)
                mask = self._unpad_mask(output, pt, pl, nh, nw, h, w)
                for c in range(num_classes):
                    vote_map[c] += (mask == c).astype(np.float32)
        return np.argmax(vote_map, axis=0).astype(np.uint8)

    def _tta_logits_averaging(self, frame, flip_combos):
        h, w = frame.shape[:2]
        accumulated_logits = None
        count = 0
        for scale in self.tta_scales:
            scaled_size = (int(self.input_size[0] * scale), int(self.input_size[1] * scale))
            for flip_h, flip_v in flip_combos:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_padded, s, pt, pl, nh, nw = self._letterbox_resize(img_rgb, scaled_size)
                img_float = self._normalize(img_padded)
                tensor = np.expand_dims(img_float.transpose((2, 0, 1)), axis=0).astype(np.float32)
                tensor = self._flip_tensor(tensor, flip_h, flip_v)
                output = self._run_single(tensor)
                output = np.squeeze(output)
                output = self._flip_output(output, flip_h, flip_v)
                logits_resized = self._unpad_logits(output, pt, pl, nh, nw, h, w)
                if accumulated_logits is None:
                    accumulated_logits = logits_resized
                else:
                    accumulated_logits += logits_resized
                count += 1
        accumulated_logits /= count
        return np.argmax(accumulated_logits, axis=0).astype(np.uint8)


class DetectionEngine:
    def __init__(self, model_path, input_size=960, conf_threshold=0.35, iou_threshold=0.45):
        self.session = create_onnx_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if self.output_shape and len(self.output_shape) == 3:
            if self.output_shape[1] is not None and self.output_shape[2] is not None:
                if isinstance(self.output_shape[1], int) and isinstance(self.output_shape[2], int):
                    self.is_v8_format = self.output_shape[1] < self.output_shape[2]
                else:
                    self.is_v8_format = True
            else:
                self.is_v8_format = True
        else:
            self.is_v8_format = True

    def preprocess(self, img):
        h, w = img.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img_resized

        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = canvas_rgb.astype(np.float32) / 255.0
        blob = blob.transpose((2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob, scale, pad_w, pad_h

    def postprocess(self, output, scale, pad_w, pad_h, orig_h, orig_w):
        pred = output[0]
        if self.is_v8_format and pred.shape[1] < pred.shape[2]:
            pred = pred.transpose(0, 2, 1)
        pred = np.squeeze(pred)

        if pred.ndim == 1:
            pred = np.expand_dims(pred, axis=0)

        if pred.shape[1] == 5:
            boxes_cxcywh = pred[:, :4]
            scores = pred[:, 4]
        else:
            boxes_cxcywh = pred[:, :4]
            class_scores = pred[:, 4:]
            scores = np.max(class_scores, axis=1)

        keep_mask = scores > self.conf_threshold
        boxes_cxcywh = boxes_cxcywh[keep_mask]
        scores = scores[keep_mask]

        if len(boxes_cxcywh) == 0:
            return np.array([]), np.array([])

        boxes_xyxy = np.zeros_like(boxes_cxcywh)
        boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / scale
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        indices = self._nms(boxes_xyxy, scores, self.iou_threshold)
        return boxes_xyxy[indices], scores[indices]

    def _nms(self, boxes, scores, iou_threshold):
        if len(boxes) == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

    def infer(self, frame):
        h, w = frame.shape[:2]
        blob, scale, pad_w, pad_h = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: blob})
        return self.postprocess(outputs, scale, pad_w, pad_h, h, w)


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.frame_count = 0
        self.total_seg_time = 0.0
        self.total_det_time = 0.0
        self.total_time = 0.0
        self.alert_frames = 0
        self.total_persons = 0
        self.total_on_road = 0
        self.fps_history = []
        self.start_time = time.time()

    def update(self, frame_info):
        self.frame_count += 1
        self.total_seg_time += frame_info["t_seg"]
        self.total_det_time += frame_info["t_det"]
        self.total_time += frame_info["t_total"]
        self.total_persons += frame_info["num_persons"]
        self.total_on_road += frame_info["num_on_road"]
        if frame_info["alert"]:
            self.alert_frames += 1
        self.fps_history.append(1.0 / max(frame_info["t_total"], 1e-6))

    def get_summary(self):
        wall_time = time.time() - self.start_time
        return {
            "total_frames": self.frame_count,
            "wall_time_sec": round(wall_time, 2),
            "avg_fps": round(self.frame_count / max(wall_time, 1e-6), 2),
            "avg_seg_ms": round(self.total_seg_time / max(self.frame_count, 1) * 1000, 1),
            "avg_det_ms": round(self.total_det_time / max(self.frame_count, 1) * 1000, 1),
            "avg_pipeline_ms": round(self.total_time / max(self.frame_count, 1) * 1000, 1),
            "min_fps": round(min(self.fps_history) if self.fps_history else 0, 2),
            "max_fps": round(max(self.fps_history) if self.fps_history else 0, 2),
            "total_persons_detected": int(self.total_persons),
            "total_on_road_events": int(self.total_on_road),
            "alert_frames": int(self.alert_frames),
            "alert_ratio": round(self.alert_frames / max(self.frame_count, 1) * 100, 1),
        }


class LanePersonPipeline:
    def __init__(self, seg_model_path, det_model_path, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.seg_engine = SegmentationEngine(
            seg_model_path,
            input_size=self.config["seg_input_size"],
            use_tta=self.config.get("use_tta", True),
            tta_scales=self.config.get("tta_scales", [1.0]),
            tta_flip_h=self.config.get("tta_flip_h", True),
            tta_flip_v=self.config.get("tta_flip_v", True),
        )
        self.det_engine = DetectionEngine(
            det_model_path,
            input_size=self.config["det_input_size"],
            conf_threshold=self.config["det_conf_threshold"],
            iou_threshold=self.config["det_iou_threshold"],
        )
        self.metrics = MetricsTracker()

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        frame_info = {
            "num_persons": 0,
            "num_on_road": 0,
            "alert": False,
        }

        t_seg = time.time()
        seg_mask = self.seg_engine.infer(frame)
        t_seg = time.time() - t_seg

        road_mask = (seg_mask == self.config["road_class_id"]).astype(np.uint8) * 255
        kernel = np.ones((self.config["road_dilate_px"], self.config["road_dilate_px"]), np.uint8)
        danger_zone = cv2.dilate(road_mask, kernel, iterations=1)

        t_det = time.time()
        boxes, scores = self.det_engine.infer(frame)
        t_det = time.time() - t_det

        frame_info["num_persons"] = len(boxes)
        persons_on_road = []
        persons_safe = []

        if len(boxes) > 0:
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                person_region = danger_zone[y1:y2, x1:x2]
                if person_region.size == 0:
                    continue

                overlap_pixels = np.count_nonzero(person_region)
                total_pixels = person_region.size
                overlap_ratio = overlap_pixels / max(total_pixels, 1)

                if overlap_ratio > self.config["overlap_ratio_threshold"]:
                    persons_on_road.append((box, score, overlap_ratio))
                else:
                    persons_safe.append((box, score, overlap_ratio))

        frame_info["num_on_road"] = len(persons_on_road)
        frame_info["alert"] = len(persons_on_road) > 0
        frame_info["t_seg"] = t_seg
        frame_info["t_det"] = t_det
        frame_info["t_total"] = t_seg + t_det

        vis_frame = self._visualize(frame, seg_mask, road_mask, danger_zone, persons_on_road, persons_safe, frame_info)
        self.metrics.update(frame_info)
        return vis_frame, frame_info

    def _visualize(self, frame, seg_mask, road_mask, danger_zone, persons_on_road, persons_safe, info):
        h, w = frame.shape[:2]
        vis = frame.copy()

        result = seg_mask.astype(np.uint8)
        c1 = cv2.LUT(result, COLOR_MAP[:, 0].astype(np.uint8))
        c2 = cv2.LUT(result, COLOR_MAP[:, 1].astype(np.uint8))
        c3 = cv2.LUT(result, COLOR_MAP[:, 2].astype(np.uint8))
        pseudo_color = np.dstack((c3, c2, c1))
        vis = cv2.addWeighted(vis, 0.7, pseudo_color, 0.3, 0)

        road_overlay = np.zeros_like(vis)
        road_overlay[road_mask > 0] = COLORS["road"]
        vis = cv2.addWeighted(vis, 1.0, road_overlay, 0.15, 0)

        contours_dz, _ = cv2.findContours(danger_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours_dz, -1, COLORS["warning"], 1)

        for box, score, _ in persons_safe:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS["safe"], 2)
            self._draw_label(vis, f"Person {score:.0%}", x1, y1, COLORS["safe"])

        for box, score, overlap in persons_on_road:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS["danger"], 3)
            self._draw_label(vis, f"DANGER {score:.0%} | Overlap {overlap:.0%}", x1, y1, COLORS["danger"])

        if info["alert"]:
            cv2.rectangle(vis, (0, 0), (w - 1, h - 1), COLORS["danger"], 8)
            banner_h = 70
            overlay_banner = vis[:banner_h, :].copy()
            cv2.rectangle(vis, (0, 0), (w, banner_h), (0, 0, 180), -1)
            vis[:banner_h, :] = cv2.addWeighted(vis[:banner_h, :], 0.7, overlay_banner, 0.3, 0)
            warn_text = f"CANH BAO: {info['num_on_road']} NGUOI TREN LAN DUONG"
            cv2.putText(vis, warn_text, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        self._draw_hud(vis, info)
        return vis

    def _draw_label(self, img, text, x, y, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        y_label = max(0, y - 8)
        cv2.rectangle(img, (x, y_label - th - 4), (x + tw + 4, y_label + 4), color, -1)
        cv2.putText(img, text, (x + 2, y_label), font, font_scale, (255, 255, 255), thickness)

    def _draw_hud(self, vis, info):
        h, w = vis.shape[:2]
        hud_w, hud_h = 295, 130
        hud_x = w - hud_w - 10
        hud_y = h - hud_h - 10

        overlay = vis.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (30, 30, 30), -1)
        vis[:] = cv2.addWeighted(overlay, 0.7, vis, 0.3, 0)

        lines = [
            f"FPS: {1.0 / max(info['t_total'], 1e-6):.1f}",
            f"Seg: {info['t_seg'] * 1000:.0f}ms | Det: {info['t_det'] * 1000:.0f}ms",
            f"Persons: {info['num_persons']}",
            f"On Road: {info['num_on_road']}",
        ]
        for i, line in enumerate(lines):
            color = COLORS["danger"] if i == 3 and info["num_on_road"] > 0 else (200, 200, 200)
            cv2.putText(vis, line, (hud_x + 10, hud_y + 25 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


@lru_cache(maxsize=4)
def get_pipeline_cached(seg_model_path, det_model_path, config_signature):
    config = json.loads(config_signature)
    return LanePersonPipeline(seg_model_path, det_model_path, config)


def process_image_file(pipeline, input_path, output_path):
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Không đọc được ảnh đầu vào.")

    pipeline.metrics.reset()
    result, info = pipeline.process_frame(image)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ok = cv2.imwrite(output_path, result)
    if not ok:
        raise RuntimeError("Không ghi được ảnh đầu ra.")

    summary = pipeline.metrics.get_summary()
    summary["single_frame"] = info
    return summary


def process_video_file(pipeline, input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_vid, (w, h))
    pipeline.metrics.reset()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            vis_frame, _ = pipeline.process_frame(frame)
            out.write(vis_frame)
    finally:
        cap.release()
        out.release()

    summary = pipeline.metrics.get_summary()
    summary["input_video"] = {
        "total_frames": total_frames,
        "fps": round(float(fps_vid), 2),
        "width": w,
        "height": h,
    }

    metrics_path = output_path.rsplit('.', 1)[0] + '_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary, metrics_path
