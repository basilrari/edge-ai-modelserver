import time

import cv2
import numpy as np
import torch

from core.cuda_runtime import require_cuda
from core.gpu_runtime import cuda_stream, get_flood_stream, sync_all
from core.segment_policy import CLF_FLOOD_CLASS
from core.trt_runner import TrtFloodClassifier, TrtFloodSegmenter

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INV255 = np.float32(1.0 / 255.0)


class InferenceEngine:

    def __init__(self):
        self.device = require_cuda()
        print(f"[INFERENCE ENGINE] Device: {self.device}")

        self._clf_input = torch.empty(
            (1, 3, 224, 224), device=self.device, dtype=torch.float32
        )
        self._seg_input = torch.empty(
            (1, 3, 256, 256), device=self.device, dtype=torch.float32
        )

        self._clf_host = torch.empty(
            (3, 224, 224), pin_memory=True, dtype=torch.float32
        )
        self._seg_host = torch.empty(
            (3, 256, 256), pin_memory=True, dtype=torch.float32
        )

        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._flood_stream = get_flood_stream()

    def _copy_preprocessed(self, frame_bgr, size, host_tensor, device_tensor):
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img *= _INV255
        img -= _IMAGENET_MEAN
        img /= _IMAGENET_STD
        np.copyto(host_tensor.numpy(), np.transpose(img, (2, 0, 1)))
        device_tensor.copy_(host_tensor.unsqueeze(0), non_blocking=True)

    def _forward_pass_ms(self, model, device_tensor):
        self._start_event.record()
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(device_tensor)
        self._end_event.record()
        self._end_event.synchronize()
        return output, self._start_event.elapsed_time(self._end_event)

    def run_classification(self, model, frame):
        if isinstance(model, TrtFloodClassifier):
            return model.run_classification(frame)
        with cuda_stream(self._flood_stream):
            self._copy_preprocessed(frame, 224, self._clf_host, self._clf_input)
            output, inference_ms = self._forward_pass_ms(model, self._clf_input)
        pred = int(output.argmax(dim=1).item())
        return {
            "label": "Flooded" if pred == 0 else "Non-Flooded",
            "inference_ms": inference_ms,
        }

    def run_segmentation(self, model, frame):
        if isinstance(model, TrtFloodSegmenter):
            return model.run_segmentation(frame)
        with cuda_stream(self._flood_stream):
            self._copy_preprocessed(frame, 256, self._seg_host, self._seg_input)
            output, inference_ms = self._forward_pass_ms(model, self._seg_input)

        if isinstance(output, dict):
            output = output["out"]

        mask = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        flood_ratio = float((mask == 1).mean())
        mask_full = cv2.resize(
            mask,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        return {
            "flood_ratio": flood_ratio,
            "mask": mask_full,
            "inference_ms": inference_ms,
        }

    def run_flood_pipeline(self, clf_model, seg_model, frame, run_segmenter=True):
        """Classify first; run DeepLab only when run_segmenter=True."""
        if isinstance(clf_model, TrtFloodClassifier):
            clf_out = clf_model.run_classification(frame)
            classification_ms = clf_out.pop("inference_ms", 0.0)
            clf_pred = 0 if clf_out.get("label") == "Flooded" else 1
            clf_result = {**clf_out, "class_index": clf_pred}
        else:
            with cuda_stream(self._flood_stream):
                self._copy_preprocessed(frame, 224, self._clf_host, self._clf_input)
                output, classification_ms = self._forward_pass_ms(clf_model, self._clf_input)
            clf_pred = int(output.argmax(dim=1).item())
            clf_result = {
                "label": "Flooded" if clf_pred == CLF_FLOOD_CLASS else "Non-Flooded",
                "class_index": clf_pred,
            }

        seg_result = {}
        segmentation_ms = 0.0
        mask = None
        flood_ratio = 0.0

        if run_segmenter and seg_model is not None:
            if isinstance(seg_model, TrtFloodSegmenter):
                seg_out = seg_model.run_segmentation(frame)
                segmentation_ms = seg_out.pop("inference_ms", 0.0)
                mask = seg_out.pop("mask", None)
                flood_ratio = float(seg_out.get("flood_ratio", 0.0))
                seg_result = {"flood_ratio": flood_ratio, **seg_out}
            else:
                with cuda_stream(self._flood_stream):
                    self._copy_preprocessed(frame, 256, self._seg_host, self._seg_input)
                    output, segmentation_ms = self._forward_pass_ms(seg_model, self._seg_input)
                if isinstance(output, dict):
                    output = output["out"]
                mask_arr = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                flood_ratio = float((mask_arr == 1).mean())
                mask = cv2.resize(
                    mask_arr,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                seg_result = {"flood_ratio": flood_ratio}

        return {
            "classification": clf_result,
            "classification_ms": classification_ms,
            "clf_class_index": clf_pred,
            "segmentation": seg_result,
            "segmentation_ms": segmentation_ms,
            "segmentation_skipped": not (run_segmenter and seg_model is not None),
            "mask": mask,
            "flood_ratio": flood_ratio,
            "total_inference_ms": classification_ms + segmentation_ms,
        }

    def run_detection(self, model, frame):
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            return model(frame)

    def warmup(self, clf_model, seg_model, frame, iterations=10):
        for _ in range(iterations):
            self.run_flood_pipeline(clf_model, seg_model, frame, run_segmenter=True)
        sync_all()
        print(f"[INFERENCE ENGINE] GPU warmup complete ({iterations} iters)")
