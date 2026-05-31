import time

import cv2
import numpy as np
import torch

from core.cuda_runtime import require_cuda

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
        self._copy_preprocessed(frame, 224, self._clf_host, self._clf_input)
        output, inference_ms = self._forward_pass_ms(model, self._clf_input)
        pred = int(output.argmax(dim=1).item())
        # ImageFolder class order: 0=Flood, 1=Non_Flood
        return {
            "label": "Flooded" if pred == 0 else "Non-Flooded",
            "inference_ms": inference_ms,
        }

    def run_segmentation(self, model, frame):
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

    def run_detection(self, model, frame):
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            return model(frame)

    def warmup(self, clf_model, seg_model, frame, iterations=20):
        for _ in range(iterations):
            self.run_classification(clf_model, frame)
            self.run_segmentation(seg_model, frame)
        torch.cuda.synchronize()
        print(f"[INFERENCE ENGINE] GPU warmup complete ({iterations} iters)")
