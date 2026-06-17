"""TensorRT inference for flood models (ResNet18 / DeepLab) on Jetson."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from core.cuda_runtime import require_cuda

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INV255 = np.float32(1.0 / 255.0)


def _preprocess_bgr(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img *= _INV255
    img -= _IMAGENET_MEAN
    img /= _IMAGENET_STD
    return np.transpose(img, (2, 0, 1))[np.newaxis, ...].astype(np.float32)


class TrtEngine:
    """TensorRT 10.x runner using PyTorch CUDA tensors."""

    def __init__(self, engine_path: Path, label: str):
        import tensorrt as trt

        self.engine_path = Path(engine_path)
        self.label = label
        self.device = require_cuda()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        self.input_name = None
        self.output_name = None
        self.input_tensor: torch.Tensor | None = None
        self.output_tensor: torch.Tensor | None = None

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = torch.float16 if dtype == np.float16 else torch.float32
            tensor = torch.empty(shape, dtype=torch_dtype, device=self.device)

            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_tensor = tensor
            else:
                self.output_name = name
                self.output_tensor = tensor

        if self.input_tensor is None or self.output_tensor is None:
            raise RuntimeError(f"Invalid engine IO: {self.engine_path}")

        device_index = self.device.index if self.device.index is not None else 0
        self._cuda_stream = torch.cuda.Stream(device=device_index)

        print(
            f"[TRT {label}] {self.engine_path.name} "
            f"in={tuple(self.input_tensor.shape)} out={tuple(self.output_tensor.shape)}"
        )

    def _infer_sync(self) -> np.ndarray:
        assert self.input_tensor is not None and self.output_tensor is not None
        self.context.set_tensor_address(self.input_name, self.input_tensor.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output_tensor.data_ptr())
        with torch.cuda.stream(self._cuda_stream):
            self.context.execute_async_v3(self._cuda_stream.cuda_stream)
        self._cuda_stream.synchronize()
        return self.output_tensor.detach().cpu().numpy()


class TrtFloodClassifier(TrtEngine):
    INPUT_SIZE = 224

    def run_classification(self, frame_bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()
        host = _preprocess_bgr(frame_bgr, self.INPUT_SIZE)
        self.input_tensor.copy_(
            torch.from_numpy(host).to(device=self.device, dtype=self.input_tensor.dtype)
        )
        out = self._infer_sync()
        pred = int(np.argmax(out, axis=1).item())
        ms = (time.perf_counter() - t0) * 1000.0
        return {
            "label": "Flooded" if pred == 0 else "Non-Flooded",
            "inference_ms": ms,
            "backend": "tensorrt",
        }


class TrtFloodSegmenter(TrtEngine):
    INPUT_SIZE = 256

    def run_segmentation(self, frame_bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()
        host = _preprocess_bgr(frame_bgr, self.INPUT_SIZE)
        self.input_tensor.copy_(
            torch.from_numpy(host).to(device=self.device, dtype=self.input_tensor.dtype)
        )
        out = self._infer_sync()
        logits = out[0] if out.ndim == 4 else out
        mask = np.argmax(logits, axis=0).astype(np.uint8)
        flood_ratio = float((mask == 1).mean())
        mask_full = cv2.resize(
            mask,
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        return {
            "flood_ratio": flood_ratio,
            "mask": mask_full,
            "inference_ms": ms,
            "backend": "tensorrt",
        }
