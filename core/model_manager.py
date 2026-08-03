import torch
import gc

from ultralytics import YOLO
from pathlib import Path

from core.cuda_runtime import require_cuda
from core.perf_config import USE_TENSORRT, USE_TORCH_COMPILE, YOLO_IMGSZ, YOLO_ROBUST_IMGSZ
from core.human_detector_tier import get_tier
from core.flood_segmenter_tier import get_tier as get_seg_tier
from core.flood_models import build_deeplab_segmenter, build_resnet18_classifier
from core.runtime_profiler import RuntimeProfiler

CLF_ENGINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "models/flood_classifier/flood_resnet18.engine"
)
SEG_DIR = (
    Path(__file__).resolve().parents[1]
    / "models/flood_segmentation/DeepLabv3_plus/flood_segmentation"
)
SEG_LIGHTWEIGHT_ENGINE = SEG_DIR / "flood_deeplab.engine"
SEG_LIGHTWEIGHT_WEIGHTS = SEG_DIR / "best_model.pth"
SEG_ROBUST_ENGINE = SEG_DIR / "flood_deeplab_robust.engine"
SEG_ROBUST_WEIGHTS = SEG_DIR / "best_model_robust_floodnet.pth"
# Legacy alias
SEG_ENGINE_PATH = SEG_LIGHTWEIGHT_ENGINE


ROBUST_HUMAN_PT = (
    Path(__file__).resolve().parents[1]
    / "models/human_detector/yolo11s_visdrone_human_1280.pt"
)
ROBUST_HUMAN_ENGINE = (
    Path(__file__).resolve().parents[1]
    / "models/human_detector/yolo11s_visdrone_human_1280.engine"
)

# VisDrone human classes merged at inference as "human".
VISDRONE_HUMAN_CLASS_IDS = (0, 1)  # pedestrian, people
COCO_PERSON_CLASS_ID = 0


class ModelManager:

    def __init__(self):

        # =====================================================
        # MODEL CACHE
        # =====================================================
        self.models = {}

        self.device = require_cuda()
        print(f"[MODEL MANAGER] Device: {self.device}")
        print(f"[CUDA] GPU: {torch.cuda.get_device_name(0)}")

        # =====================================================
        # PROFILER
        # =====================================================
        self.profiler = RuntimeProfiler()

        # =====================================================
        # PROJECT ROOT
        # =====================================================
        self.base_dir = (
            Path(__file__)
            .resolve()
            .parents[1]
        )
        self.human_imgsz = YOLO_IMGSZ
        self.human_backend = "yolov8n"
        self.human_tier = "lightweight"
        self.human_class_ids = (COCO_PERSON_CLASS_ID,)
        self.human_detector_key = "yolov8n"
        self.clf_backend = "pytorch"
        self.seg_backend = "pytorch"
        self.seg_tier = "lightweight"
        self.seg_weights_key = "best_model.pth"
        self.seg_cache_sig: tuple | None = None

    def _maybe_compile(self, model, label: str):
        if not USE_TORCH_COMPILE:
            return model
        try:
            compiled = torch.compile(model, mode="reduce-overhead")
            print(f"[{label}] torch.compile enabled")
            return compiled
        except Exception as exc:
            print(f"[{label}] torch.compile skipped: {exc}")
            return model

    def _resolve_human_weights(self) -> tuple[Path, str]:
        tier = get_tier()

        if tier == "robust":
            self.human_tier = "robust"
            self.human_imgsz = YOLO_ROBUST_IMGSZ
            self.human_class_ids = VISDRONE_HUMAN_CLASS_IDS
            self.human_detector_key = "yolo11s_visdrone_human_1280"
            if USE_TENSORRT and ROBUST_HUMAN_ENGINE.exists():
                return ROBUST_HUMAN_ENGINE, "tensorrt"
            if ROBUST_HUMAN_PT.exists():
                return ROBUST_HUMAN_PT, "pytorch"
            print("[HUMAN] robust weights missing; falling back to lightweight yolov8n")

        self.human_tier = "lightweight"
        self.human_imgsz = YOLO_IMGSZ
        self.human_class_ids = (COCO_PERSON_CLASS_ID,)
        self.human_detector_key = "yolov8n"
        engine_path = self.base_dir / "yolov8n.engine"
        pt_path = self.base_dir / "yolov8n.pt"
        if USE_TENSORRT and engine_path.exists():
            return engine_path, "tensorrt"
        if pt_path.exists():
            return pt_path, "pytorch"
        return Path("yolov8n.pt"), "pytorch"

    def _clf_weights_path(self) -> Path:
        return (self.base_dir / "models/flood_classifier/flood_resnet18.pth").resolve()

    def _resolve_seg_paths(self, tier: str | None = None) -> tuple[Path, str]:
        tier = tier or get_seg_tier()
        self.seg_tier = tier

        if tier == "robust":
            self.seg_weights_key = "best_model_robust_floodnet.pth"
            if USE_TENSORRT and SEG_ROBUST_ENGINE.exists():
                return SEG_ROBUST_ENGINE, "tensorrt"
            if SEG_ROBUST_WEIGHTS.exists():
                return SEG_ROBUST_WEIGHTS, "pytorch"
            print("[FLOOD SEG] robust weights missing; falling back to lightweight")

        self.seg_tier = "lightweight"
        self.seg_weights_key = "best_model.pth"
        if USE_TENSORRT and SEG_LIGHTWEIGHT_ENGINE.exists():
            return SEG_LIGHTWEIGHT_ENGINE, "tensorrt"
        weights = SEG_LIGHTWEIGHT_WEIGHTS
        if weights.exists():
            return weights, "pytorch"
        return weights, "pytorch"

    # =====================================================
    # PROFILE LOAD
    # =====================================================
    def profile_model_load(
        self,
        model_name,
        load_function
    ):

        start_time = (
            self.profiler
            .start_load_timer()
        )

        model = load_function()

        elapsed = (
            self.profiler
            .stop_load_timer(start_time)
        )

        print(
            f"[LOAD] {model_name} "
            f"loaded in {elapsed:.4f} sec"
        )

        return model

    # =====================================================
    # UNLOAD
    # =====================================================
    def unload_model(self, model_name):

        if model_name in self.models:

            start_time = (
                self.profiler
                .start_unload_timer()
            )

            # ---------------------------------------------
            # DELETE MODEL
            # ---------------------------------------------
            del self.models[model_name]

            # ---------------------------------------------
            # CLEAN MEMORY
            # ---------------------------------------------
            gc.collect()

            torch.cuda.empty_cache()

            elapsed = (
                self.profiler
                .stop_unload_timer(start_time)
            )

            print(
                f"[UNLOAD] {model_name} "
                f"unloaded in {elapsed:.4f} sec"
            )

            return elapsed

        return 0

    # =====================================================
    # SWITCH
    # =====================================================
    def switch_model(
        self,
        old_model_name,
        new_model_name,
        load_function
    ):

        switch_start = (
            self.profiler
            .start_switch_timer()
        )

        # ---------------------------------------------
        # UNLOAD OLD
        # ---------------------------------------------
        self.unload_model(old_model_name)

        # ---------------------------------------------
        # LOAD NEW
        # ---------------------------------------------
        model = self.profile_model_load(
            new_model_name,
            load_function
        )

        self.models[new_model_name] = model

        switch_latency = (
            self.profiler
            .stop_switch_timer(switch_start)
        )

        print(
            f"[SWITCH] "
            f"{old_model_name} → "
            f"{new_model_name} "
            f"in {switch_latency:.4f} sec"
        )

        return model

    # =====================================================
    # YOLO DETECTOR
    # =====================================================
    def load_human_detector(self):

        if "human_detector" not in self.models:

            def _load():
                weights, backend = self._resolve_human_weights()
                print(f"Loading human detector ({self.human_tier})...")
                model = YOLO(str(weights))
                self.human_backend = backend
                print(
                    f"[HUMAN] tier={self.human_tier} backend={backend} "
                    f"weights={weights.name} imgsz={self.human_imgsz} "
                    f"classes={self.human_class_ids}"
                )
                return model

            self.models["human_detector"] = (
                self.profile_model_load(
                    "human_detector",
                    _load
                )
            )

        return self.models["human_detector"]

    # =====================================================
    # FLOOD CLASSIFIER (RESNET18)
    # =====================================================
    def load_flood_classifier(self):

        if "flood_classifier" not in self.models:

            def _load():
                print("Loading Flood Classification Model...")
                engine_path = CLF_ENGINE_PATH
                if USE_TENSORRT and engine_path.exists():
                    from core.trt_runner import TrtFloodClassifier

                    self.clf_backend = "tensorrt"
                    print(f"[CLASSIFIER] TensorRT {engine_path.name}")
                    return TrtFloodClassifier(engine_path, "CLASSIFIER")

                weights = self._clf_weights_path()
                if not weights.exists():
                    raise FileNotFoundError(f"Classifier not found at {weights}")

                model = build_resnet18_classifier(weights, self.device)
                self.clf_backend = "pytorch"
                print("[CLASSIFIER] PyTorch", next(model.parameters()).device)
                return self._maybe_compile(model, "CLASSIFIER")

            self.models["flood_classifier"] = (

                self.profile_model_load(
                    "flood_classifier",
                    _load
                )
            )

        return self.models["flood_classifier"]

    # =====================================================
    # FLOOD SEGMENTER
    # =====================================================
    def load_flood_segmenter(self):
        weights_path, backend = self._resolve_seg_paths()
        cache_sig = (self.seg_tier, weights_path.name, backend)
        if (
            "flood_segmenter" in self.models
            and self.seg_cache_sig == cache_sig
        ):
            return self.models["flood_segmenter"]

        if "flood_segmenter" in self.models:
            self.unload_model("flood_segmenter")

        def _load():
            print(
                f"Loading Flood Segmentation ({self.seg_tier}) "
                f"DeepLabv3+ MobileNetV3..."
            )
            if backend == "tensorrt":
                from core.trt_runner import TrtFloodSegmenter

                self.seg_backend = "tensorrt"
                print(f"[SEGMENTER] tier={self.seg_tier} TensorRT {weights_path.name}")
                return TrtFloodSegmenter(weights_path, "SEGMENTER")

            if not weights_path.exists():
                raise FileNotFoundError(f"Segmentation model not found at {weights_path}")

            model = build_deeplab_segmenter(weights_path, self.device)
            self.seg_backend = "pytorch"
            print(
                f"[SEGMENTER] tier={self.seg_tier} PyTorch "
                f"{weights_path.name} {next(model.parameters()).device}"
            )
            return self._maybe_compile(model, "SEGMENTER")

        self.seg_cache_sig = cache_sig
        self.models["flood_segmenter"] = self.profile_model_load(
            "flood_segmenter",
            _load,
        )
        return self.models["flood_segmenter"]

    # =====================================================
    # METRICS
    # =====================================================
    def get_runtime_metrics(self):

        return (
            self.profiler
            .get_report()
        )
