import torch
import gc

from ultralytics import YOLO
from pathlib import Path

from core.cuda_runtime import require_cuda
from core.runtime_profiler import RuntimeProfiler


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

                print(
                    "Loading YOLOv8 model..."
                )

                model = YOLO("yolov8n.pt")

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

                print(
                    "Loading Flood "
                    "Classification Model..."
                )

                # -----------------------------------------
                # MODEL PATH
                # -----------------------------------------
                model_path = (

                    self.base_dir /

                    "models/flood_classifier/"
                    "flood_resnet18.pth"

                ).resolve()

                if not model_path.exists():

                    raise FileNotFoundError(

                        "Classifier model "
                        f"not found at {model_path}"
                    )

                # -----------------------------------------
                # IMPORT MODEL
                # -----------------------------------------
                from models.flood_classifier.realtime_flood_detection import Net

                # -----------------------------------------
                # CREATE MODEL
                # -----------------------------------------
                model = Net()

                # -----------------------------------------
                # LOAD WEIGHTS
                # -----------------------------------------
                state_dict = torch.load(
                    model_path,
                    map_location=self.device
                )

                # -----------------------------------------
                # CLEAN STATE DICT
                # -----------------------------------------
                cleaned = {}

                for k, v in state_dict.items():

                    cleaned[
                        k.replace("model.", "")
                    ] = v

                # -----------------------------------------
                # LOAD PARAMETERS
                # -----------------------------------------
                model.load_state_dict(
                    cleaned,
                    strict=False
                )

                # -----------------------------------------
                # MOVE TO CUDA
                # -----------------------------------------
                model.to(self.device)

                model.eval()

                # -----------------------------------------
                # VERIFY DEVICE
                # -----------------------------------------
                print(
                    "[CLASSIFIER] Running on:",
                    next(
                        model.parameters()
                    ).device
                )

                return model

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

        if "flood_segmenter" not in self.models:

            def _load():

                print(
                    "Loading Flood "
                    "Segmentation Model "
                    "(MobileNetV3 DeepLabV3)..."
                )

                # -----------------------------------------
                # MODEL PATH
                # -----------------------------------------
                model_path = (

                    self.base_dir /

                    "models/flood_segmentation/"
                    "DeepLabv3_plus/"
                    "flood_segmentation/"
                    "best_model.pth"

                ).resolve()

                if not model_path.exists():

                    raise FileNotFoundError(

                        "Segmentation model "
                        f"not found at {model_path}"
                    )

                # -----------------------------------------
                # IMPORT MODEL
                # -----------------------------------------
                from torchvision.models.segmentation import (
                    deeplabv3_mobilenet_v3_large
                )

                # -----------------------------------------
                # CREATE MODEL
                # -----------------------------------------
                model = (
                    deeplabv3_mobilenet_v3_large(
                        weights=None
                    )
                )

                # -----------------------------------------
                # BINARY OUTPUT
                # -----------------------------------------
                model.classifier[4] = (
                    torch.nn.Conv2d(
                        256,
                        2,
                        kernel_size=1
                    )
                )

                # -----------------------------------------
                # LOAD WEIGHTS
                # -----------------------------------------
                state_dict = torch.load(
                    model_path,
                    map_location=self.device
                )

                model.load_state_dict(
                    state_dict,
                    strict=False
                )

                # -----------------------------------------
                # MOVE TO CUDA
                # -----------------------------------------
                model.to(self.device)

                model.eval()

                # -----------------------------------------
                # VERIFY DEVICE
                # -----------------------------------------
                print(
                    "[SEGMENTER] Running on:",
                    next(
                        model.parameters()
                    ).device
                )

                return model

            self.models["flood_segmenter"] = (

                self.profile_model_load(
                    "flood_segmenter",
                    _load
                )
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
