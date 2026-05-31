import psutil
import time
import random
import platform
import os


class ContextEvaluator:
    """
    FIXED VERSION:
    - Always provides required keys
    - Ensures numeric values for ModelSelector
    - Prevents KeyError + type mismatch errors
    """

    def __init__(self):
        self.start_time = time.time()

    # =========================================================
    # SYSTEM CONTEXT
    # =========================================================
    def get_system_context(self):
        return {
            "cpu_usage": float(psutil.cpu_percent(interval=0.1)),
            "memory_usage": float(psutil.virtual_memory().percent),
            "memory_available_gb": float(psutil.virtual_memory().available / (1024 ** 3)),
            "disk_usage": float(psutil.disk_usage("/").percent),
            "system_platform": platform.system(),
            "processor": platform.processor()
        }

    # =========================================================
    # GPU CONTEXT
    # =========================================================
    def get_gpu_context(self):
        gpu_usage = 0.0

        try:
            if os.path.exists("/sys/devices/gpu.0/load"):
                with open("/sys/devices/gpu.0/load", "r") as f:
                    gpu_usage = float(int(f.read()) / 10)
        except:
            pass

        return {
            "gpu_available": False,
            "gpu_usage": gpu_usage,
            "gpu_memory_used_mb": 0.0
        }

    # =========================================================
    # UAV CONTEXT
    # =========================================================
    def get_uav_context(self):
        return {
            "battery": float(random.randint(25, 100)),
            "altitude": float(random.uniform(5, 50)),
            "speed": float(random.uniform(1, 15)),
            "gps_signal_strength": random.choice(["weak", "medium", "strong"])
        }

    # =========================================================
    # ENVIRONMENT CONTEXT (IMPORTANT FIX HERE)
    # =========================================================
    def get_environment_context(self):
        visibility_map = {"low": 0.2, "medium": 0.5, "high": 0.9}

        visibility_str = random.choice(["low", "medium", "high"])
        wind_str = random.choice(["low", "medium", "high"])
        light_str = random.choice(["low", "medium", "high"])

        return {
            "wind": wind_str,
            "light": light_str,
            "rain": random.choice([0.0, 0.3, 0.8]),
            "temperature": float(random.uniform(18, 38)),
            "visibility": visibility_map[visibility_str],
        }

    # =========================================================
    # MISSION CONTEXT
    # =========================================================
    def get_mission_context(self):
        return {
            "mission": "flood_monitoring",
            "priority": {"low": 0.2, "medium": 0.5, "high": 0.9}[
                random.choice(["low", "medium", "high"])
            ],
            "target": "flood_zone_alpha"
        }

    # =========================================================
    # RUNTIME CONTEXT
    # =========================================================
    def get_runtime_context(self):
        return {
            "uptime_seconds": float(time.time() - self.start_time)
        }

    # =========================================================
    # FINAL CONTEXT OUTPUT
    # =========================================================
    def get_context(self):
        context = {}

        context.update(self.get_system_context())
        context.update(self.get_gpu_context())
        context.update(self.get_uav_context())
        context.update(self.get_environment_context())
        context.update(self.get_mission_context())
        context.update(self.get_runtime_context())

        context.setdefault("flood_ratio", 0.0)
        context["visibility"] = float(context["visibility"])

        return context
