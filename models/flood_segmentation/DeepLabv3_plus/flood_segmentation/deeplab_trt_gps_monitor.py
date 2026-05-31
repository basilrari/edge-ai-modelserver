import cv2
import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import csv
from datetime import datetime

# -----------------------------
# GPS Conversion (Mocked Example)
# -----------------------------
def pixel_to_gps(x, y, frame_shape, base_gps=(12.9716, 77.5946), scale=0.00001):
    h, w = frame_shape[:2]
    lat = base_gps[0] + (y - h / 2) * scale
    lon = base_gps[1] + (x - w / 2) * scale
    return lat, lon

# -----------------------------
# TensorRT FP16 Engine Wrapper
# -----------------------------
class UNeTFP16Optimized:
    def __init__(self, engine_path="deeplab_fp16.engine"):
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self._allocate_buffers()

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        return self.runtime.deserialize_cuda_engine(engine_data)

    def _allocate_buffers(self):
        self.inputs = []
        self.outputs = []
        self.bindings = []

        print("Engine bindings:")
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.engine.get_tensor_shape(name))
            size = int(np.prod(shape))
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            host_mem = np.empty(size, dtype=dtype)
            binding = {"name": name, "host": host_mem, "device": device_mem, "shape": shape, "dtype": dtype}

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                print(f" - input: {shape}, dtype={dtype}")
                self.inputs.append(binding)
            else:
                print(f" - output: {shape}, dtype={dtype}")
                self.outputs.append(binding)

            self.bindings.append(int(device_mem))

    def infer(self, img_tensor):
        """Run TensorRT inference"""
        np.copyto(self.inputs[0]['host'], img_tensor.ravel())

        # H2D copy
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Set tensor addresses (TensorRT 10.x)
        for inp in self.inputs:
            self.context.set_tensor_address(inp["name"], inp["device"])
        for out in self.outputs:
            self.context.set_tensor_address(out["name"], out["device"])

        # Execute inference
        self.context.execute_async_v3(self.stream.handle)

        # D2H copy
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        # Reshape based on model output
        output_shape = self.outputs[0]["shape"]  # (1, 2, 256, 256)
        logits = self.outputs[0]["host"].reshape(output_shape)

        # Take argmax across class dimension -> binary mask (Flood=1)
        mask = np.argmax(logits, axis=1).astype(np.uint8)[0]
        return mask

# -----------------------------
# Main Flood Detection Monitor
# -----------------------------
class FloodDetectionMonitor:
    def __init__(self, engine_path="deeplab_fp16.engine"):
        self.model = UNeTFP16Optimized(engine_path)
        self.cap = cv2.VideoCapture(0)  # webcam
        self.prev_time = time.time()
        self.fps_window = []
        self.rolling_window = 30
        self.csv_file = open("inference_log.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Timestamp", "FrameTime(ms)", "InferenceTime(ms)", "TotalTime(ms)", "FPS", "Flood_Lat", "Flood_Lon"])

    def preprocess(self, frame):
        img = cv2.resize(frame, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def run(self):
        print("Starting Flood Detection Monitor...")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            start_total = time.time()
            img_tensor = self.preprocess(frame)
            start_infer = time.time()
            mask = self.model.infer(img_tensor)
            infer_time = (time.time() - start_infer) * 1000

            # Resize mask to original frame
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

            # Find flooded regions
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                center = (int(x), int(y))
                gps_lat, gps_lon = pixel_to_gps(int(x), int(y), frame.shape)
                cv2.circle(frame, center, int(radius), (0, 0, 255), 2)
                cv2.putText(frame, f"Flood @ ({gps_lat:.5f}, {gps_lon:.5f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                gps_lat, gps_lon = (0.0, 0.0)

            # Overlay mask
            overlay = cv2.addWeighted(frame, 0.7, cv2.applyColorMap(mask_resized * 255, cv2.COLORMAP_JET), 0.3, 0)

            # FPS calculation
            frame_time = (time.time() - start_total) * 1000
            fps = 1000.0 / frame_time if frame_time > 0 else 0
            self.fps_window.append(fps)
            if len(self.fps_window) > self.rolling_window:
                self.fps_window.pop(0)
            avg_fps = np.mean(self.fps_window)

            total_time = (time.time() - start_total) * 1000
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.csv_writer.writerow([timestamp, f"{frame_time:.2f}", f"{infer_time:.2f}",
                                      f"{total_time:.2f}", f"{avg_fps:.2f}",
                                      f"{gps_lat:.5f}", f"{gps_lon:.5f}"])
            self.csv_file.flush()

            # Display info
            cv2.putText(overlay, f"FPS: {avg_fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(overlay, f"Infer: {infer_time:.2f} ms", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Flood Detection Monitor", overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.csv_file.close()

# -----------------------------
# Run the Monitor
# -----------------------------
if __name__ == "__main__":
    monitor = FloodDetectionMonitor("deeplab_fp16.engine")
    monitor.run()

