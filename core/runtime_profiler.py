import time
import statistics


class RuntimeProfiler:

    def __init__(self):

        self.load_times = []
        self.unload_times = []
        self.switch_latencies = []

    # -----------------------------------
    # MODEL LOAD TIMER
    # -----------------------------------
    def start_load_timer(self):

        return time.time()

    def stop_load_timer(self, start_time):

        elapsed = time.time() - start_time
        self.load_times.append(elapsed)

        return elapsed

    # -----------------------------------
    # MODEL UNLOAD TIMER
    # -----------------------------------
    def start_unload_timer(self):

        return time.time()

    def stop_unload_timer(self, start_time):

        elapsed = time.time() - start_time
        self.unload_times.append(elapsed)

        return elapsed

    # -----------------------------------
    # SWITCH LATENCY TIMER
    # -----------------------------------
    def start_switch_timer(self):

        return time.time()

    def stop_switch_timer(self, start_time):

        elapsed = time.time() - start_time
        self.switch_latencies.append(elapsed)

        return elapsed

    # -----------------------------------
    # STATISTICS
    # -----------------------------------
    def get_statistics(self, values):

        if len(values) == 0:
            return None

        values_sorted = sorted(values)

        return {
            "mean": statistics.mean(values_sorted),
            "p95": values_sorted[int(0.95 * len(values_sorted)) - 1],
            "p99": values_sorted[int(0.99 * len(values_sorted)) - 1],
            "max": max(values_sorted),
            "min": min(values_sorted)
        }

    # -----------------------------------
    # EXPORT RESULTS
    # -----------------------------------
    def get_report(self):

        return {

            "model_load_time": self.get_statistics(self.load_times),

            "model_unload_time": self.get_statistics(self.unload_times),

            "model_switch_latency": self.get_statistics(
                self.switch_latencies
            )
        }