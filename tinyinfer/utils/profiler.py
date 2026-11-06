import inspect
import threading
import time
from pathlib import Path
from collections import deque

import torch


class TimeProfiler:
    PROFILER = True                 # Global profiling switch
    # Shared registry: all TimeProfiler instances are stored here.
    profilers = []
    _registry_lock = threading.Lock()

    def __init__(self):
        # self.record_list = []  # [(cpu_ms, gpu_start, gpu_end)]
        self.record_list = deque()          # [(cpu_ms, gpu_start, gpu_end)]
        self.data_list = deque(maxlen=1000) # [(cpu_ms, gpu_ms)]
        self._active = None    # One active measurement per profiler.
        self.stream = None
        self._lock = threading.Lock()

        self.title = "Unregistered profiler"
        self.registered = False
        self.fig = None
        self.ax = None

        self.cuda_enabled = torch.cuda.is_available()


    # Register this profiler and capture the caller's file and line
    def register(self, name="", description=""):
        if not TimeProfiler.PROFILER:
            return self
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        try:
            filename = caller.f_code.co_filename if caller is not None else "<unknown>"
            lineno = caller.f_lineno if caller is not None else 0
        finally:
            del frame
            del caller

        name = name or "Time profiler"
        self.title = f"{Path(filename).name}:{lineno} | {name}"
        if description:
            self.title += f"\n{description}"

        with TimeProfiler._registry_lock:
            if not self.registered:
                TimeProfiler.profilers.append(self)
                self.registered = True

        return self


    # Record CPU start time and enqueue a CUDA start Event
    def record_start(self):
        if not TimeProfiler.PROFILER:
            return
        if self._active is not None:
            raise RuntimeError("Fatal: record_start() called before the previous record_end()")

        if self.cuda_enabled:
            stream = torch.cuda.current_stream()
            if self.stream is None:
                self.stream = stream
            elif (stream.device != self.stream.device or stream.cuda_stream != self.stream.cuda_stream):
                raise RuntimeError("Profiler CUDA stream changed")
            gpu_start = torch.cuda.Event(enable_timing=True)
            gpu_end = torch.cuda.Event(enable_timing=True)
            gpu_start.record(stream)
        else:
            stream = None
            gpu_start = gpu_end = None

        # Start CPU timing after enqueueing the start Event.
        cpu_start = time.perf_counter_ns()
        self._active = (cpu_start, gpu_start, gpu_end, stream) # 对_active进行占位



    # Record CPU end time and enqueue the CUDA end Event without synchronizing
    def record_end(self):
        if not TimeProfiler.PROFILER:
            return
        if self._active is None:
            raise RuntimeError("Fatal: record_end() called without record_start()")

        cpu_end = time.perf_counter_ns()
        cpu_start, gpu_start, gpu_end, stream = self._active
        self._active = None # 清除占位

        cpu_ms = (cpu_end - cpu_start) / 1e6
        if gpu_end is not None:
            gpu_end.record(stream)

        with self._lock:
            self.record_list.append((cpu_ms, gpu_start, gpu_end))




    # Move completed records to data_list using non-blocking Event.query()
    def drain(self):
        with self._lock:
            while self.record_list:
                cpu_ms, gpu_start, gpu_end = self.record_list[0]

                # GPU 尚未执行到当前 End Event, 后面的 Event 也不可能完成，直接退出
                if gpu_end is not None and not gpu_end.query():
                    break

                # 当前记录已经完成，从队首移除
                self.record_list.popleft()

                # 计算 GPU 耗时
                gpu_ms = (gpu_start.elapsed_time(gpu_end) if gpu_end is not None else None)

                self.data_list.append((cpu_ms, gpu_ms))




    # Update this profiler's plot. Must be called from the GUI thread
    def draw(self):
        import matplotlib.pyplot as plt

        with self._lock:
            data = list(self.data_list)

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(9, 3.5))
            manager = getattr(self.fig.canvas, "manager", None)
            if manager is not None and hasattr(manager, "set_window_title"):
                manager.set_window_title(self.title)

        self.ax.clear()

        if not data:
            self.ax.text(
                0.5, 0.5, "Waiting for samples...",
                ha="center", va="center",
                transform=self.ax.transAxes,
            )
        else:
            x = list(range(1, len(data) + 1))
            cpu = [item[0] for item in data]
            gpu = [item[1] for item in data]

            self.ax.plot(x, cpu, color="tab:blue", linewidth=1,
                         label="CPU wall time")
            self.ax.plot(x, gpu, color="tab:red", linewidth=1,
                         label="CUDA Event time")

            cpu_avg = sum(cpu) / len(cpu)
            self.ax.axhline(cpu_avg, color="lightblue",
                            linestyle="--", linewidth=1)
            self.ax.text(
                1.01, cpu_avg, f"CPU avg: {cpu_avg:.4f} ms",
                transform=self.ax.get_yaxis_transform(),
                color="tab:blue", va="center", fontsize=8,
                clip_on=False,
            )

            gpu_values = [value for value in gpu if value is not None]
            if gpu_values:
                gpu_avg = sum(gpu_values) / len(gpu_values)
                self.ax.axhline(gpu_avg, color="lightcoral",
                                linestyle="--", linewidth=1)
                self.ax.text(
                    1.01, gpu_avg, f"CUDA avg: {gpu_avg:.4f} ms",
                    transform=self.ax.get_yaxis_transform(),
                    color="tab:red", va="center", fontsize=8,
                    clip_on=False,
                )

            self.ax.set_xlabel("Sample")
            self.ax.set_ylabel("Time (ms)")
            self.ax.legend(loc="upper left")
            self.ax.grid(True, alpha=0.25)

        self.ax.set_title(self.title, loc="left", fontsize=9)
        self.fig.subplots_adjust(right=0.78)
        self.fig.canvas.draw_idle()
