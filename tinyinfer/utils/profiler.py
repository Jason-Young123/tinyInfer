import inspect
import threading
import time
from collections import deque
from pathlib import Path

import torch


class TimeProfiler:
    PROFILER = True
    profilers = []
    _registry_lock = threading.Lock()


    def __init__(self):
        self.cycle = 1                      # 被调用多少次才记录
        self.cnt = 0                        # 当前被调用了多少次
        self.enabled = True
        self.record_list = deque()          # [(cpu_ms, gpu_start, gpu_end)]
        self.data_list = deque(maxlen=300)  # [(cpu_ms, gpu_ms)]
        self.sample_count = 0               # Total completed samples, never capped
        self._active = None
        self.stream = None
        self._lock = threading.Lock()
        self.title = "Unregistered profiler"
        self.registered = False
        self.cuda_enabled = torch.cuda.is_available()


    def register(self, name="", description="", cycle=1, enabled=True):
        if not TimeProfiler.PROFILER:
            return self
        if not enabled:
            self.enabled = False
            return self

        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        try:
            filename = caller.f_code.co_filename if caller is not None else "<unknown>"
            lineno = caller.f_lineno if caller is not None else 0
        finally:
            del frame
            del caller

        self.title = f"{Path(filename).name}:{lineno} | {name or 'Time profiler'}"
        if description:
            self.title += f"\n{description}"
        self.cycle = cycle
        self.cnt = 0

        with TimeProfiler._registry_lock:
            if not self.registered:
                TimeProfiler.profilers.append(self)
                self.registered = True
        return self


    def record_start(self):
        if not TimeProfiler.PROFILER or not self.enabled:
            return
        if self._active is not None:
            raise RuntimeError("record_start() called before the previous record_end()")

        self.cnt = (self.cnt + 1) % self.cycle
        if self.cnt != 0: 
            return

        # 只有self.cnt = 0的时候才会执行后续逻辑
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

        cpu_start = time.perf_counter_ns()
        self._active = (cpu_start, gpu_start, gpu_end, stream)


    def record_end(self):
        if not TimeProfiler.PROFILER or not self.enabled:
            return
        if self.cnt != 0:
            return
        if self._active is None:
            raise RuntimeError("record_end() called without record_start()")

        cpu_end = time.perf_counter_ns()
        cpu_start, gpu_start, gpu_end, stream = self._active
        self._active = None

        cpu_ms = (cpu_end - cpu_start) / 1e6
        if gpu_end is not None:
            gpu_end.record(stream)

        with self._lock:
            self.record_list.append((cpu_ms, gpu_start, gpu_end))


    def drain(self):
        """Move completed records to data_list without synchronizing the GPU."""
        while True:
            with self._lock:
                if not self.record_list:
                    return
                record = self.record_list[0]

            cpu_ms, gpu_start, gpu_end = record
            if gpu_end is not None and not gpu_end.query():
                return

            gpu_ms = gpu_start.elapsed_time(gpu_end) if gpu_end is not None else None

            with self._lock:
                if self.record_list and self.record_list[0] is record:
                    self.record_list.popleft()
                    self.data_list.append((cpu_ms, gpu_ms))
                    self.sample_count += 1

    @classmethod
    def drain_all(cls):
        with cls._registry_lock:
            profilers = list(cls.profilers)
        for profiler in profilers:
            profiler.drain()

    @classmethod
    def snapshot_all(cls):
        """Return the latest (at most 300) samples for each registered profiler."""
        with cls._registry_lock:
            profilers = list(cls.profilers)

        snapshots = []
        for profiler in profilers:
            with profiler._lock:
                samples = list(profiler.data_list)
                total = profiler.sample_count
            snapshots.append({
                "id": str(id(profiler)),
                "title": profiler.title,
                "sample_start": total - len(samples),
                "total_samples": total,
                "samples": samples,
                "reset": True,
            })
        return snapshots
