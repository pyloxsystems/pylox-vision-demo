"""GPU Inference Queue — manages shared GPU resources across V2 services.

Coordinates GPU usage between:
  - Anomalib (anomaly detection)
  - MediaPipe/pose estimation
  - SMPL body model inference
  - (Re-ID runs separately on port 3101)

Uses asyncio semaphore for concurrency control + priority queue.
"""

import asyncio
import time
import logging
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

logger = logging.getLogger("pylox-v2.gpu")


class Priority(IntEnum):
    CRITICAL = 0   # Real-time anomaly detection
    HIGH = 1       # Pose estimation for active tracks
    NORMAL = 2     # Anomalib batch inference
    LOW = 3        # Model training, non-urgent tasks


@dataclass(order=True)
class GPUTask:
    priority: int
    submit_time: float = field(compare=False)
    name: str = field(compare=False)
    func: Callable = field(compare=False)
    args: tuple = field(compare=False, default=())
    kwargs: dict = field(compare=False, default_factory=dict)
    future: asyncio.Future = field(compare=False, default=None)


class GPUQueue:
    def __init__(self, max_concurrent: int = 2):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue = asyncio.PriorityQueue()
        self._running = False
        self._workers = []
        self._stats = {
            "total_submitted": 0,
            "total_completed": 0,
            "total_errors": 0,
            "current_running": 0,
            "queue_depth": 0,
            "avg_wait_ms": 0,
            "avg_exec_ms": 0,
        }
        self._wait_times = []
        self._exec_times = []

    async def start(self, num_workers: int = 2):
        """Start the GPU queue workers."""
        self._running = True
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(f"gpu-worker-{i}"))
            self._workers.append(worker)
        logger.info(f"GPU queue started with {num_workers} workers, "
                    f"max concurrent: {self.max_concurrent}")

    async def stop(self):
        """Stop the GPU queue."""
        self._running = False
        for w in self._workers:
            w.cancel()
        self._workers.clear()
        logger.info("GPU queue stopped")

    async def submit(self, name: str, func: Callable, *args,
                     priority: Priority = Priority.NORMAL, **kwargs) -> Any:
        """Submit a task to the GPU queue and wait for result."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        task = GPUTask(
            priority=priority,
            submit_time=time.time(),
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            future=future,
        )

        await self.queue.put(task)
        self._stats["total_submitted"] += 1
        self._stats["queue_depth"] = self.queue.qsize()

        return await future

    def submit_fire_and_forget(self, name: str, func: Callable, *args,
                                priority: Priority = Priority.LOW, **kwargs):
        """Submit a task without waiting for result."""
        async def _submit():
            try:
                await self.submit(name, func, *args, priority=priority, **kwargs)
            except Exception as e:
                logger.error(f"Fire-and-forget task {name} failed: {e}")
        asyncio.create_task(_submit())

    async def _worker(self, worker_name: str):
        """Process tasks from the queue."""
        logger.info(f"{worker_name} started")
        while self._running:
            try:
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            wait_time = time.time() - task.submit_time
            self._wait_times.append(wait_time)
            if len(self._wait_times) > 100:
                self._wait_times = self._wait_times[-100:]

            async with self.semaphore:
                self._stats["current_running"] += 1
                exec_start = time.time()

                try:
                    if asyncio.iscoroutinefunction(task.func):
                        result = await task.func(*task.args, **task.kwargs)
                    else:
                        # Run sync function in thread pool
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None, lambda: task.func(*task.args, **task.kwargs)
                        )

                    if task.future and not task.future.done():
                        task.future.set_result(result)

                    self._stats["total_completed"] += 1

                except Exception as e:
                    logger.error(f"GPU task {task.name} failed: {e}")
                    self._stats["total_errors"] += 1
                    if task.future and not task.future.done():
                        task.future.set_exception(e)

                finally:
                    exec_time = time.time() - exec_start
                    self._exec_times.append(exec_time)
                    if len(self._exec_times) > 100:
                        self._exec_times = self._exec_times[-100:]

                    self._stats["current_running"] -= 1
                    self._stats["queue_depth"] = self.queue.qsize()
                    self._stats["avg_wait_ms"] = round(
                        sum(self._wait_times) / len(self._wait_times) * 1000, 1
                    ) if self._wait_times else 0
                    self._stats["avg_exec_ms"] = round(
                        sum(self._exec_times) / len(self._exec_times) * 1000, 1
                    ) if self._exec_times else 0

    def get_stats(self) -> dict:
        return {**self._stats}
