import time
import os
import threading

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool
from concurrent.futures import CancelledError

# Batch runner ------------------------------------------------------------

class BatchSequentialProcessor:

    def __init__(self, agent_wrapper):
        self.agent_wrapper = agent_wrapper
        self.max_workers = 1

    def close(self):
        pass

    def __call__(self, argument_batch):
        return [self.agent_wrapper(arguments)
                for arguments in argument_batch]


class BatchParallelProcessor:

    def __init__(self, agent_wrapper, max_workers : int = 20, max_restarts: int = 100, 
                    poll_interval : float = 1.0, task_timeout : int = 300):
        self.agent_wrapper = agent_wrapper
        self.max_workers = max_workers
        self.max_restarts = max_restarts
        self.poll_interval = poll_interval
        self.task_timeout = task_timeout

        self._restarts = 0
        self._process_pool = None
        self._restart_exceptions = []
        self._start()

    def _start(self):
        if self._process_pool is not None:
            try:
                self._process_pool.shutdown(wait = False, cancel_futures = True)
            except Exception:
                pass
        self._process_pool = ProcessPoolExecutor(
            max_workers = self.max_workers,
            max_tasks_per_child = 10,
            mp_context=mp.get_context("spawn"),
        )

    def _kill_pool(self):
        pool = self._process_pool
        if pool is None:
            return

        # Python 3.14+
        if hasattr(pool, "kill_workers"):
            pool.kill_workers()
        elif hasattr(pool, "terminate_workers"):
            pool.terminate_workers()
        else:
            # Older Python: private API fallback
            for p in getattr(pool, "_processes", {}).values():
                try:
                    p.kill()      # or p.terminate()
                except Exception:
                    pass
            pool.shutdown(wait=False, cancel_futures=True)

        self._process_pool = None

    
    def close(self):
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=True, cancel_futures=True)
            self._process_pool = None
        
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _start_stucking_watchdog(self, futures, start_times, abort_event, timeout_tids):
        
        def _watch():
            while not abort_event.is_set():
                abort_event.wait(timeout = self.poll_interval)
                if abort_event.is_set(): break
                now = time.monotonic()
                stuck = [
                    (fut, tid) for fut, tid in futures.items()
                    if not fut.done() and now - start_times[tid] > self.task_timeout
                ]
                if stuck:
                    print(f">> Detected stuck tasks: {sorted(t for _, t in stuck)}")
                    for fut, tid in stuck:
                        timeout_tids.append(tid)

                    abort_event.set()

        t = threading.Thread(target = _watch, daemon = True)
        t.start()
        return t

    def run(self, argument_batch):
        print(f">> Parallel call of {len(argument_batch)} agents")

        self._restarts = 0
        tasks   = {i: arguments for i, arguments in enumerate(argument_batch)}
        pending = set(tasks.keys()) 
        results = [None] * len(tasks)

        while pending:
            if self._restarts >= self.max_restarts:
                raise BrokenProcessPool(f"Exceeded restart limit of {self.max_restarts}. Exceptions:\n{'\n'.join(f"- {str(e)}" for e in self._restart_exceptions)}")
            
            futures = {}
            start_times = {}
            for tid in pending:
                fut = self._process_pool.submit(
                    self.agent_wrapper, tasks[tid]
                )
                futures[fut] = tid
                start_times[tid] = time.monotonic()
            
            abort_event = threading.Event()
            timeout_tids = []
            watchdog = self._start_stucking_watchdog(
                futures, start_times, abort_event, timeout_tids
            )

            try:
                unfinished = set(futures.keys())

                while unfinished and not abort_event.is_set():
                    done, unfinished = wait(
                        unfinished,
                        timeout=self.poll_interval,
                        return_when=FIRST_COMPLETED,
                    )

                    for fut in done:
                        tid = futures[fut]
                        results[tid] = fut.result()
                        pending.discard(tid)

                    if abort_event.is_set():
                        raise TimeoutError(f"Tasks exceeded {self.task_timeout}s: {sorted(timeout_tids)}")

            except (BrokenProcessPool, TimeoutError, CancelledError) as e:
                self._restarts += 1
                self._restart_exceptions.append(str(e))
                
                self._kill_pool()
                self._start()
            finally:
                abort_event.set()
                watchdog.join(timeout = self.poll_interval * 2)
        
        return results
    
    def __call__(self, argument_batch):
        return self.run(argument_batch)
    