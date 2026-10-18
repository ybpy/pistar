"""Bounded background prefetch with exception propagation and cancellation."""
import queue
import threading


class PrefetchIterator:
    def __init__(self, iterable, depth: int = 3):
        if depth <= 0:
            raise ValueError("depth must be positive")
        self._source = iter(iterable)
        self._queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._produce, name="value-batch-prefetch", daemon=True)
        self._thread.start()

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def _produce(self):
        try:
            for item in self._source:
                if not self._put((True, item)):
                    break
            self._put((False, StopIteration()))
        except Exception as exc:
            self._put((False, exc))
        finally:
            close = getattr(self._source, "close", None)
            if close is not None:
                close()

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        ok, item = self._queue.get()
        if not ok:
            self.close()
            raise item
        return item

    def close(self):
        self._closed = True
        self._stop.set()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=1)


def prefetch_iterator(iterable, depth: int = 3):
    return iter(iterable) if depth == 0 else PrefetchIterator(iterable, depth)
