import threading
import time
from typing import Callable, Generic, TypeVar, Optional

from utils.logger import get_logger

logger = get_logger("lazy_loader")

T = TypeVar("T")

class LazyLoader(Generic[T]):
    """
    Thread-safe generic wrapper for lazy loading expensive objects in the background.
    Useful for starting slow initializations (like loading ML models) concurrently
    without blocking the main thread, while providing blocking access when the
    object is actually needed.
    """

    def __init__(self, name: str, factory: Callable[[], T]):
        """
        :param name: Identifier for logging purposes.
        :param factory: A zero-argument function that returns the heavy object.
        """
        self.name = name
        self._factory = factory
        self._instance: Optional[T] = None
        self._error: Optional[Exception] = None
        self._ready_event = threading.Event()
        self._start_time = time.time()
        
        # Start background load immediately
        self._thread = threading.Thread(
            target=self._load, 
            name=f"LazyLoad-{self.name}", 
            daemon=True
        )
        self._thread.start()

    def _load(self) -> None:
        try:
            logger.debug("lazy_load_started", component=self.name)
            self._instance = self._factory()
            elapsed = time.time() - self._start_time
            logger.info("lazy_load_completed", component=self.name, duration_sec=round(elapsed, 2))
        except Exception as e:
            self._error = e
            logger.error("lazy_load_failed", component=self.name, error=str(e), exc_info=True)
        finally:
            self._ready_event.set()

    def get(self, timeout: Optional[float] = None) -> T:
        """
        Get the loaded instance, blocking if it hasn't finished loading yet.
        Raises any exception that occurred during loading.
        """
        if not self._ready_event.is_set():
            logger.debug("lazy_load_waiting", component=self.name)
            
        success = self._ready_event.wait(timeout=timeout)
        if not success:
            raise TimeoutError(f"Timed out waiting for lazy component: {self.name}")
            
        if self._error:
            raise RuntimeError(f"Failed to load lazy component {self.name}: {self._error}") from self._error
            
        if self._instance is None:
            raise ValueError(f"Lazy component {self.name} loaded None value")
            
        return self._instance

    @property
    def is_ready(self) -> bool:
        """Check if the object is loaded without blocking."""
        return self._ready_event.is_set() and self._error is None
