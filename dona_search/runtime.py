import logging
import platform
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock

CORE_CHECKPOINT = Path(__file__).resolve().parents[1] / 'checkpoints' / 'core'
logger = logging.getLogger(__name__)


def CoreModel(*args, **kwargs):
    from .core import CoreModel as Model
    return Model(*args, **kwargs)


class ModelRuntime:
    """Own both resident models and serialize GPU calls on one worker."""

    def __init__(self, build_query=None, *, build_ranker=None, build_organizer=None, query_only=False, organizer_only=False):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("apple_silicon_required")
        if organizer_only and (build_organizer is None or build_query is not None or build_ranker is not None or query_only):
            raise ValueError('invalid_organizer_only_runtime')
        if build_organizer is not None and (build_ranker is not None or query_only or (build_query is None and not organizer_only)):
            raise ValueError('invalid_organizer_runtime')
        if query_only and (build_query is None or build_ranker is not None):
            raise ValueError('invalid_query_only_runtime')
        self.query_processor = None
        self.core = None
        self.ranker = None
        self.organizer = None
        self._closed = False
        self._lock = Lock()
        self._cancel = Event()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dona-model")
        try:
            self._worker.submit(self._load, build_query, build_ranker, query_only, build_organizer).result()
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            try:
                self.close()
            except Exception:
                logger.debug('startup_cleanup_failed', exc_info=True)
            raise

    def _load(self, build_query, build_ranker=None, query_only=False, build_organizer=None):
        if build_ranker is not None or query_only or build_organizer is not None:
            if build_query is not None:
                self.query_processor = build_query()
            self.ranker = build_ranker() if build_ranker is not None else None
            self.organizer = build_organizer() if build_organizer is not None else None
            for model in (self.query_processor, self.organizer, self.ranker):
                setter = getattr(model, 'set_cancel_event', None)
                if callable(setter):
                    setter(self._cancel)
            if self.query_processor is not None and not callable(getattr(self.query_processor, 'plan', None)):
                raise ValueError('query_model_interface_invalid')
            if self.ranker is not None and not callable(getattr(self.ranker, 'rank', None)):
                raise ValueError('ranking_model_interface_invalid')
            if self.organizer is not None and not callable(getattr(self.organizer, 'organize', None)):
                raise ValueError('organizing_model_interface_invalid')
            try:
                if self.query_processor is not None:
                    self.query_processor.warmup()
                if self.ranker is not None:
                    self.ranker.warmup()
                if self.organizer is not None:
                    self.organizer.warmup()
            finally:
                self._cleanup()
            return
        self.core = CoreModel(CORE_CHECKPOINT)
        self.query_processor = build_query()
        try:
            self.query_processor.warmup()
            self.core.warmup()
        finally:
            self._cleanup()

    def _cleanup(self):
        self._each_model('cleanup')

    def _each_model(self, method):
        errors = []
        for model in (self.organizer, self.ranker, self.query_processor, self.core):
            if model is not None:
                try:
                    getattr(model, method)()
                except Exception as error:
                    logger.debug('model_cleanup_failed', exc_info=True)
                    errors.append(error)
        if errors:
            raise errors[0]

    def _invoke(self, fn, args, kwargs):
        try:
            return fn(*args, **kwargs)
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            try:
                self._cleanup()
            finally:
                if not self._closed:
                    self._cancel.clear()

    def call(self, fn, *args, **kwargs):
        with self._lock:
            if self._closed:
                raise RuntimeError("runtime_closed")
            future = self._worker.submit(self._invoke, fn, args, kwargs)
        try:
            return future.result()
        except KeyboardInterrupt:
            self._cancel.set()
            future.cancel()
            raise

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cancel.set()
            future = self._worker.submit(self._release)
        try:
            future.result()
        finally:
            self._worker.shutdown(wait=True)

    def _release(self):
        try:
            self._each_model('close')
        finally:
            self.ranker = None
            self.query_processor = None
            self.core = None
            self.organizer = None
