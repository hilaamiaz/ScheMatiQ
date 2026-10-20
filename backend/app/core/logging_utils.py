"""Session-aware logging utilities.

Uses Python's contextvars to automatically inject session IDs into all
logger.* calls via a custom logging.Filter. Set the session context once
at the entry point (route handler or service method).

Propagation is not uniform, which is worth knowing before adding a call:
``asyncio.create_task`` and ``asyncio.to_thread`` copy the current context,
so the session id survives. Plain ``loop.run_in_executor`` does NOT: the
function runs in a bare thread with a fresh context and every record from it
logs as ``no-session``. Use ``ContextPropagatingThreadPoolExecutor`` below
for executors, or ``asyncio.to_thread`` instead of the default executor.
"""

import contextvars
import functools
import logging
from concurrent.futures import ThreadPoolExecutor

# Context variable holding the current session ID.
# Empty string means no session context (e.g., startup, system-level logs).
current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    'current_session_id', default=''
)


class SessionFilter(logging.Filter):
    """Injects session_id into every log record from the contextvar."""

    def filter(self, record: logging.LogRecord) -> bool:
        sid = current_session_id.get('')
        record.session_id = sid[:8] if sid else 'no-session'
        return True


class ContextPropagatingThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that carries the caller's context into the worker.

    ``loop.run_in_executor(pool, fn)`` calls ``pool.submit(fn)``, and a plain
    pool runs ``fn`` in a thread with a fresh context, so ``current_session_id``
    is lost and the whole offloaded operation logs as ``no-session``. Copying
    the context at submit time fixes every call site through this pool without
    touching any of them.
    """

    def submit(self, fn, /, *args, **kwargs):
        ctx = contextvars.copy_context()
        return super().submit(ctx.run, functools.partial(fn, *args, **kwargs))


def set_session_context(session_id: str) -> contextvars.Token:
    """Set the session_id for the current async/thread context.

    Returns a token that can be used to reset the context via
    current_session_id.reset(token).
    """
    return current_session_id.set(session_id)


class _TeeStream:
    """Write-through copy of a console stream into a log file.

    Catches the library's ``print()`` output, which never reaches a logging
    handler. The file is line-buffered so nothing is lost if the process dies.
    """

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, text):
        try:
            self._log_file.write(text)
        except Exception:
            pass
        try:
            return self._stream.write(text)
        except UnicodeEncodeError:
            # A console that can't show e.g. emoji must not crash the caller.
            enc = getattr(self._stream, "encoding", None) or "ascii"
            return self._stream.write(text.encode(enc, "replace").decode(enc))

    def flush(self):
        try:
            self._log_file.flush()
        except Exception:
            pass
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def setup_file_logging(log_dir, formatter: logging.Formatter) -> None:
    """Mirror all backend output into files under ``log_dir``.

    - ``backend.log``: every log record (root and uvicorn loggers, so request
      500 tracebacks are included) plus everything printed to stdout/stderr.
    - ``crash.log``: Python stacks of every thread when the process dies from
      a native crash (e.g. inside Docling), which produces no Python traceback.
    """
    import faulthandler
    import sys
    from datetime import datetime
    from logging.handlers import RotatingFileHandler
    from pathlib import Path

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        log_dir / "backend.log", maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    handler.setFormatter(formatter)
    handler.addFilter(SessionFilter())
    logging.getLogger().addHandler(handler)
    # uvicorn's logger doesn't propagate to root (uvicorn.error propagates
    # into it), so attach explicitly to catch request 500 tracebacks.
    logging.getLogger("uvicorn").addHandler(handler)

    prints_file = open(log_dir / "stdout.log", "a", encoding="utf-8", buffering=1)
    prints_file.write(f"\n===== backend started {datetime.now().isoformat()} =====\n")
    sys.stdout = _TeeStream(sys.stdout, prints_file)
    sys.stderr = _TeeStream(sys.stderr, prints_file)

    crash_file = open(log_dir / "crash.log", "a", encoding="utf-8")
    crash_file.write(f"\n===== backend started {datetime.now().isoformat()} =====\n")
    crash_file.flush()
    faulthandler.enable(file=crash_file, all_threads=True)
    # Keep the handle alive for the life of the process.
    setup_file_logging._crash_file = crash_file
