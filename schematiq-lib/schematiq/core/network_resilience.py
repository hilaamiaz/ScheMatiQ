"""Riding out network outages during long unattended runs.

The LLM backends already retry a failed request a few times, but those retries
span only a couple of minutes. A longer outage (Wi-Fi drop, VPN reconnect,
provider incident) would otherwise make every document processed during it
fail and be skipped. Callers that own a unit of work (a document, a discovery
batch) use these helpers to tell such failures apart from real ones and to
wait for the network to come back before retrying that unit.
"""

from __future__ import annotations

import os
import socket
import time
from typing import Callable, Optional

import httpx

from schematiq.core.llm_backends import (
    _is_invalid_api_key_error,
    _is_rate_limit_error,
    _is_retryable_server_error,
)

# How long to keep waiting for connectivity before giving up on a unit of work.
NETWORK_WAIT_MAX_SECONDS = float(os.getenv("SCHEMATIQ_NETWORK_WAIT_SECONDS", "3600"))
# How many times a single unit of work is retried after a transient failure.
TRANSIENT_UNIT_RETRIES = int(os.getenv("SCHEMATIQ_TRANSIENT_UNIT_RETRIES", "3"))
# Pause before retrying a unit when the network itself is reachable (provider
# overload, rate limit): gives the provider time to recover.
TRANSIENT_RETRY_PAUSE_SECONDS = float(os.getenv("SCHEMATIQ_TRANSIENT_RETRY_PAUSE_SECONDS", "60"))

_PROBE_HOSTS = (
    ("generativelanguage.googleapis.com", 443),
    ("api.openai.com", 443),
)
_PROBE_INTERVAL_SECONDS = 15.0

_NETWORK_MESSAGE_HINTS = (
    "connection reset", "connection aborted", "connection refused",
    "connection error", "network is unreachable", "temporary failure in name resolution",
    "name or service not known", "getaddrinfo failed", "timed out", "read timeout",
    "remote end closed", "server disconnected", "eof occurred", "ssl",
    "failed to establish a new connection", "max retries exceeded",
)


def _iter_causes(error: BaseException):
    seen = set()
    current: Optional[BaseException] = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_transient_error(error: BaseException) -> bool:
    """True when error (or anything in its cause chain) is a network failure,
    timeout, rate limit or provider overload -- i.e. retrying later can work."""
    for err in _iter_causes(error):
        if _is_invalid_api_key_error(err):
            return False
        if isinstance(err, (httpx.TransportError, ConnectionError, socket.timeout, TimeoutError)):
            return True
        if _is_rate_limit_error(err) or _is_retryable_server_error(err):
            return True
        message = str(err).lower()
        if any(hint in message for hint in _NETWORK_MESSAGE_HINTS):
            return True
    return False


def network_available(timeout: float = 5.0) -> bool:
    """Cheap reachability probe: can we open a TCP connection to an LLM API?"""
    for host, port in _PROBE_HOSTS:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def wait_for_network(
    should_stop: Optional[Callable[[], bool]] = None,
    max_wait_seconds: float = NETWORK_WAIT_MAX_SECONDS,
    log: Callable[[str], None] = print,
) -> bool:
    """Block until the network is reachable again.

    Returns True once it is, False when max_wait_seconds elapse or should_stop()
    turns true first.
    """
    if network_available():
        return True
    log("🌐 Network unreachable - waiting for the connection to come back...")
    deadline = time.monotonic() + max_wait_seconds
    while time.monotonic() < deadline:
        if should_stop and should_stop():
            return False
        time.sleep(_PROBE_INTERVAL_SECONDS)
        if network_available():
            log("🌐 Network is back - resuming")
            return True
    log(f"🌐 Network still unreachable after {max_wait_seconds:.0f}s - giving up on this item")
    return False


def recover_from_transient_error(
    error: BaseException,
    attempt: int,
    should_stop: Optional[Callable[[], bool]] = None,
    log: Callable[[str], None] = print,
) -> bool:
    """Decide whether a failed unit of work should be retried, waiting as needed.

    attempt is the number of retries already made for this unit. Returns True
    when the caller should retry now, False when it should give up (permanent
    error, retries exhausted, network never came back, or stop requested).
    """
    if attempt >= TRANSIENT_UNIT_RETRIES or not is_transient_error(error):
        return False
    if should_stop and should_stop():
        return False
    log(
        f"⏳ Transient failure ({type(error).__name__}: {str(error)[:200]}); "
        f"retry {attempt + 1}/{TRANSIENT_UNIT_RETRIES} after the connection is confirmed"
    )
    if not wait_for_network(should_stop=should_stop, log=log):
        return False
    deadline = time.monotonic() + TRANSIENT_RETRY_PAUSE_SECONDS
    while time.monotonic() < deadline:
        if should_stop and should_stop():
            return False
        time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))
    return True
