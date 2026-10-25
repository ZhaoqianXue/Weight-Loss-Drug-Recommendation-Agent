"""Shared API pacing and per-worker HTTPS reuse; no network at import time."""
from http.client import HTTPSConnection, HTTPException
from io import BytesIO
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

from .reddit import BASE_URL, BudgetReached, RateLimited


class RequestGate:
    """One process-wide budget/pacing/cooldown for every attempt and retry."""
    def __init__(self, budget, interval, persist_cooldown):
        self.budget, self.interval = budget, interval
        self.calls, self.next_start, self.not_before = 0, 0.0, 0.0
        self.stopped = False
        self.condition = threading.Condition()
        self.persist_cooldown = persist_cooldown

    def acquire(self):
        with self.condition:
            while True:
                if self.not_before > time.time():
                    raise RateLimited(self.not_before)
                if self.stopped or self.calls >= self.budget:
                    raise BudgetReached()
                delay = self.next_start - time.monotonic()
                if delay <= 0:
                    self.next_start = time.monotonic() + self.interval
                    self.calls += 1
                    return
                self.condition.wait(timeout=min(delay, 1))

    def cooldown(self, until):
        with self.condition:
            self.not_before = max(self.not_before, until)
            # Durable before another thread can reserve a new request.
            self.persist_cooldown(self.not_before)
            self.condition.notify_all()

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()


class KeepAliveTransport:
    """One verified HTTPS connection owned exclusively by one community worker.

    Fully consume each response before reuse. A dropped connection is retried
    by ArcticClient, so the shared gate counts and paces reconnect attempts too.
    """
    def __init__(self):
        self.connection = None

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __call__(self, request, timeout=40):
        url = urlsplit(request.full_url)
        origin = urlsplit(BASE_URL)
        if url.scheme != 'https' or url.netloc != origin.netloc:
            raise ValueError('Unexpected API origin')
        if self.connection is None:
            self.connection = HTTPSConnection(url.hostname, timeout=timeout)
        try:
            self.connection.request('GET', url.path + '?' + url.query,
                                    headers=dict(request.header_items()))
            response = self.connection.getresponse()
            body = response.read()
            status, headers, reason = response.status, response.headers, response.reason
            if response.will_close:
                self.close()
        except (OSError, HTTPException) as error:
            self.close()
            raise URLError(error) from error
        stream = BytesIO(body)
        if status >= 300:
            raise HTTPError(request.full_url, status, reason, headers, stream)
        stream.status = status
        return stream
