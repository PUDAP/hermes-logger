"""Regression tests using the real InfluxDB SDK and an isolated HTTP sink."""

import importlib.util
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


@pytest.fixture
def writer_module(monkeypatch):
    # Never load deployment credentials from a developer's .env during tests.
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    spec = importlib.util.spec_from_file_location(
        "logger_writer_under_test", Path(__file__).parents[1] / "influx_writer.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sink():
    requests = []
    received = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append((self.path, body))
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            received.set()

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests, received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def make_writer(module, monkeypatch, url, *, batch_size=2, flush_ms=2000):
    for key, value in {
        "INFLUXDB_URL": url,
        "INFLUXDB_TOKEN": "test-only-token",
        "INFLUXDB_ORG": "test-org",
        "INFLUXDB_DATABASE": "test-database",
        "INFLUXDB_BATCH_SIZE": str(batch_size),
        "INFLUXDB_FLUSH_INTERVAL_MS": str(flush_ms),
    }.items():
        monkeypatch.setenv(key, value)
    return module.InfluxWriter()


def point(module, number):
    return module.InfluxWriter._point(
        "hermes_tool_call", {"session_id": "test-session"},
        {"event": "tool_end", "duration_ms": number}, number,
    )


def test_buffers_until_batch_size(writer_module, monkeypatch, sink):
    url, requests, received = sink
    writer = make_writer(writer_module, monkeypatch, url)
    try:
        writer._write(point(writer_module, 1))
        assert not received.wait(0.1), "first point must be buffered, not sent synchronously"
        writer._write(point(writer_module, 2))
        assert received.wait(5), "full batch was not delivered"
        assert len(requests) == 1
        assert len(requests[0][1].splitlines()) == 2
        assert "/api/v2/write?" in requests[0][0]
        assert "bucket=test-database" in requests[0][0]
    finally:
        writer.close()


def test_flush_interval_sends_partial_batch(writer_module, monkeypatch, sink):
    url, requests, received = sink
    writer = make_writer(writer_module, monkeypatch, url, flush_ms=50)
    try:
        writer._write(point(writer_module, 1))
        assert received.wait(5), "partial batch was not flushed on schedule"
        assert len(requests) == 1
        assert len(requests[0][1].splitlines()) == 1
    finally:
        writer.close()


def test_close_flushes_partial_batch(writer_module, monkeypatch, sink):
    url, requests, received = sink
    writer = make_writer(writer_module, monkeypatch, url)
    try:
        writer._write(point(writer_module, 1))
    finally:
        writer.close()
    assert received.wait(5), "shutdown discarded buffered point"
    assert len(requests) == 1
    assert len(requests[0][1].splitlines()) == 1


def test_write_failure_does_not_log_payload_or_exception(
    writer_module, monkeypatch, sink, caplog
):
    url, _, _ = sink
    writer = make_writer(writer_module, monkeypatch, url)
    secret = "FAKE-SENSITIVE-PAYLOAD"

    def fail_write(**kwargs):
        raise RuntimeError(f"server echoed {secret}")

    monkeypatch.setattr(writer._write_api, "write", fail_write)
    record = writer_module.InfluxWriter._point(
        "hermes_tool_call", {"session_id": "test-session"},
        {"args": secret}, 1,
    )
    try:
        writer._write(record)
        assert secret not in caplog.text
        assert "Failed to write InfluxDB point" in caplog.text
        assert "RuntimeError" in caplog.text
        assert all(r.exc_info is None for r in caplog.records)
    finally:
        writer.close()
