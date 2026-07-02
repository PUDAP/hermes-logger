"""
InfluxDB write client for the Hermes observer plugin.

Wraps influxdb-client with batched async writes, env-based config,
and a simple point-building API for each measurement type.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, WriteOptions
from influxdb_client.client.write_api import ASYNCHRONOUS
from influxdb_client.domain.write_precision import WritePrecision

load_dotenv()

logger = logging.getLogger("hermes_logger.influx_writer")

_MAX_TEXT_BYTES = int(os.getenv("HERMES_LOGGER_MAX_TEXT_BYTES", "8192"))


def _truncate(text: str | None) -> str:
    if not text:
        return ""
    encoded = text.encode("utf-8")
    if _MAX_TEXT_BYTES and len(encoded) > _MAX_TEXT_BYTES:
        return encoded[:_MAX_TEXT_BYTES].decode("utf-8", errors="replace") + "…"
    return text


def _json_field(value: Any) -> str:
    """Serialize a value to a compact JSON string, truncated if needed."""
    if value is None:
        return ""
    if isinstance(value, str):
        return _truncate(value)
    try:
        return _truncate(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return _truncate(str(value))


def _now_ns() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1e9)


class InfluxWriter:
    """
    Thin wrapper around influxdb-client that exposes one write method per
    Hermes measurement family.  All writes are batched; call close() on
    shutdown to flush remaining points.
    """

    MEASUREMENT_TURN = "hermes_llm_turn"
    MEASUREMENT_API = "hermes_api_request"
    MEASUREMENT_TOOL = "hermes_tool_call"
    MEASUREMENT_SESSION = "hermes_session"
    MEASUREMENT_SUBAGENT = "hermes_subagent"

    def __init__(self) -> None:
        url = os.environ["INFLUXDB_URL"]
        token = os.environ["INFLUXDB_TOKEN"]
        org = os.environ["INFLUXDB_ORG"]
        self.bucket = os.environ["INFLUXDB_BUCKET"]
        self.org = org

        batch_size = int(os.getenv("INFLUXDB_BATCH_SIZE", "500"))
        flush_interval = int(os.getenv("INFLUXDB_FLUSH_INTERVAL_MS", "10000"))

        self._client = InfluxDBClient(url=url, token=token, org=org)
        self._write_api = self._client.write_api(
            write_options=WriteOptions(
                batch_size=batch_size,
                flush_interval=flush_interval,
                write_type=ASYNCHRONOUS,
            )
        )
        logger.info("InfluxWriter connected to %s (bucket=%s)", url, self.bucket)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, record: dict) -> None:
        try:
            self._write_api.write(
                bucket=self.bucket,
                org=self.org,
                record=record,
                write_precision=WritePrecision.NANOSECONDS,
            )
        except Exception:
            logger.exception("Failed to write InfluxDB point: %s", record)

    @staticmethod
    def _point(measurement: str, tags: dict, fields: dict, ts_ns: int) -> dict:
        """Build an influxdb-client record dict (line protocol dict form)."""
        return {
            "measurement": measurement,
            "tags": {k: str(v) for k, v in tags.items() if v is not None},
            "fields": {k: v for k, v in fields.items() if v is not None},
            "time": ts_ns,
        }

    # ------------------------------------------------------------------
    # Session events
    # ------------------------------------------------------------------

    def write_session_start(self, **kw) -> None:
        tags = {
            "session_id": kw.get("session_id", ""),
        }
        fields = {
            "event": "start",
            "platform": kw.get("platform", ""),
            "model": kw.get("model", ""),
        }
        self._write(self._point(self.MEASUREMENT_SESSION, tags, fields, _now_ns()))

    def write_session_end(self, **kw) -> None:
        tags = {
            "session_id": kw.get("session_id", ""),
            "completed": str(kw.get("completed", True)),
            "interrupted": str(kw.get("interrupted", False)),
        }
        fields = {
            "event": "end",
            "reason": kw.get("reason", ""),
        }
        self._write(self._point(self.MEASUREMENT_SESSION, tags, fields, _now_ns()))

    # ------------------------------------------------------------------
    # Turn-level LLM events  (user prompt + final agent response)
    # ------------------------------------------------------------------

    def write_llm_turn_start(self, **kw) -> None:
        tags = {
            "session_id": kw.get("session_id", ""),
            "turn_id": kw.get("turn_id", ""),
            "model": kw.get("model", ""),
            "platform": kw.get("platform", ""),
            "is_first_turn": str(kw.get("is_first_turn", False)),
        }
        fields = {
            "event": "turn_start",
            "user_message": _truncate(kw.get("user_message", "")),
            "history_length": len(kw.get("conversation_history") or []),
        }
        self._write(self._point(self.MEASUREMENT_TURN, tags, fields, _now_ns()))

    def write_llm_turn_end(self, **kw) -> None:
        tags = {
            "session_id": kw.get("session_id", ""),
            "turn_id": kw.get("turn_id", ""),
            "model": kw.get("model", ""),
            "platform": kw.get("platform", ""),
        }
        fields = {
            "event": "turn_end",
            "user_message": _truncate(kw.get("user_message", "")),
            "assistant_response": _truncate(kw.get("assistant_response", "")),
            "history_length": len(kw.get("conversation_history") or []),
        }
        self._write(self._point(self.MEASUREMENT_TURN, tags, fields, _now_ns()))

    # ------------------------------------------------------------------
    # Request-scoped API events  (individual provider calls + reasoning)
    # ------------------------------------------------------------------

    def write_api_request_start(self, **kw) -> None:
        tags = {
            "session_id": kw.get("session_id", ""),
            "turn_id": kw.get("turn_id", ""),
            "api_request_id": kw.get("api_request_id", ""),
            "model": kw.get("model", ""),
            "provider": kw.get("provider", ""),
            "api_mode": kw.get("api_mode", ""),
        }
        fields = {
            "event": "api_start",
            "api_call_count": int(kw.get("api_call_count") or 0),
            "message_count": int(kw.get("message_count") or 0),
            "tool_count": int(kw.get("tool_count") or 0),
            "approx_input_tokens": int(kw.get("approx_input_tokens") or 0),
            "request_char_count": int(kw.get("request_char_count") or 0),
            "max_tokens": int(kw.get("max_tokens") or 0),
            "request_payload": _json_field(kw.get("request")),
        }
        ts = kw.get("started_at")
        ts_ns = int(ts * 1e9) if ts else _now_ns()
        self._write(self._point(self.MEASUREMENT_API, tags, fields, ts_ns))

    def write_api_request_end(self, **kw) -> None:
        tags = {
            "session_id": kw.get("session_id", ""),
            "turn_id": kw.get("turn_id", ""),
            "api_request_id": kw.get("api_request_id", ""),
            "model": kw.get("model", ""),
            "provider": kw.get("provider", ""),
            "finish_reason": kw.get("finish_reason", ""),
        }
        usage = kw.get("usage") or {}
        fields = {
            "event": "api_end",
            "api_duration_ms": float(kw.get("api_duration") or 0) * 1000,
            "prompt_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "assistant_content_chars": int(kw.get("assistant_content_chars") or 0),
            "assistant_tool_call_count": int(kw.get("assistant_tool_call_count") or 0),
            "response_payload": _json_field(kw.get("response")),
        }
        ts = kw.get("ended_at") or kw.get("started_at")
        ts_ns = int(ts * 1e9) if ts else _now_ns()
        self._write(self._point(self.MEASUREMENT_API, tags, fields, ts_ns))

    def write_api_request_error(self, **kw) -> None:
        tags = {
            "session_id": kw.get("session_id", ""),
            "turn_id": kw.get("turn_id", ""),
            "api_request_id": kw.get("api_request_id", ""),
            "model": kw.get("model", ""),
            "provider": kw.get("provider", ""),
            "status_code": str(kw.get("status_code", "")),
            "retryable": str(kw.get("retryable", False)),
        }
        error = kw.get("error") or {}
        fields = {
            "event": "api_error",
            "api_duration_ms": float(kw.get("api_duration") or 0) * 1000,
            "retry_count": int(kw.get("retry_count") or 0),
            "max_retries": int(kw.get("max_retries") or 0),
            "reason": kw.get("reason", ""),
            "error_type": error.get("type", ""),
            "error_message": _truncate(error.get("message", "")),
        }
        ts = kw.get("ended_at") or kw.get("started_at")
        ts_ns = int(ts * 1e9) if ts else _now_ns()
        self._write(self._point(self.MEASUREMENT_API, tags, fields, ts_ns))

    # ------------------------------------------------------------------
    # Tool call events
    # ------------------------------------------------------------------

    def write_tool_start(self, **kw) -> None:
        tags = {
            "session_id": kw.get("session_id", ""),
            "turn_id": kw.get("turn_id", ""),
            "tool_call_id": kw.get("tool_call_id", ""),
            "tool_name": kw.get("tool_name", ""),
            "api_request_id": kw.get("api_request_id", ""),
        }
        fields = {
            "event": "tool_start",
            "args": _json_field(kw.get("args")),
        }
        self._write(self._point(self.MEASUREMENT_TOOL, tags, fields, _now_ns()))

    def write_tool_end(self, **kw) -> None:
        tags = {
            "session_id": kw.get("session_id", ""),
            "turn_id": kw.get("turn_id", ""),
            "tool_call_id": kw.get("tool_call_id", ""),
            "tool_name": kw.get("tool_name", ""),
            "status": kw.get("status", ""),
            "error_type": kw.get("error_type", ""),
        }
        fields = {
            "event": "tool_end",
            "duration_ms": float(kw.get("duration_ms") or 0),
            "result": _json_field(kw.get("result")),
            "error_message": _truncate(kw.get("error_message", "")),
        }
        self._write(self._point(self.MEASUREMENT_TOOL, tags, fields, _now_ns()))

    # ------------------------------------------------------------------
    # Subagent events
    # ------------------------------------------------------------------

    def write_subagent_start(self, **kw) -> None:
        tags = {
            "parent_session_id": kw.get("parent_session_id", ""),
            "child_session_id": kw.get("child_session_id", ""),
            "child_subagent_id": kw.get("child_subagent_id", ""),
            "child_role": kw.get("child_role", ""),
        }
        fields = {
            "event": "subagent_start",
            "parent_turn_id": kw.get("parent_turn_id", ""),
            "child_goal": _truncate(kw.get("child_goal", "")),
        }
        self._write(self._point(self.MEASUREMENT_SUBAGENT, tags, fields, _now_ns()))

    def write_subagent_stop(self, **kw) -> None:
        tags = {
            "parent_session_id": kw.get("parent_session_id", ""),
            "child_session_id": kw.get("child_session_id", ""),
            "child_subagent_id": kw.get("child_subagent_id", ""),
            "child_role": kw.get("child_role", ""),
            "status": kw.get("status", ""),
        }
        fields = {
            "event": "subagent_stop",
            "duration_ms": float(kw.get("duration_ms") or 0),
            "child_summary": _truncate(kw.get("child_summary", "")),
        }
        self._write(self._point(self.MEASUREMENT_SUBAGENT, tags, fields, _now_ns()))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush remaining buffered points and close the client."""
        try:
            self._write_api.close()
            self._client.close()
            logger.info("InfluxWriter closed.")
        except Exception:
            logger.exception("Error closing InfluxWriter")
