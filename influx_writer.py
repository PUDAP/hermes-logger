"""
InfluxDB write client for the Hermes observer plugin.

Wraps influxdb-client with batched async writes, env-based config,
and a simple point-building API for each measurement type.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, cast

from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, WriteOptions
from influxdb_client.client.write_api import ASYNCHRONOUS
from influxdb_client.domain.write_precision import WritePrecision

load_dotenv()

logger = logging.getLogger("hermes_logger.influx_writer")

_MAX_TEXT_BYTES = int(os.getenv("HERMES_LOGGER_MAX_TEXT_BYTES", "8192"))
_AGENT_ID = os.getenv("AGENT_ID", "unset")


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
    MEASUREMENT_APPROVAL = "hermes_approval"

    def __init__(self) -> None:
        url = os.environ["INFLUXDB_URL"]
        token = os.environ["INFLUXDB_TOKEN"]
        self.org = os.environ["INFLUXDB_ORG"]
        self.database = os.environ["INFLUXDB_DATABASE"]

        batch_size = int(os.getenv("INFLUXDB_BATCH_SIZE", "500"))
        flush_interval = int(os.getenv("INFLUXDB_FLUSH_INTERVAL_MS", "10000"))

        self._client = InfluxDBClient(url=url, token=token, org=self.org)
        self._write_api = self._client.write_api(
            write_options=WriteOptions(
                batch_size=batch_size,
                flush_interval=flush_interval,
                write_type=ASYNCHRONOUS,
            )
        )
        logger.info("InfluxWriter connected to %s (database=%s)", url, self.database)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, record: dict) -> None:
        try:
            self._write_api.write(
                bucket=self.database,
                org=self.org,
                record=record,
                write_precision=cast(WritePrecision, WritePrecision.NS),
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

    def _emit(self, measurement: str, tags: dict, fields: dict, ts_ns: int, kw: dict) -> None:
        """Build and write a point, tagging it with the observer schema version and agent id."""
        tags = dict(tags)
        schema_version = kw.get("telemetry_schema_version")
        if schema_version:
            tags["telemetry_schema_version"] = schema_version
        if _AGENT_ID:
            tags["agent_id"] = _AGENT_ID
        self._write(self._point(measurement, tags, fields, ts_ns))

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
        self._emit(self.MEASUREMENT_SESSION, tags, fields, _now_ns(), kw)

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
        self._emit(self.MEASUREMENT_SESSION, tags, fields, _now_ns(), kw)

    def write_session_reset(self, **kw) -> None:
        """CLI/gateway moved from an old session identity to a new one."""
        tags = {
            "old_session_id": kw.get("old_session_id", ""),
            "new_session_id": kw.get("new_session_id", ""),
        }
        fields = {
            "event": "reset",
            "reason": kw.get("reason", ""),
        }
        self._emit(self.MEASUREMENT_SESSION, tags, fields, _now_ns(), kw)

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
            "sender_id": kw.get("sender_id", ""),
        }
        fields = {
            "event": "turn_start",
            "user_message": _truncate(kw.get("user_message", "")),
            "history_length": len(kw.get("conversation_history") or []),
        }
        self._emit(self.MEASUREMENT_TURN, tags, fields, _now_ns(), kw)

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
        self._emit(self.MEASUREMENT_TURN, tags, fields, _now_ns(), kw)

    # ------------------------------------------------------------------
    # Request-scoped API events  (individual provider calls + reasoning)
    # ------------------------------------------------------------------

    def write_api_request_start(self, **kw) -> None:
        tags = {
            "session_id": kw.get("session_id", ""),
            "task_id": kw.get("task_id", ""),
            "turn_id": kw.get("turn_id", ""),
            "api_request_id": kw.get("api_request_id", ""),
            "model": kw.get("model", ""),
            "provider": kw.get("provider", ""),
            "platform": kw.get("platform", ""),
            "api_mode": kw.get("api_mode", ""),
            "base_url": kw.get("base_url", ""),
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
        self._emit(self.MEASUREMENT_API, tags, fields, ts_ns, kw)

    def write_api_request_end(self, **kw) -> None:
        tags = {
            "session_id": kw.get("session_id", ""),
            "task_id": kw.get("task_id", ""),
            "turn_id": kw.get("turn_id", ""),
            "api_request_id": kw.get("api_request_id", ""),
            "model": kw.get("model", ""),
            "provider": kw.get("provider", ""),
            "platform": kw.get("platform", ""),
            "api_mode": kw.get("api_mode", ""),
            "base_url": kw.get("base_url", ""),
            "finish_reason": kw.get("finish_reason", ""),
        }
        usage = kw.get("usage") or {}
        fields = {
            "event": "api_end",
            "api_duration_ms": float(kw.get("api_duration") or 0) * 1000,
            "message_count": int(kw.get("message_count") or 0),
            "response_model": kw.get("response_model", ""),
            "prompt_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "usage_payload": _json_field(usage),
            "assistant_content_chars": int(kw.get("assistant_content_chars") or 0),
            "assistant_tool_call_count": int(kw.get("assistant_tool_call_count") or 0),
            "response_payload": _json_field(kw.get("response")),
        }
        ts = kw.get("ended_at") or kw.get("started_at")
        ts_ns = int(ts * 1e9) if ts else _now_ns()
        self._emit(self.MEASUREMENT_API, tags, fields, ts_ns, kw)

    def write_api_request_error(self, **kw) -> None:
        tags = {
            "session_id": kw.get("session_id", ""),
            "task_id": kw.get("task_id", ""),
            "turn_id": kw.get("turn_id", ""),
            "api_request_id": kw.get("api_request_id", ""),
            "model": kw.get("model", ""),
            "provider": kw.get("provider", ""),
            "platform": kw.get("platform", ""),
            "api_mode": kw.get("api_mode", ""),
            "base_url": kw.get("base_url", ""),
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
            "request_payload": _json_field(kw.get("request")),
        }
        ts = kw.get("ended_at") or kw.get("started_at")
        ts_ns = int(ts * 1e9) if ts else _now_ns()
        self._emit(self.MEASUREMENT_API, tags, fields, ts_ns, kw)

    # ------------------------------------------------------------------
    # Tool call events
    # ------------------------------------------------------------------

    def write_tool_start(self, **kw) -> None:
        tags = {
            "session_id": kw.get("session_id", ""),
            "task_id": kw.get("task_id", ""),
            "turn_id": kw.get("turn_id", ""),
            "tool_call_id": kw.get("tool_call_id", ""),
            "tool_name": kw.get("tool_name", ""),
            "api_request_id": kw.get("api_request_id", ""),
        }
        fields = {
            "event": "tool_start",
            "args": _json_field(kw.get("args")),
        }
        self._emit(self.MEASUREMENT_TOOL, tags, fields, _now_ns(), kw)

    def write_tool_end(self, **kw) -> None:
        tags = {
            "session_id": kw.get("session_id", ""),
            "task_id": kw.get("task_id", ""),
            "turn_id": kw.get("turn_id", ""),
            "tool_call_id": kw.get("tool_call_id", ""),
            "tool_name": kw.get("tool_name", ""),
            "api_request_id": kw.get("api_request_id", ""),
            "status": kw.get("status", ""),
            "error_type": kw.get("error_type", ""),
        }
        fields = {
            "event": "tool_end",
            "duration_ms": float(kw.get("duration_ms") or 0),
            "result": _json_field(kw.get("result")),
            "error_message": _truncate(kw.get("error_message", "")),
        }
        self._emit(self.MEASUREMENT_TOOL, tags, fields, _now_ns(), kw)

    # ------------------------------------------------------------------
    # Subagent events
    # ------------------------------------------------------------------

    def write_subagent_start(self, **kw) -> None:
        tags = {
            "parent_session_id": kw.get("parent_session_id", ""),
            "parent_subagent_id": kw.get("parent_subagent_id", ""),
            "child_session_id": kw.get("child_session_id", ""),
            "child_subagent_id": kw.get("child_subagent_id", ""),
            "child_role": kw.get("child_role", ""),
        }
        fields = {
            "event": "subagent_start",
            "parent_turn_id": kw.get("parent_turn_id", ""),
            "child_goal": _truncate(kw.get("child_goal", "")),
        }
        self._emit(self.MEASUREMENT_SUBAGENT, tags, fields, _now_ns(), kw)

    def write_subagent_stop(self, **kw) -> None:
        tags = {
            "parent_session_id": kw.get("parent_session_id", ""),
            "parent_subagent_id": kw.get("parent_subagent_id", ""),
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
        self._emit(self.MEASUREMENT_SUBAGENT, tags, fields, _now_ns(), kw)

    # ------------------------------------------------------------------
    # Approval events
    # ------------------------------------------------------------------

    def write_approval_request(self, **kw) -> None:
        tags = {
            "session_id": kw.get("session_id", ""),
            "task_id": kw.get("task_id", ""),
            "turn_id": kw.get("turn_id", ""),
            "session_key": kw.get("session_key", ""),
            "surface": kw.get("surface", ""),
            "pattern_key": kw.get("pattern_key", ""),
        }
        fields = {
            "event": "approval_request",
            "command": _truncate(kw.get("command", "")),
            "description": _truncate(kw.get("description", "")),
            "pattern_keys": _json_field(kw.get("pattern_keys")),
        }
        self._emit(self.MEASUREMENT_APPROVAL, tags, fields, _now_ns(), kw)

    def write_approval_response(self, **kw) -> None:
        tags = {
            "session_id": kw.get("session_id", ""),
            "task_id": kw.get("task_id", ""),
            "turn_id": kw.get("turn_id", ""),
            "session_key": kw.get("session_key", ""),
            "surface": kw.get("surface", ""),
            "pattern_key": kw.get("pattern_key", ""),
            "choice": kw.get("choice", ""),
        }
        fields = {
            "event": "approval_response",
            "command": _truncate(kw.get("command", "")),
            "description": _truncate(kw.get("description", "")),
            "pattern_keys": _json_field(kw.get("pattern_keys")),
        }
        self._emit(self.MEASUREMENT_APPROVAL, tags, fields, _now_ns(), kw)

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
