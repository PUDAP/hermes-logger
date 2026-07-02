"""
hermes-logger: Hermes observer plugin for InfluxDB telemetry.

Registers read-only observer hooks per the hermes.observer.v1 contract and
writes structured points to InfluxDB for every user prompt, LLM API request
(including reasoning payloads), tool call, and agent response.

Registration
------------
Drop this directory into ~/.hermes/plugins/ and ensure a .env file is present
(see .env.example) before starting the agent.  Enable the plugin with:

    hermes plugins enable hermes-logger

Measurements written
--------------------
  hermes_session      – session start / end lifecycle
  hermes_llm_turn     – per-turn user prompt and final assistant response
  hermes_api_request  – individual provider API attempts with timing & tokens
  hermes_tool_call    – tool call start / end with args, result, and status
  hermes_subagent     – delegated child-agent start / stop events
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

_log_level = os.getenv("HERMES_LOGGER_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, _log_level, logging.INFO))
logger = logging.getLogger("hermes_logger")

# Lazy-initialised so import errors surface at registration time, not at
# module load time, preserving Hermes' fail-open plugin contract.
_writer = None


def _get_writer():
    global _writer
    if _writer is None:
        from .influx_writer import InfluxWriter
        _writer = InfluxWriter()
    return _writer


def _safe(fn, **kwargs):
    """Call fn with kwargs, swallowing any exception so hooks stay fail-open."""
    try:
        fn(**kwargs)
    except Exception:
        logger.exception("hermes-logger hook error in %s", fn.__name__)


# -------------------------------------------------------------------------
# Hook callbacks
# -------------------------------------------------------------------------

def on_session_start(**kwargs):
    _safe(_get_writer().write_session_start, **kwargs)


def on_session_end(**kwargs):
    _safe(_get_writer().write_session_end, **kwargs)


def on_pre_llm_call(**kwargs):
    """
    Fires before the tool loop begins for a user turn.
    Captures the raw user prompt and conversation context.
    """
    _safe(_get_writer().write_llm_turn_start, **kwargs)


def on_post_llm_call(**kwargs):
    """
    Fires after the turn completes with the final assistant output.
    Captures the agent's response text.
    """
    _safe(_get_writer().write_llm_turn_end, **kwargs)


def on_pre_api_request(**kwargs):
    """
    Fires immediately before each provider API call.
    Captures model, token estimates, and the sanitized request payload
    (which includes the messages sent to the LLM, i.e. its reasoning context).
    """
    _safe(_get_writer().write_api_request_start, **kwargs)


def on_post_api_request(**kwargs):
    """
    Fires after a successful provider response.
    Captures timing, token usage, finish reason, and the sanitized response
    payload (which includes assistant content and any reasoning/thinking blocks).
    """
    _safe(_get_writer().write_api_request_end, **kwargs)


def on_api_request_error(**kwargs):
    """
    Fires after a failed or retryable provider request.
    """
    _safe(_get_writer().write_api_request_error, **kwargs)


def on_pre_tool_call(**kwargs):
    """
    Fires before guardrail-approved tool dispatch.
    Captures tool name and arguments.
    """
    _safe(_get_writer().write_tool_start, **kwargs)


def on_post_tool_call(**kwargs):
    """
    Fires after tool dispatch regardless of outcome (ok / error / blocked /
    cancelled).  Captures result, duration, and status.
    """
    _safe(_get_writer().write_tool_end, **kwargs)


def on_subagent_start(**kwargs):
    _safe(_get_writer().write_subagent_start, **kwargs)


def on_subagent_stop(**kwargs):
    _safe(_get_writer().write_subagent_stop, **kwargs)


def on_session_finalize(**kwargs):
    """Flush and close the InfluxDB write client when the session tears down."""
    global _writer
    if _writer is not None:
        try:
            _writer.close()
        except Exception:
            logger.exception("hermes-logger: error closing InfluxWriter on finalize")
        finally:
            _writer = None


# -------------------------------------------------------------------------
# Plugin entry point
# -------------------------------------------------------------------------

def register(ctx):
    """
    Called by Hermes' plugin manager.  Registers all observer hooks.
    Returns early without raising if InfluxDB config is missing so the agent
    can still start; a WARNING is logged instead.
    """
    required = ["INFLUXDB_URL", "INFLUXDB_TOKEN", "INFLUXDB_ORG", "INFLUXDB_BUCKET"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        logger.warning(
            "hermes-logger: missing env vars %s — plugin will not log to InfluxDB.",
            missing,
        )
        return

    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_hook("on_session_finalize", on_session_finalize)

    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)

    ctx.register_hook("pre_api_request", on_pre_api_request)
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_hook("api_request_error", on_api_request_error)

    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)

    ctx.register_hook("subagent_start", on_subagent_start)
    ctx.register_hook("subagent_stop", on_subagent_stop)

    logger.info(
        "hermes-logger registered — writing to %s/%s",
        os.getenv("INFLUXDB_URL"),
        os.getenv("INFLUXDB_BUCKET"),
    )
