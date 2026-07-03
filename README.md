# hermes-logger

A [Hermes observer plugin](https://github.com/NousResearch/hermes-agent/tree/main/docs/observability) that writes structured telemetry to **InfluxDB** for every user prompt, LLM API call (including reasoning payloads), tool call, and final agent response.

## Measurements

| Measurement | What it captures |
|---|---|
| `hermes_session` | Session start / end / reset lifecycle |
| `hermes_llm_turn` | Per-turn **user prompt** (`turn_start`) and **final agent response** (`turn_end`) |
| `hermes_api_request` | Individual provider API attempts — timing, token usage, sanitized **request & response payloads** (LLM reasoning context) |
| `hermes_tool_call` | Tool call start / end — name, **args**, **result**, duration, status |
| `hermes_subagent` | Delegated child-agent start / stop with goal and summary |
| `hermes_approval` | Dangerous-command approval requests and the user's response |

All measurements share common tag fields where applicable: `session_id`, `task_id`, `turn_id`, `model`, `provider`, `tool_name`, `status`. Every point is also tagged with `telemetry_schema_version` whenever Hermes supplies one (currently `hermes.observer.v1`).

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure credentials

```bash
cp .env.example .env
$EDITOR .env
```

Required variables:

| Variable | Description |
|---|---|
| `INFLUXDB_URL` | HTTP(S) address of your InfluxDB instance, e.g. `http://192.168.1.100:8086` |
| `INFLUXDB_TOKEN` | InfluxDB API token with write access to the target database |
| `INFLUXDB_ORG` | InfluxDB organization name |
| `INFLUXDB_DATABASE` | Destination database, e.g. `hermes-logs` |

Optional tuning variables:

| Variable | Default | Description |
|---|---|---|
| `INFLUXDB_BATCH_SIZE` | `500` | Points buffered before a forced flush |
| `INFLUXDB_FLUSH_INTERVAL_MS` | `10000` | Maximum ms between automatic flushes |
| `HERMES_LOGGER_MAX_TEXT_BYTES` | `8192` | Truncate text fields longer than this (0 = unlimited) |
| `HERMES_LOGGER_LOG_LEVEL` | `INFO` | Plugin log verbosity |

### 3. Install the plugin

Drop (or symlink) this directory into your Hermes user plugins folder and enable it:

```bash
ln -s /path/to/hermes-logger ~/.hermes/plugins/hermes-logger
hermes plugins enable hermes-logger
```

Hermes discovers the `plugin.yaml` manifest and calls `register(ctx)` from `__init__.py` at agent start.  If any required env var is missing the plugin logs a warning and skips registration so the agent still starts normally.

## InfluxDB schema overview

Every point additionally carries a `telemetry_schema_version` tag when Hermes supplies one (see `InfluxWriter._emit`). It is omitted below for brevity.

### `hermes_session`

Written by `write_session_start` / `write_session_end` / `write_session_reset`.

```
event="start" (on_session_start)
  tags:   session_id
  fields: platform, model

event="end" (on_session_end)
  tags:   session_id, completed, interrupted
  fields: reason

event="reset" (on_session_reset)
  tags:   old_session_id, new_session_id
  fields: reason
```

### `hermes_llm_turn`

Written by `write_llm_turn_start` (`pre_llm_call`) / `write_llm_turn_end` (`post_llm_call`).

```
event="turn_start"
  tags:   session_id, turn_id, model, platform, is_first_turn, sender_id
  fields: user_message, history_length

event="turn_end"
  tags:   session_id, turn_id, model, platform
  fields: user_message, assistant_response, history_length
```

### `hermes_api_request`

Written by `write_api_request_start` (`pre_api_request`) / `write_api_request_end` (`post_api_request`) / `write_api_request_error` (`api_request_error`).

```
event="api_start"
  tags:   session_id, task_id, turn_id, api_request_id, model, provider, platform,
          api_mode, base_url
  fields: api_call_count, message_count, tool_count, approx_input_tokens,
          request_char_count, max_tokens, request_payload

event="api_end"
  tags:   session_id, task_id, turn_id, api_request_id, model, provider, platform,
          api_mode, base_url, finish_reason
  fields: api_duration_ms, message_count, response_model, prompt_tokens,
          completion_tokens, total_tokens, usage_payload, assistant_content_chars,
          assistant_tool_call_count, response_payload

event="api_error"
  tags:   session_id, task_id, turn_id, api_request_id, model, provider, platform,
          api_mode, base_url, status_code, retryable
  fields: api_duration_ms, retry_count, max_retries, reason, error_type,
          error_message, request_payload
```

`request_payload` / `response_payload` are the sanitized JSON payloads Hermes builds for the provider request/response (includes reasoning/thinking blocks). `usage_payload` is the full raw `usage` object as JSON, in addition to the three normalized numeric token fields.

### `hermes_tool_call`

Written by `write_tool_start` (`pre_tool_call`) / `write_tool_end` (`post_tool_call`).

```
event="tool_start"
  tags:   session_id, task_id, turn_id, tool_call_id, tool_name, api_request_id
  fields: args

event="tool_end"
  tags:   session_id, task_id, turn_id, tool_call_id, tool_name, api_request_id,
          status, error_type
  fields: duration_ms, result, error_message
```

`status` is the observer-grade lifecycle outcome: `ok`, `error`, `blocked`, or `cancelled`.

### `hermes_subagent`

Written by `write_subagent_start` (`subagent_start`) / `write_subagent_stop` (`subagent_stop`).

```
event="subagent_start"
  tags:   parent_session_id, parent_subagent_id, child_session_id,
          child_subagent_id, child_role
  fields: parent_turn_id, child_goal

event="subagent_stop"
  tags:   parent_session_id, parent_subagent_id, child_session_id,
          child_subagent_id, child_role, status
  fields: duration_ms, child_summary
```

### `hermes_approval`

Written by `write_approval_request` (`pre_approval_request`) / `write_approval_response` (`post_approval_response`).

```
event="approval_request"
  tags:   session_id, task_id, turn_id, session_key, surface, pattern_key
  fields: command, description, pattern_keys

event="approval_response"
  tags:   session_id, task_id, turn_id, session_key, surface, pattern_key, choice
  fields: command, description, pattern_keys
```

`choice` reflects how the user resolved the prompt: `once`, `session`, `always`, `deny`, or `timeout`.

## Querying examples (Flux)

**All user prompts in the last hour:**
```flux
from(bucket: "hermes-logs")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "hermes_llm_turn" and r._field == "user_message" and r.event == "turn_start")
```

**Tool call durations by tool name:**
```flux
from(bucket: "hermes-logs")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "hermes_tool_call" and r._field == "duration_ms")
  |> group(columns: ["tool_name"])
  |> mean()
```

**Token usage per turn:**
```flux
from(bucket: "hermes-logs")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "hermes_api_request" and r._field == "total_tokens")
  |> group(columns: ["session_id", "turn_id"])
  |> sum()
```

**Failed tool calls:**
```flux
from(bucket: "hermes-logs")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "hermes_tool_call" and r.status == "error")
  |> keep(columns: ["_time", "tool_name", "error_type", "session_id"])
```

**Denied or timed-out approvals:**
```flux
from(bucket: "hermes-logs")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "hermes_approval" and (r.choice == "deny" or r.choice == "timeout"))
  |> keep(columns: ["_time", "session_id", "choice", "pattern_key"])
```

**Subagent tree for a parent session:**
```flux
from(bucket: "hermes-logs")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "hermes_subagent" and r.parent_session_id == "<session_id>")
  |> keep(columns: ["_time", "event", "child_session_id", "child_role", "child_subagent_id"])
```
