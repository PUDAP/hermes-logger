# hermes-logger

A [Hermes observer plugin](https://github.com/NousResearch/hermes-agent/tree/main/docs/observability) that writes structured telemetry to **InfluxDB** for every user prompt, LLM API call (including reasoning payloads), tool call, and final agent response.

## Measurements

| Measurement | What it captures |
|---|---|
| `hermes_session` | Session start / end lifecycle |
| `hermes_llm_turn` | Per-turn **user prompt** (`turn_start`) and **final agent response** (`turn_end`) |
| `hermes_api_request` | Individual provider API attempts — timing, token usage, sanitized **request & response payloads** (LLM reasoning context) |
| `hermes_tool_call` | Tool call start / end — name, **args**, **result**, duration, status |
| `hermes_subagent` | Delegated child-agent start / stop with goal and summary |

All measurements share common tag fields: `session_id`, `turn_id`, and where applicable `model`, `provider`, `tool_name`, and `status`.

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
| `INFLUXDB_TOKEN` | InfluxDB API token with write access to the target bucket |
| `INFLUXDB_ORG` | InfluxDB organization name |
| `INFLUXDB_BUCKET` | Destination bucket, e.g. `hermes-logs` |

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

### `hermes_llm_turn`

```
tags:  session_id, turn_id, model, platform, is_first_turn
fields (turn_start): event="turn_start", user_message, history_length
fields (turn_end):   event="turn_end",   user_message, assistant_response, history_length
```

### `hermes_api_request`

```
tags:  session_id, turn_id, api_request_id, model, provider, finish_reason / status_code
fields (api_start): event, api_call_count, message_count, approx_input_tokens, request_payload
fields (api_end):   event, api_duration_ms, prompt_tokens, completion_tokens, total_tokens,
                    assistant_content_chars, assistant_tool_call_count, response_payload
fields (api_error): event, api_duration_ms, retry_count, error_type, error_message
```

### `hermes_tool_call`

```
tags:  session_id, turn_id, tool_call_id, tool_name, status, error_type
fields (tool_start): event="tool_start", args
fields (tool_end):   event="tool_end",   duration_ms, result, error_message
```

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
