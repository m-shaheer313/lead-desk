# Lead Desk

An AI-powered freelance lead triage assistant built with the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) and Google's **Gemini 2.5 Flash**, accessed through the OpenAI-compatible Gemini endpoint.

Lead Desk reads inbound client messages (from Upwork, Fiverr, or anywhere else), routes them to specialist agents, guards against requests to misrepresent experience, and persists high-priority opportunities to a local JSON store — all from a single command.

## Features

- **Multi-agent routing** — a Lead Desk agent hands off to specialist agents based on message content:
  - **Pricing Specialist** — money, budget, price, or payment-term questions
  - **Scope Clarifier** — messages too vague to quote
- **Structured triage output** — every lead is classified into a typed `LeadTriage` result (`intent`, `budget_pkr`, `red_flags`, `priority`, `suggested_reply`), not free-form text
- **Input guardrail** — a pure-Python (zero API cost) guardrail rejects messages asking the freelancer to overstate, fabricate, or misrepresent experience; blocked messages return instantly
- **Privacy by design** — the freelancer's minimum rate never appears in instructions or any message sent to the model; it lives only in the run context
- **Context-gated tools** — `send_proposal` is only visible to the model when `profile.verified` is `True` (SDK `is_enabled` callback)
- **Deterministic persistence** — Python code, not the model, decides when a lead is saved: high-priority triage results are appended to `saved.json`
- **Lifecycle observability** — `RunHooks` callbacks log every agent start/end, handoff, and tool call (name, arguments, result) in fire order

## Architecture

```
Client message
      │
      ▼
┌─────────────────┐   tripwire (pure Python) ──► polite decline, ~ms
│  Lead Desk      │◄──────────────────────────────────────┐
│  (router)       │                                       │
└────────┬────────┘                                       │
         │ handoff (money / vague / other)                │
         ▼                                                 │
┌───────────────────┐  ┌──────────────────┐   ┌────────────────────────────┐
│ Pricing Specialist│  │Scope Clarifier   │   │ send_proposal (tool,       │
│ output_type=      │  │ output_type=     │   │  only if profile.verified) │
│ LeadTriage        │  │ LeadTriage       │   └────────────────────────────┘
└───────────────────┘  └──────────────────┘
         └─────────── LeadTriage ───────────┘
                      │
                      ▼
        priority == "high" ? ──► save to saved.json (Python decides)
```

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (project manager)
- A [Gemini API key](https://aistudio.google.com/apikey)

## Setup

```bash
# 1. Create the environment and install dependencies
uv sync

# 2. Configure your API key
#    Copy the example below into a new `.env` file
echo "GEMINI_API_KEY=your_key_here" > .env
```

Open `.env` and replace `your_key_here` with your real Gemini API key.

> `.env` is gitignored — never commit your key.

## Usage

Run the full demo with a single command:

```bash
uv run main.py
```

The demo exercises, in order:

1. **Blocked message** — a request to fabricate experience is declined instantly by the guardrail (no model call)
2. **Verified-gated tooling** — proves `send_proposal` is visible to the model only when `profile.verified` is `True`, then runs it against both profiles
3. **Lead routing** — processes sample leads from `leads.json`, showing which agent produced each answer and saving high-priority results to `saved.json`

### Tuning the demo

| What | Where |
|---|---|
| Which leads to process | `SELECTED_LEAD_IDS` in `main.py` |
| Sample lead messages | `leads.json` (6 leads: budgeted, revenue-share, vague, urgent, tiny, aggressive) |
| Freelancer profile (rate, skills, hours, verified) | `FreelancerProfile(...)` in `main()` |
| Saved results | `saved.json` (append-only; delete it to reset) |

### Verifying the tool schemas

```bash
uv run python -c "import main, json; print(json.dumps(main.send_proposal.params_json_schema, indent=2))"
uv run python -c "import main, json; print(json.dumps(main.lookup_rate_card.params_json_schema, indent=2))"
```

Both schemas expose only their business parameters — the run context is never part of the JSON schema the model sees.

## Project structure

```
lead-desk/
├── main.py          # Agents, tools, guardrail, hooks, demo runner
├── leads.json       # 6 sample client messages of varying quality
├── saved.json       # High-priority triage results (created at runtime)
├── .env             # GEMINI_API_KEY (gitignored)
├── pyproject.toml   # uv project config (Python 3.13+)
└── uv.lock          # Locked dependencies
```

## Notes & limitations

- The Gemini OpenAI-compatible endpoint does not accept `tools` + `response_format: json_schema` in a single request, so the triage agent routes via handoffs while specialist agents carry `output_type=LeadTriage`.
- Gemini free-tier accounts are limited to 20 requests/day for `gemini-2.5-flash`; expect HTTP 429s once exhausted. A paid tier or a fresh key resets this.
- `lookup_rate_card` and `check_availability` are defined as context-driven tools but are currently not attached to any agent (see the limitation above); they are ready to attach in a tool-enabled turn.

## License

MIT
