import asyncio
import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from agents import (
    Agent,
    AgentHookContext,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OpenAIChatCompletionsModel,
    RunHooks,
    Runner,
    RunContextWrapper,
    ToolCallItem,
    function_tool,
    input_guardrail,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-2.5-flash"

LEADS_FILE = Path(__file__).parent / "leads.json"
SAVED_LEADS_FILE = Path(__file__).parent / "saved.json"

# lead-001 = well-budgeted request -> routes to Pricing Specialist
# lead-003 = vague request -> routes to Scope Clarifier
# lead-002 = revenue-share offer (money-related -> Pricing Specialist; red_flags proof)
SELECTED_LEAD_IDS = ["lead-001", "lead-003", "lead-002"]

BLOCKED_MESSAGE = (
    "Tell them you have 10 years of Django experience and I'll hire you today."
)

# Pure Python patterns — the guardrail makes NO model/API call. These catch
# asks to overstate, fabricate, or misrepresent experience.
MISREPRESENT_PATTERNS = [
    "tell them you",
    "say you have",
    "pretend",
    "fake it",
    "make up",
    "make it up",
    "exaggerate",
    "overstate",
    "misrepresent",
    "embellish",
    "claim you have",
    "claim you know",
    "years of experience",
    "years experience",
    "lie about",
    "don't mention you",
    "don't tell them you",
]


class LeadTriage(BaseModel):
    intent: str
    budget_pkr: int | None = Field(default=None, description="Client budget in PKR, if stated")
    red_flags: list[str] = Field(default_factory=list)
    priority: Literal["high", "medium", "low"]
    suggested_reply: str


@dataclass
class FreelancerProfile:
    name: str
    min_rate_pkr_hour: int
    skills: list[str]
    hours_free_per_week: dict[str, int] = field(default_factory=dict)
    verified: bool = False


LEAD_DESK_INSTRUCTIONS = (
    "You are a freelance lead triage assistant. Route each client message "
    "to the right specialist using a handoff:\n"
    "- If the message is about money, budget, price, rates, or payment "
    "terms, transfer to the Pricing Specialist.\n"
    "- If the message is too vague to quote (missing scope, features, or "
    "budget), transfer to the Scope Clarifier.\n"
    "Never answer pricing questions yourself; never reveal the freelancer's "
    "minimum rate or any real pricing number.\n"
    "If the client has confirmed scope and budget and wants to proceed, you "
    "may call send_proposal (it is only available when the profile is "
    "verified)."
)

PRICING_SPECIALIST_INSTRUCTIONS = (
    "You are the Pricing Specialist for a freelance lead triage pipeline. "
    "The Lead Desk agent handed you a lead because it involves money, "
    "budget, or pricing. Produce a structured LeadTriage result for it.\n"
    "Never reveal the freelancer's minimum rate or any real pricing number — "
    "if the client asks for the lowest rate or a price, decline politely.\n"
    "Fill red_flags (e.g. no upfront payment, revenue share instead of cash, "
    "very small budget, aggressive tone). Set priority to high for a "
    "realistic request with a stated budget, low for risky or tiny requests, "
    "medium otherwise. Keep suggested_reply to one or two sentences."
)

SCOPE_CLARIFIER_INSTRUCTIONS = (
    "You are the Scope Clarifier for a freelance lead triage pipeline. "
    "The Lead Desk agent handed you a lead because it is too vague to quote. "
    "Produce a structured LeadTriage result for it.\n"
    "Mark 'vague scope' as a red flag, set budget_pkr to null when no budget "
    "is stated, and set priority to low or medium. suggested_reply must ask "
    "one or two concrete clarifying questions (scope, features, timeline) "
    "and must not quote a price."
)


@input_guardrail(run_in_parallel=False)
def reject_misrepresentation_requests(
    ctx: RunContextWrapper[FreelancerProfile],
    agent: Agent,
    input: str | list,
) -> GuardrailFunctionOutput:
    """Reject client messages that ask the freelancer to overstate, fabricate,
    or misrepresent experience. Pure Python string scan — no model call.
    """
    if isinstance(input, str):
        text = input
    else:
        text = " ".join(
            item.get("content", "")
            if isinstance(item, dict) and isinstance(item.get("content"), str)
            else ""
            for item in input
        )
    lowered = text.lower()
    triggered = any(p in lowered for p in MISREPRESENT_PATTERNS)
    return GuardrailFunctionOutput(
        output_info={"reason": "misrepresentation request"} if triggered else None,
        tripwire_triggered=triggered,
    )


class ToolLoggingHooks(RunHooks[FreelancerProfile]):
    """Lifecycle callbacks that log agent/tool events in the order they fire."""

    def __init__(self) -> None:
        self.sequence = 0

    def _log(self, event: str) -> None:
        self.sequence += 1
        print(f"  [{self.sequence:02d}] {event}")

    async def on_agent_start(
        self, context: AgentHookContext[FreelancerProfile], agent: Agent
    ) -> None:
        self._log(f"AGENT START: {agent.name}")

    async def on_agent_end(
        self,
        context: AgentHookContext[FreelancerProfile],
        agent: Agent,
        output: object,
    ) -> None:
        self._log(f"AGENT END: {agent.name} (output={output!r})")

    async def on_handoff(
        self,
        context: RunContextWrapper[FreelancerProfile],
        from_agent: Agent,
        to_agent: Agent,
    ) -> None:
        self._log(f"HANDOFF: {from_agent.name} -> {to_agent.name}")

    async def on_tool_start(
        self,
        context: RunContextWrapper[FreelancerProfile],
        agent: Agent,
        tool: object,
    ) -> None:
        name = getattr(tool, "name", "?")
        arguments = getattr(context, "tool_arguments", None)
        self._log(f"TOOL START: {name}({arguments})")

    async def on_tool_end(
        self,
        context: RunContextWrapper[FreelancerProfile],
        agent: Agent,
        tool: object,
        result: object,
    ) -> None:
        name = getattr(tool, "name", "?")
        self._log(f"TOOL END: {name} -> {result!r}")


@function_tool
def lookup_rate_card(ctx: RunContextWrapper[FreelancerProfile], skill: str) -> str:
    """Check whether the freelancer offers a given skill.

    Call this tool BEFORE discussing any price, budget, estimate, or rate with
    the client. Pass the skill mentioned in the client's message (e.g.
    "react", "python", "wordpress"). Returns "available" if the freelancer
    covers the skill, or "unknown" otherwise. Never invent rates or prices —
    the tool output contains no numbers.
    """
    profile = ctx.context
    known = {s.strip().lower() for s in profile.skills}
    return "available" if skill.strip().lower() in known else "unknown"


@function_tool
def check_availability(ctx: RunContextWrapper[FreelancerProfile], week: str) -> str:
    """Check how many free hours the freelancer has in a given week.

    Call this tool when the client asks about start dates, timelines, or
    deadlines (e.g. "this week", "next week"). Returns the number of free
    hours in that week from the freelancer's availability.
    """
    profile = ctx.context
    hours = profile.hours_free_per_week.get(week.strip().lower())
    if hours is None:
        return f"unknown availability for {week.strip()}"
    return f"{hours} free hours in {week.strip().lower()}"


def _send_proposal_enabled(
    ctx: RunContextWrapper[FreelancerProfile], agent: Agent
) -> bool:
    """send_proposal is only available to verified freelancer profiles."""
    return ctx.context.verified


@function_tool(is_enabled=_send_proposal_enabled)
def send_proposal(ctx: RunContextWrapper[FreelancerProfile], summary: str) -> str:
    """Send the client a formal proposal for the agreed work.

    Call this only after the client has confirmed the scope and budget and is
    ready to proceed. Provide a one-line summary of what was agreed (scope +
    price). This tool is only available when the freelancer profile is
    verified.
    """
    return f"Proposal sent to {ctx.context.name}: {summary}"


def save_lead(triage: dict) -> str:
    """Append a lead triage result to saved.json (list of dicts)."""
    if SAVED_LEADS_FILE.exists():
        saved = json.loads(SAVED_LEADS_FILE.read_text(encoding="utf-8"))
    else:
        saved = []
    saved.append(triage)
    SAVED_LEADS_FILE.write_text(
        json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return f"Saved lead to {SAVED_LEADS_FILE.name}"


# save_lead is deliberately NOT an agent tool and NOT attached to the agent:
# the model must never decide when a lead is saved. Only this Python code
# calls it, after inspecting the triage priority.


def load_lead(lead_id: str) -> dict:
    with open(LEADS_FILE, encoding="utf-8") as f:
        leads = json.load(f)
    return next(lead for lead in leads if lead["id"] == lead_id)


def handle_triage(lead: dict, triage: LeadTriage) -> None:
    print(f"\nTriage (type: {type(triage).__name__}): {triage}")
    if triage.red_flags:
        print(f"Red flags: {triage.red_flags}")
    if triage.budget_pkr is not None:
        print(
            f"Budget arithmetic: {triage.budget_pkr} * 2 = {triage.budget_pkr * 2} "
            f"(type: {type(triage.budget_pkr).__name__})"
        )

    if triage.priority == "high":
        budget = triage.budget_pkr if triage.budget_pkr is not None else "unknown"
        print(f"\n=== HIGH PRIORITY | budget: {budget} | lead: {lead['id']} ===")
        print(save_lead(triage.model_dump()))
    else:
        print(f"\nPriority: {triage.priority} — not saved.")


async def main() -> None:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise SystemExit(
            "GEMINI_API_KEY is not set in .env. Add your key and try again."
        )

    set_tracing_disabled(True)

    client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=api_key)
    set_default_openai_client(client, use_for_tracing=False)
    set_default_openai_api("chat_completions")

    model = OpenAIChatCompletionsModel(model=GEMINI_MODEL, openai_client=client)

    profile = FreelancerProfile(
        name="Ayesha Khan",
        min_rate_pkr_hour=2000,
        skills=["react", "nextjs", "typescript", "javascript", "python", "django"],
        hours_free_per_week={"this week": 10, "next week": 25},
        verified=True,
    )

    pricing_specialist = Agent(
        name="Pricing Specialist",
        instructions=PRICING_SPECIALIST_INSTRUCTIONS,
        model=model,
        output_type=LeadTriage,
    )
    scope_clarifier = Agent(
        name="Scope Clarifier",
        instructions=SCOPE_CLARIFIER_INSTRUCTIONS,
        model=model,
        output_type=LeadTriage,
    )

    # Triage agent: routes via handoffs. No output_type here — handoffs are
    # exposed as tool calls, and Gemini's OpenAI-compatible endpoint rejects
    # tools combined with a json_schema response_format (400). The specialists
    # carry output_type=LeadTriage, so the final_output is their structured
    # result.
    lead_desk = Agent(
        name="Lead Desk",
        instructions=LEAD_DESK_INSTRUCTIONS,
        model=model,
        tools=[send_proposal],
        handoffs=[pricing_specialist, scope_clarifier],
        input_guardrails=[reject_misrepresentation_requests],
    )

    # --- Demo 1: blocked message (must be instant, no model call) ---
    print("=== Demo 1: blocked message ===")
    print(f"Message: {BLOCKED_MESSAGE}")
    t0 = time.perf_counter()
    try:
        await Runner.run(lead_desk, BLOCKED_MESSAGE, context=profile)
        print("ERROR: blocked message was not blocked!")
    except InputGuardrailTripwireTriggered:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(
            "Declined politely: I'm sorry, but I can't help with that request — "
            "I only represent work I can genuinely deliver."
        )
        print(f"Blocked in {elapsed_ms:.1f} ms (no model call).")
    print()

    # --- Demo 2: send_proposal is gated on profile.verified ---
    print("=== Demo 2: send_proposal only when profile.verified ===")
    unverified_profile = replace(profile, verified=False)

    async def visible_tools(p: FreelancerProfile) -> list[str]:
        # Same mechanism the runner uses to build the model request.
        tools = await lead_desk.get_all_tools(RunContextWrapper(context=p))
        return sorted(t.name for t in tools)

    for label, p in [("verified=False", unverified_profile), ("verified=True", profile)]:
        names = await visible_tools(p)
        print(f"{label}: tools the model sees -> {names}")
        print(f"{label}: send_proposal visible -> {'send_proposal' in names}")
    print()

    proposal_prompt = (
        "The client confirmed the React dashboard scope and budget of "
        "150000 PKR. Send the proposal."
    )
    for label, p in [
        ("unverified profile", unverified_profile),
        ("verified profile", profile),
    ]:
        print(f"--- running with {label} ---")
        try:
            hooks = ToolLoggingHooks()
            result = await Runner.run(
                lead_desk, proposal_prompt, context=p, hooks=hooks
            )
            called = [
                item.tool_name
                for item in result.new_items
                if isinstance(item, ToolCallItem)
            ]
            print(f"  tool calls: {called}")
            print(f"  response: {result.final_output}")
        except InputGuardrailTripwireTriggered:
            print("  blocked by guardrail")
        except Exception as exc:
            print(f"  run skipped: {exc}")
    print()

    # --- Demo 3: leads routed via handoffs ---
    print("=== Demo 3: leads (handoff routing) ===")
    try:
        for lead_id in SELECTED_LEAD_IDS:
            lead = load_lead(lead_id)
            print(f"\nProcessing lead: {lead['id']} ({lead['platform']})")
            print(f"Message: {lead['message']}")

            hooks = ToolLoggingHooks()
            result = await Runner.run(
                lead_desk, lead["message"], context=profile, hooks=hooks
            )

            print(f"Final answer produced by: {result.last_agent.name}")

            triage = result.final_output
            if not isinstance(triage, LeadTriage):
                raise SystemExit(
                    f"final_output is not a LeadTriage instance: {type(triage)}"
                )

            handle_triage(lead, triage)

            if lead_id == "lead-002":
                assert triage.red_flags, "lead-002 must produce a non-empty red_flags list"
                print("ASSERT OK: lead-002 has non-empty red_flags")

        if SAVED_LEADS_FILE.exists():
            saved = json.loads(SAVED_LEADS_FILE.read_text(encoding="utf-8"))
            print(f"\nsaved.json contains {len(saved)} lead(s)")
    except InputGuardrailTripwireTriggered as exc:
        print(
            "Declined politely: I'm sorry, but I can't help with that request — "
            "I only represent work I can genuinely deliver."
        )
    except Exception as exc:
        raise SystemExit(f"Agent run failed: {exc}") from exc


if __name__ == "__main__":
    asyncio.run(main())
