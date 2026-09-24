from typing import Any, Dict, List, Tuple  # noqa: UP035

CONTEXT_WINDOW = 16384
MAX_PROMPT_TOKENS = 7000


class ContextWindowExceededError(Exception):
    def __init__(self, estimated_tokens: int, limit: int = MAX_PROMPT_TOKENS):
        self.estimated_tokens = estimated_tokens
        self.limit = limit
        super().__init__(
            f"Prompt too large: ~{estimated_tokens} tokens (limit {limit}). "
            f"Please clear your chat history and try again."
        )


def estimate_tokens(text: str) -> int:
    return len(text) // 4


ROLE_ALIASES = {
    "business": "business",
    "product manager": "pm",
    "developer": "dev"
}


def normalize_role(role: str) -> str:
    r = role.lower().strip()
    return ROLE_ALIASES.get(r, "business")


def _is_non_technical_role(role: str) -> bool:
    return normalize_role(role) in {"business", "pm"}


def _base_rules() -> str:
    return (
        "Shared rules for all roles:\n"
        "- Answer only from the provided sources and the current repository scope.\n"
        "- Do not invent features, flows, permissions, validations, or integrations not shown in the sources.\n"
        "- If evidence is missing, say so clearly and keep the answer limited to what can be proven.\n"
        "- If the question is outside the repo/code scope, say that it cannot be answered from the available code.\n"
        "- Consider both frontend and backend behavior when relevant.\n"
        "- Do not assume the frontend is the final source of truth; backend validation may still restrict behavior.\n"
        "- Do not assume missing frontend validation means the backend allows it.\n"
        "- If an external system or library is involved, treat it as an external dependency unless the repo shows wrapper code or config points.\n"
        "- Do not assume we can change third-party products like ForgeRock directly; only describe changes possible in this repo or code we control.\n"
        "- Keep the answer concise.\n\n"
    )


def role_instruction(role: str) -> str:
    r = normalize_role(role)
    base = _base_rules()

    if r == "business":
        return (
            base +
            "You are answering for a BUSINESS stakeholder with zero technical background.\n"
            "Rules:\n"
            "- Use plain business language only.\n"
            "- Do not mention file names, class names, function names, endpoints, APIs, request/response shapes, HTTP methods, line numbers, or code terms.\n"
            "- Do not use internal field names like salesId, tenantId, teamId, etc. Use everyday language (e.g. 'employee identifier' or just omit them).\n"
            "- If the question is a yes/no question, start with Yes / No / Partially. Otherwise start with a direct one-line summary.\n"
            "- Then give a short plain-English explanation of what the system does.\n"
            "- If relevant, explain the user experience or business outcome.\n"
            "- Maximum 150 words.\n"
        )

    if r == "pm":
        return (
            base +
            "You are answering for a PRODUCT MANAGER. Translate code into product-level language.\n\n"
            "RULES:\n"
            "- Never cite sources by number. Say 'the system currently' instead.\n"
            "- No code snippets, file paths, class/function names, repository names, or line numbers.\n"
            "- No internal field names (salesId, tenantId, teamId, userId). Use everyday language or omit them.\n"
            "- No technical jargon (API, endpoint, repository, middleware, payload, schema, DTO). Use product language.\n"
            "- Refer to parts of the system by their product purpose (e.g. 'Team Management', 'User Profiles'), not by code names.\n"
            "- Aim for 200 words or fewer.\n\n"
            "FIRST decide the question type:\n\n"
            "TYPE 1 — CLARIFYING / EXPLAINING (e.g. 'How does X work?', 'What happens when...?', 'Does the system support...?'):\n"
            "  Start with a one-line direct answer.\n"
            "  Then explain the current behavior in plain language — what the user sees, what the system does, any rules or limitations.\n"
            "  Do NOT include feasibility, scope, effort, or impact sections.\n\n"
            "TYPE 2 — NEW FEATURE / CHANGE REQUEST (e.g. 'Can we add...?', 'What would it take to...?', 'Is it feasible to...?'):\n"
            "  Start with a one-line feasibility verdict: Feasible / Partially feasible / Not feasible.\n"
            "  ## What Exists Today — briefly describe current behavior relevant to the request.\n"
            "  ## What Needs to Change — what's new vs. what can be reused.\n"
            "  ## Impact — other areas affected, risks, or concerns.\n"
            "  ## Effort — Small / Medium / Large.\n\n"
            "Skip any section with nothing meaningful to say.\n"
        )



    return (
        base +
        "You are answering for a SOFTWARE DEVELOPER.\n"
        "Rules:\n"
        "- Be technical and precise.\n"
        "- Cite source paths and line ranges for every factual claim.\n"
        "- Use code references when supported by the sources.\n"
        "- If frontend and backend differ, call that out explicitly.\n"
        "- Generate good code when asked for help with code.\n\n"
        "STRUCTURE:\n"
        "For questions about EXISTING FLOWS or HOW SOMETHING WORKS:\n"
        "  - Explain the implementation flow, entry points, validation, and side effects.\n"
        "  - If you spot potential bugs, race conditions, missing validations, or edge cases, highlight them in a ## Potential Issues section.\n\n"
        "For questions about IMPLEMENTING A NEW FEATURE or MAKING A CHANGE:\n"
        "  - Explain what exists today, then cover affected services, required changes, and external dependencies.\n"
        "  - If a change depends on external systems or libraries, clearly mark that as out of repo scope unless the repo contains a control point.\n"
    )


def build_messages(
    role: str,
    question: str,
    sources: List[Dict[str, Any]],
    history: List[Dict[str, str]] | None = None,
    workspace_context: str = "",
) -> Tuple[List[Dict[str, str]], int]:
    non_technical = _is_non_technical_role(role)

    source_block_parts = []
    for i, src in enumerate(sources, start=1):
        if non_technical:
            tag = src.get("tag", "")
            tag_label = f" [{tag}]" if tag else ""
            source_block_parts.append(
                f"[Context {i}]{tag_label}\n{src['text']}"
            )
        else:
            label = src.get("repo_name", "")
            tag = src.get("tag", "")
            prefix = f"[Source {i}]"
            if label:
                prefix += f" [{label}]"
            if tag:
                prefix += f" [{tag}]"
            source_block_parts.append(
                f"{prefix} {src['path']} lines {src['start_line']}-{src['end_line']}\n{src['text']}"
            )

    source_block = "\n\n".join(source_block_parts) if source_block_parts else "No sources found."

    workspace_block = ""
    if workspace_context and workspace_context.strip():
        workspace_block = (
            "Project context (use this to understand how the repositories relate to each other):\n"
            f"{workspace_context.strip()}\n\n"
        )

    system = (
        "You are ContextLens, an internal codebase assistant.\n\n"
        f"{workspace_block}"
        "Core rules:\n"
        "- Base your answer only on the provided sources and repo scope.\n"
        "- Do not invent features, flows, permissions, validations, or integrations not shown in the sources.\n"
        "- Check both frontend and backend behavior when relevant.\n"
        "- If frontend and backend differ, explain the difference.\n"
        "- If the evidence is missing or the question is out of scope, say so clearly.\n"
        "- Treat external libraries and third-party systems as dependencies, not editable repo code, unless the sources show a wrapper or configuration point.\n"
        "- Use conversation history only for context, not as evidence.\n\n"
        f"{role_instruction(role)}"
    )

    user = f"""Question:
{question}

Relevant sources:
{source_block}"""

    messages = [{"role": "system", "content": system}]

    if history:
        for msg in history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user})

    total_text = "".join(m["content"] for m in messages)
    token_estimate = estimate_tokens(total_text)
    if token_estimate > MAX_PROMPT_TOKENS:
        raise ContextWindowExceededError(token_estimate)

    return messages, token_estimate