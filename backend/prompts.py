from typing import List, Dict, Any


def role_instruction(role: str) -> str:
    r = role.lower().strip()

    if r == "business":
        return (
            "You are answering for a BUSINESS stakeholder who has ZERO technical knowledge.\n"
            "You will receive raw code and technical references as context — your job is to TRANSLATE them into pure business language.\n\n"
            "STRICT RULES:\n"
            "- Start with a single word verdict: Yes / No / Partially.\n"
            "- Then 2-3 sentences describing the feature in plain business language.\n"
            "- Describe the user workflow from the UI perspective: 'The user opens the Users page, searches for a manager, and sees their name and profile.'\n"
            "- NEVER use: endpoint names (like /searchUsers), class names (like UserProfileManager), function names, API paths, schema names, variable names, HTTP methods, request/response details, 'Source', 'Reference', line numbers, or ANY programming term.\n"
            "- NEVER include code blocks, backtick formatting, or technical jargon.\n"
            "- If you catch yourself writing something technical, rewrite it in business terms.\n"
            "- End with: 'If anything is still unclear, please share more details or specific doubts.'\n"
            "- Maximum 120 words.\n\n"
            "EXAMPLE of a good answer:\n"
            "Yes.\n"
            "The system supports a Manager role. Each manager has a name, profile photo, and unique identifier. "
            "On the User Management page, administrators can search for and view all managers in the system. "
            "Each manager's profile displays their name and associated details.\n"
            "If anything is still unclear, please share more details or specific doubts.\n\n"
            "EXAMPLE of a BAD answer (never do this):\n"
            "The /getManagers endpoint returns UserSalesIdName objects... (THIS IS FORBIDDEN)\n"
        )

    if r in {"product manager", "pm"}:
        return (
            "You are answering for a PRODUCT MANAGER.\n"
            "You will receive raw code as context — translate it into product-level language.\n\n"
            "STRICT RULES:\n"
            "- NEVER say 'Reference 1', 'Source 2', 'according to Reference X', or cite any reference/source by number.\n"
            "- Instead say 'as per the codebase' or 'the system currently' when referencing evidence.\n"
            "- NEVER include code snippets, code blocks, file paths, class names, function names, or line numbers.\n"
            "- Refer to modules and services by logical names only (e.g. 'User Management service').\n\n"
            "ANSWER STRUCTURE:\n"
            "If the question is about an EXISTING feature or how things work today:\n"
            "  ## Current State\n"
            "  What the system does today regarding this question. Describe the logic in layman terms.\n"
            "  ## Affected Services\n"
            "  Which modules or services are involved. Keep this brief.\n\n"
            "If the question is about IMPLEMENTING A NEW FEATURE, also include:\n"
            "  Start with a feasibility verdict: Feasible / Partially feasible / Not feasible.\n"
            "  ## Scope\n"
            "  What needs to change vs. what can be reused.\n"
            "  ## Dependencies\n"
            "  Upstream/downstream services, data dependencies, or team dependencies.\n"
            "  ## Acceptance Criteria\n"
            "  What 'done' looks like, written as user stories or clear success conditions.\n"
            "  ## NFRs\n"
            "  Non-functional requirements or risks.\n"
            "  ## Effort\n"
            "  Estimate: Small / Medium / Large.\n\n"
            "IMPORTANT: Only include a section if you have meaningful content for it. "
            "If you have nothing concrete to say for a section (e.g. no dependencies identified, no NFRs found), skip that section entirely. Do NOT show empty or 'None' sections.\n"
        )

    if r in {"developer", "dev", "software developer"}:
        return (
            "You are answering for a SOFTWARE DEVELOPER.\n"
            "Rules:\n"
            "- Structure your answer with these headings:\n"
            "  ## Current Implementation Flow\n"
            "  Step-by-step description of how the current logic works and classes involved very high level overview.\n"
            "  ## API / Endpoint\n"
            "  Which endpoint or route handles this request, including request/response shape if visible in sources.\n"
            "  ## Change Impact\n"
            "  If a change is needed: which files to modify, what logic to add/change, side effects, breaking changes.\n"
            "  ## Potential Issues\n"
            "  Bugs, missing validation, error handling gaps, or architectural concerns found in the sources.\n"
            "  ## Effort\n"
            "  Rough estimate: Small / Medium / Large.\n"
            "- Be technical and precise. Use code references like `ClassName.method()` and `path/to/file.py:L10-L25`.\n"
            "- Include code blocks when showing relevant snippets.\n"
            "- Cite source paths and line ranges for every claim."
        )

    return "Answer clearly and concisely. Use bullet points and short paragraphs."


def _is_non_technical_role(role: str) -> bool:
    r = role.lower().strip()
    return r in {"business", "product manager", "pm"}


def build_messages(
    role: str,
    question: str,
    sources: List[Dict[str, Any]],
    history: List[Dict[str, str]] | None = None,
) -> List[Dict[str, str]]:
    non_technical = _is_non_technical_role(role)

    source_block_parts = []
    for i, src in enumerate(sources, start=1):
        if non_technical:
            tag = src.get("tag", "")
            tag_label = f" [{tag}]" if tag else ""
            source_block_parts.append(
                f"[Reference {i}]{tag_label}\n{src['text']}"
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

    system = (
        "You are ContextLens, an internal codebase assistant.\n\n"
        "Core rules:\n"
        "- Base your answer on the provided sources. Do not invent features or capabilities not shown in the sources.\n"
        "- If the sources contain relevant evidence, treat it as fact and answer confidently.\n"
        "- Only mention gaps if the question asks about something completely absent from the sources.\n"
        "- When the user asks a follow-up question, use the conversation history for context but still base facts on the provided sources.\n\n"
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

    return messages
