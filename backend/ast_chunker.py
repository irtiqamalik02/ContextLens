import logging
from typing import List, Tuple, Optional

from tree_sitter import Language, Parser, Node

import config

logger = logging.getLogger("contextlens")

# ---------------------------------------------------------------------------
# Language registry
# ---------------------------------------------------------------------------

_PARSERS: dict[str, Parser] = {}

_EXT_TO_LANG: dict[str, tuple[str, callable]] = {}


def _register_languages():
    """Populate the extension→language map on first use."""
    if _EXT_TO_LANG:
        return

    try:
        import tree_sitter_python
        _EXT_TO_LANG[".py"] = ("python", tree_sitter_python.language)
    except ImportError:
        pass

    try:
        import tree_sitter_javascript
        _EXT_TO_LANG[".js"] = ("javascript", tree_sitter_javascript.language)
        _EXT_TO_LANG[".jsx"] = ("javascript", tree_sitter_javascript.language)
    except ImportError:
        pass

    try:
        import tree_sitter_typescript
        _EXT_TO_LANG[".ts"] = ("typescript", tree_sitter_typescript.language_typescript)
        _EXT_TO_LANG[".tsx"] = ("tsx", tree_sitter_typescript.language_tsx)
    except ImportError:
        pass

    try:
        import tree_sitter_java
        _EXT_TO_LANG[".java"] = ("java", tree_sitter_java.language)
    except ImportError:
        pass

    try:
        import tree_sitter_kotlin
        _EXT_TO_LANG[".kt"] = ("kotlin", tree_sitter_kotlin.language)
    except ImportError:
        pass


def _get_parser(ext: str) -> Optional[Parser]:
    """Return a cached Parser for the given file extension, or None."""
    _register_languages()

    entry = _EXT_TO_LANG.get(ext.lower())
    if entry is None:
        return None

    lang_name, lang_fn = entry
    if lang_name not in _PARSERS:
        lang = Language(lang_fn())
        parser = Parser(lang)
        _PARSERS[lang_name] = parser

    return _PARSERS[lang_name]


def is_ast_supported(ext: str) -> bool:
    _register_languages()
    return ext.lower() in _EXT_TO_LANG


# ---------------------------------------------------------------------------
# Node‑type definitions per language
# ---------------------------------------------------------------------------

_TOP_LEVEL_NODES: dict[str, set[str]] = {
    "python": {"class_definition", "function_definition"},
    "javascript": {"class_declaration", "function_declaration", "lexical_declaration", "export_statement"},
    "typescript": {"class_declaration", "function_declaration", "lexical_declaration", "export_statement",
                   "interface_declaration", "type_alias_declaration", "enum_declaration"},
    "tsx": {"class_declaration", "function_declaration", "lexical_declaration", "export_statement",
            "interface_declaration", "type_alias_declaration", "enum_declaration"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration"},
    "kotlin": {"class_declaration", "function_declaration", "object_declaration"},
}

_METHOD_NODES: dict[str, set[str]] = {
    "python": {"function_definition"},
    "javascript": {"method_definition"},
    "typescript": {"method_definition", "public_field_definition"},
    "tsx": {"method_definition", "public_field_definition"},
    "java": {"method_declaration", "constructor_declaration"},
    "kotlin": {"function_declaration", "property_declaration"},
}


# ---------------------------------------------------------------------------
# Entity name extraction
# ---------------------------------------------------------------------------

def _node_name(node: Node) -> str:
    """Try to extract a human-readable name from an AST node."""
    for child in node.children:
        if child.type == "identifier" or child.type == "name":
            return child.text.decode("utf-8", errors="ignore")
        if child.type == "property_identifier":
            return child.text.decode("utf-8", errors="ignore")
    return ""


def _class_name(node: Node) -> str:
    """Walk up to find the enclosing class name."""
    parent = node.parent
    while parent is not None:
        if parent.type in ("class_declaration", "class_definition", "object_declaration"):
            return _node_name(parent)
        parent = parent.parent
    return ""


# ---------------------------------------------------------------------------
# AST → Chunks
# ---------------------------------------------------------------------------

def _node_lines(node: Node) -> Tuple[int, int]:
    """Return 1-indexed (start_line, end_line) for a node."""
    return node.start_point[0] + 1, node.end_point[0] + 1


def _node_text(source_lines: List[str], start: int, end: int) -> str:
    """Extract text from source lines (1-indexed)."""
    return "\n".join(source_lines[start - 1 : end]).strip()


def _split_large_chunk(
    source_lines: List[str], start: int, end: int, max_lines: int, overlap: int = 8
) -> List[Tuple[int, int]]:
    """Split a range that exceeds max_lines into overlapping sub-ranges."""
    ranges = []
    pos = start
    while pos <= end:
        chunk_end = min(end, pos + max_lines - 1)
        ranges.append((pos, chunk_end))
        if chunk_end >= end:
            break
        pos = chunk_end - overlap + 1
    return ranges


def chunk_by_ast(text: str, ext: str) -> List[Tuple[int, int, str]]:
    """
    Parse *text* using tree-sitter for the given file extension and return
    chunks aligned to logical code boundaries.

    Returns the same format as chunk_by_lines: List[(start_line, end_line, chunk_text)]
    """
    parser = _get_parser(ext)
    if parser is None:
        return []

    _register_languages()
    entry = _EXT_TO_LANG.get(ext.lower())
    if entry is None:
        return []
    lang_name = entry[0]

    source_bytes = text.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    source_lines = text.splitlines()
    total_lines = len(source_lines)
    if total_lines == 0:
        return []

    top_types = _TOP_LEVEL_NODES.get(lang_name, set())
    method_types = _METHOD_NODES.get(lang_name, set())

    # Collect top-level AST nodes
    ast_regions: List[Tuple[int, int, str, str]] = []  # (start, end, chunk_type, entity_name)

    for child in root.children:
        if child.type not in top_types:
            continue

        start_l, end_l = _node_lines(child)
        entity = _node_name(child)
        node_size = end_l - start_l + 1

        # For classes: try to extract individual methods as separate chunks
        if child.type in ("class_declaration", "class_definition", "object_declaration",
                          "interface_declaration", "enum_declaration"):
            class_name = entity
            body = None
            for sub in child.children:
                if sub.type in ("class_body", "block", "interface_body", "enum_body",
                                "class_definition_body"):
                    body = sub
                    break

            if body is not None and node_size > config.AST_MAX_CHUNK_LINES:
                # Class is large — chunk by methods
                method_regions = []
                for member in body.children:
                    if member.type in method_types:
                        ms, me = _node_lines(member)
                        m_name = _node_name(member)
                        full_name = f"{class_name}.{m_name}" if m_name else class_name
                        method_regions.append((ms, me, "method", full_name))

                if method_regions:
                    # Add class header (everything before first method)
                    first_method_start = method_regions[0][0]
                    if first_method_start > start_l:
                        ast_regions.append((start_l, first_method_start - 1, "class_header", class_name))
                    ast_regions.extend(method_regions)
                    continue

            # Small class or no methods found — keep as one chunk
            ast_regions.append((start_l, end_l, "class", class_name))
        else:
            ast_regions.append((start_l, end_l, "function", entity))

    if not ast_regions:
        # No recognisable top-level nodes — fall back to line-based
        return []

    # Sort by start line
    ast_regions.sort(key=lambda r: r[0])

    # Capture orphan code between AST nodes (imports, constants, etc.)
    all_regions: List[Tuple[int, int, str, str]] = []

    prev_end = 0
    for start_l, end_l, ctype, ename in ast_regions:
        if start_l > prev_end + 1:
            orphan_text = _node_text(source_lines, prev_end + 1, start_l - 1)
            if orphan_text.strip():
                all_regions.append((prev_end + 1, start_l - 1, "preamble", ""))
        all_regions.append((start_l, end_l, ctype, ename))
        prev_end = end_l

    # Trailing code after last AST node
    if prev_end < total_lines:
        trailing = _node_text(source_lines, prev_end + 1, total_lines)
        if trailing.strip():
            all_regions.append((prev_end + 1, total_lines, "preamble", ""))

    # Merge small adjacent regions & split oversized ones
    chunks: List[Tuple[int, int, str]] = []

    merge_buf_start: Optional[int] = None
    merge_buf_end: Optional[int] = None

    def _flush_merge():
        nonlocal merge_buf_start, merge_buf_end
        if merge_buf_start is not None:
            txt = _node_text(source_lines, merge_buf_start, merge_buf_end)
            if txt:
                chunks.append((merge_buf_start, merge_buf_end, txt))
            merge_buf_start = None
            merge_buf_end = None

    for start_l, end_l, ctype, ename in all_regions:
        size = end_l - start_l + 1

        if size > config.AST_MAX_CHUNK_LINES:
            _flush_merge()
            sub_ranges = _split_large_chunk(source_lines, start_l, end_l, config.AST_MAX_CHUNK_LINES)
            for rs, re_ in sub_ranges:
                txt = _node_text(source_lines, rs, re_)
                if txt:
                    chunks.append((rs, re_, txt))
        elif size < config.AST_MIN_CHUNK_LINES:
            # Merge with adjacent small regions
            if merge_buf_start is None:
                merge_buf_start = start_l
                merge_buf_end = end_l
            else:
                merged_size = end_l - merge_buf_start + 1
                if merged_size <= config.AST_MAX_CHUNK_LINES:
                    merge_buf_end = end_l
                else:
                    _flush_merge()
                    merge_buf_start = start_l
                    merge_buf_end = end_l
        else:
            _flush_merge()
            txt = _node_text(source_lines, start_l, end_l)
            if txt:
                chunks.append((start_l, end_l, txt))

    _flush_merge()

    if not chunks:
        return []

    logger.info(f"[ast_chunker] Produced {len(chunks)} AST chunks for {ext} file ({total_lines} lines)")
    return chunks
