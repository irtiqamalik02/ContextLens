import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger("contextlens")


def parse_workspace_context(workspace_context: str) -> Dict[str, Any]:
    """
    Parse workspace context to extract repo names and tags for filtering.
    
    Expected formats:
    - "repo: repo-name-1, repo-name-2"
    - "repos: repo-name-1, repo-name-2"
    - "tag: api-docs, backend"
    - "tags: api-docs, backend"
    - Mixed: "repo: auth-service, payment-service | tag: backend"
    
    Args:
        workspace_context: The workspace context string from the user
    
    Returns:
        Dictionary with 'repo_names' and 'tags' lists
    """
    if not workspace_context or not workspace_context.strip():
        return {"repo_names": None, "tags": None}
    
    repo_names = []
    tags = []
    
    # Extract repo names
    # Match patterns like "repo: name1, name2" or "repos: name1, name2"
    repo_pattern = r'repos?:\s*([^\|\n]+)'
    repo_matches = re.findall(repo_pattern, workspace_context, re.IGNORECASE)
    
    for match in repo_matches:
        # Split by comma and clean up
        names = [name.strip() for name in match.split(',') if name.strip()]
        repo_names.extend(names)
    
    # Extract tags
    # Match patterns like "tag: tag1, tag2" or "tags: tag1, tag2"
    tag_pattern = r'tags?:\s*([^\|\n]+)'
    tag_matches = re.findall(tag_pattern, workspace_context, re.IGNORECASE)
    
    for match in tag_matches:
        # Split by comma and clean up
        tag_list = [tag.strip().lower() for tag in match.split(',') if tag.strip()]
        tags.extend(tag_list)
    
    # Remove duplicates while preserving order
    repo_names = list(dict.fromkeys(repo_names)) if repo_names else None
    tags = list(dict.fromkeys(tags)) if tags else None
    
    if repo_names or tags:
        logger.info(f"[context_parser] Extracted repo_names={repo_names}, tags={tags}")
    
    return {"repo_names": repo_names, "tags": tags}
