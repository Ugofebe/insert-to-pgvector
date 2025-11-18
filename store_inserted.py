# import json
# import os
# import hashlib

# TITLES_JSON = "loaded_titles.json"

# def load_seen_titles(json_path=TITLES_JSON):
#     """Return a set of titles previously saved."""
#     if not os.path.exists(json_path):
#         return set()
#     with open(json_path, "r", encoding="utf-8") as f:
#         try:
#             data = json.load(f)
#             return set(data) if isinstance(data, list) else set()
#         except json.JSONDecodeError:
#             return set()

# def save_seen_titles(titles, json_path=TITLES_JSON):
#     """Write the set/list of titles to disk (overwrites)."""
#     os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(sorted(list(titles)), f, ensure_ascii=False, indent=2)

# def content_hash(text):
#     """Return short sha256 hex for given text (useful to disambiguate identical titles)."""
#     h = hashlib.sha256()
#     h.update(text.encode("utf-8"))
#     return h.hexdigest()[:16]
# store_inserted.py
import json
import os
import hashlib
import logging
from typing import Set, Dict, Any

logger = logging.getLogger(__name__)

TITLES_JSON = "loaded_titles.json"

def load_seen_titles(json_path: str = TITLES_JSON) -> Set[str]:
    """Return a set of content hashes previously processed."""
    if not os.path.exists(json_path):
        logger.info(f"No existing titles file found at {json_path}, starting fresh")
        return set()
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            titles = set(data) if isinstance(data, list) else set()
            logger.info(f"Successfully loaded {len(titles)} content hashes from {json_path}")
            return titles
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"Error loading content hashes from {json_path}: {str(e)}. Starting with empty set.")
        return set()

def save_seen_titles(content_hashes: Set[str], json_path: str = TITLES_JSON):
    """Write the set of content hashes to disk with atomic write for safety."""
    try:
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        temp_path = json_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(sorted(list(content_hashes)), f, ensure_ascii=False, indent=2)
        os.replace(temp_path, json_path)
        logger.debug(f"Successfully saved {len(content_hashes)} content hashes to {json_path}")
    except Exception as e:
        logger.error(f"Failed to save content hashes to {json_path}: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

def content_hash(text: str) -> str:
    """Return sha256 hex for given text content."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()

def generate_document_id(doc: Dict[str, Any]) -> str:
    """Generate unique ID for document based on title + content hash."""
    title = doc.get("title", "untitled")
    content = doc.get("content", "")
    
    # Create a unique identifier for this specific document chunk
    unique_string = f"{title}::{content_hash(content)}"
    return content_hash(unique_string)