"""Single source of truth for EG2 prompt prefixes.

The research notes (unsloth-embeddinggemma2.md) flag prefix mismatch as a
silent quality killer: query and corpus sides must use these exact templates
at training, indexing, and query time. Every script that embeds should import
from here rather than redefining the strings.
"""

QUERY_PREFIX = "task: search result | query: {text}"
DOC_PREFIX = "title: {title} | text: {text}"


def format_query(text):
    return QUERY_PREFIX.format(text=text)


def format_doc(text, title=""):
    t = (title or "").strip() or "none"
    return DOC_PREFIX.format(title=t, text=text)