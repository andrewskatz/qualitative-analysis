"""
Type definitions and utilities for figurative language types.
"""

from typing import List, Optional


# All supported figurative language types with their definitions
FIGURATIVE_TYPE_DEFINITIONS = {
    "metaphor": 'Metaphor: A figure of speech in which a word or phrase literally denoting one kind of object or idea is used in place of another to suggest a likeness or analogy between them (e.g., "drowning in money", "Time is a thief")',
    "simile": 'Simile: A comparison using "like" or "as" (e.g., "Fast as lightning")',
    "personification": 'Personification: Giving human qualities to non-human things (e.g., "The wind whispered")',
    "hyperbole": 'Hyperbole: Extreme exaggeration (e.g., "I\'m so hungry I could eat a horse")',
    "idiom": 'Idiom: A phrase with a meaning different from its literal interpretation (e.g., "It\'s raining cats and dogs")',
    "irony": "Irony: Saying the opposite of what is meant, or when context contradicts the literal meaning",
    "extended_metaphor": "Extended metaphor: A metaphor that continues across multiple sentences or builds on earlier imagery",
    "analogy": "Analogy: A comparison of two otherwise unlike things based on resemblance of a particular aspect",
    "other": "Other: Any other form of non-literal language use",
}

# Default order for types when not filtered
DEFAULT_TYPE_ORDER = [
    "metaphor",
    "simile",
    "personification",
    "hyperbole",
    "idiom",
    "irony",
    "extended_metaphor",
    "analogy",
    "other",
]


def get_all_types() -> List[str]:
    """Get all supported figurative language types."""
    return list(DEFAULT_TYPE_ORDER)


def validate_types(types: List[str]) -> List[str]:
    """
    Validate and normalize figurative language type names.
    
    Returns list of valid type names, ignoring invalid ones.
    Logs a warning for any unrecognized types.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    valid = []
    for t in types:
        normalized = t.lower().strip().replace(" ", "_").replace("-", "_")
        if normalized in FIGURATIVE_TYPE_DEFINITIONS:
            valid.append(normalized)
        else:
            logger.warning(f"Ignoring unrecognized figurative type: '{t}'")
    
    return valid


def generate_types_section(types: Optional[List[str]] = None) -> str:
    """
    Generate the figurative types section for prompt templates.
    
    Args:
        types: List of figurative types to include. If None, includes all types.
        
    Returns:
        Formatted string with type definitions for inclusion in prompts.
    """
    if types is None:
        types = DEFAULT_TYPE_ORDER
    else:
        types = validate_types(types)
        if not types:
            types = DEFAULT_TYPE_ORDER
    
    lines = ["Figurative language includes:"]
    for t in types:
        if t in FIGURATIVE_TYPE_DEFINITIONS:
            lines.append(f"- {FIGURATIVE_TYPE_DEFINITIONS[t]}")
    
    return "\n".join(lines)


def get_valid_types_hint(types: Optional[List[str]] = None) -> str:
    """
    Generate a hint string for valid type values in LLM output.
    
    Args:
        types: List of figurative types to include. If None, includes all types.
        
    Returns:
        String like "metaphor|simile|personification|..." for use in prompts.
    """
    if types is None:
        types = DEFAULT_TYPE_ORDER
    else:
        types = validate_types(types)
        if not types:
            types = DEFAULT_TYPE_ORDER
    
    return "|".join(types)
