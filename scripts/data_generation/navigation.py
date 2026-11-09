"""NAVSIM command decoding adapted from DrivoR-main-copy/qwen_text/prompts.py.

Qwen's NAV_COMMANDS is imported from its official source by the generation
entrypoint. NAVSIM one-hot indices must not be used as Qwen command indices.
"""
import math


def navigation_text(command):
    """Match the existing reference DrivoR NAVSIM decoder, including unknown."""
    values = [float(v) for v in command]
    if len(values) not in (3, 4) or not all(math.isfinite(v) for v in values):
        raise ValueError(f"Expected 3/4-element NAVSIM navigation command, got {command}")
    if any(v not in (0.0, 1.0) for v in values) or sum(values) > 1:
        raise ValueError(f"Invalid one-hot navigation command: {command}")
    if not any(values):
        return "unknown"
    return ["turn left", "go straight", "turn right", "unknown"][values.index(1.0)]


def qwen_navigation_text(command, vocabulary):
    """Resolve by words against official Qwen NAV_COMMANDS, not argmax index."""
    text = navigation_text(command).upper()
    if text != "UNKNOWN" and text not in vocabulary:
        raise ValueError(f"NAVSIM command {text} is absent from Qwen NAV_COMMANDS")
    return text


def build_question(prompt, camera_name, navigation_command=None):
    if "{camera_name}" not in prompt:
        raise ValueError("Prompt must contain {camera_name}")
    if ("{navigation_command}" in prompt) != (navigation_command is not None):
        raise ValueError("Navigation prompt and navigation input must be enabled together")
    question = prompt.replace("{camera_name}", camera_name)
    if navigation_command is not None:
        question = question.replace("{navigation_command}", navigation_command)
    return question
