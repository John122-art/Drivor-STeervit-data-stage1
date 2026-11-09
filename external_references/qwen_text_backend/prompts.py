"""Motion formatting/navigation mapping from action-predic-test; Qwen-Drive VQA questions.

Images are separate upstream VQA blocks, identified by ordered labels in the question.
"""

import math

CAMERAS = (
    ("CAM_F0", "FRONT"),
    ("CAM_B0", "REAR"),
    ("CAM_L0", "FRONT LEFT"),
    ("CAM_R0", "FRONT RIGHT"),
)
PROMPT_VERSION = "qwen_drive_four_view_description_v1"
SYSTEM_MESSAGE = (
    "You describe visible driving scenes for a visual driving planner. "
    "Use the provided images, ego motion history, and navigation command. "
    "Describe only visually supported objects, road regions, and spatial relationships. "
    "Do not invent hidden objects, precise distances, other actors' velocities, or future actions. "
    "Treat possible future relevance as a possibility, not an observed fact. "
    "Navigation expresses the intended route; it is not evidence of a sign's arrow, "
    "lane direction, road geometry, or right of way. Describe these only when clearly visible. "
    "Prioritize pedestrians, vehicles, traversable road regions, boundaries, and obstacles "
    "relevant to the maneuver. Omit incidental scenery and uncertain sign readings."
)


def format_number(n, decimal_places=2):
    # Original helper from action-predic-test/recogdrive/utils/utils.py.
    return f"{n:+.{decimal_places}f}" if abs(round(n, decimal_places)) > 1e-2 else "0.0"


def navigation_text(command):
    values = [float(v) for v in command]
    if len(values) not in (3, 4) or not all(math.isfinite(v) for v in values):
        raise ValueError(f"Expected 3/4-element NAVSIM navigation command, got {command}")
    if any(v not in (0.0, 1.0) for v in values) or sum(values) > 1:
        raise ValueError(f"Invalid one-hot navigation command: {command}")
    if not any(values):
        return "unknown"
    return ["turn left", "go straight", "turn right", "unknown"][values.index(1.0)]


def build_qwen_questions(history_trajectories, high_commands, relative_times):
    """Build one question per scene; coordinates/times must use the current ego frame."""
    if not (len(history_trajectories) == len(high_commands) == len(relative_times)):
        raise ValueError("History, commands, and times must have matching batch sizes")
    questions = []
    for history, command, times in zip(history_trajectories, high_commands, relative_times):
        if len(history) != len(times) or not history:
            raise ValueError("Every history pose must have a timestamp")
        if any(t > 0 for t in times) or any(a >= b for a, b in zip(times, times[1:])) or times[-1] != 0:
            raise ValueError("History times must increase to current t=0; future inputs are forbidden")
        lines = []
        for pose, time in zip(history, times):
            if len(pose) != 3 or not all(math.isfinite(float(v)) for v in pose):
                raise ValueError("History poses must be finite (x,y,heading) triples")
            lines.append(f"  t={time:.3f}s: ({', '.join(format_number(float(v)) for v in pose)})")
        questions.append(
            "The four images show the same current scene in this order: "
            "Image 1: CAM_F0 FRONT; Image 2: CAM_B0 REAR; "
            "Image 3: CAM_L0 FRONT LEFT; Image 4: CAM_R0 FRONT RIGHT.\n"
            "Ego motion history (oldest to current):\n"
            "Coordinates are relative to the current ego pose: x forward, y left, "
            "positions in meters, heading in radians counterclockwise. Times are seconds "
            "relative to the current frame.\n" + "\n".join(lines) + "\n"
            f"Active navigation command: [{navigation_text(command).upper()}].\n"
            "Describe visible objects and road regions relevant to the current driving task "
            "and potentially relevant to the upcoming maneuver. Combine the views and navigation "
            "intent to identify visual content that deserves attention. Preserve view/direction "
            "references when useful. Output one concise English paragraph, about 40-80 words, "
            "grounded in the images. Include a brief driving rationale if useful, tying "
            "the intended maneuver to visible evidence. Do not output trajectory coordinates "
            "or a per-view list. Acknowledge uncertainty where necessary."
        )
    return questions


