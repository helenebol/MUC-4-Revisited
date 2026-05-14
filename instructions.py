"""
Shared instruction strings for training and inference
"""

FIELD_LABELS = [
    "event type",
    "date",
    "country",
    "city",
    "event stage",
    "weapon",
    "weapon type",
    "perpetrator category",
    "perpetrator individual",
    "perpetrator organization",
    "perpetrator confidence",
    "physical target",
    "physical target type",
    "physical target number",
    "effect on physical target",
    "victim name",
    "victim description",
    "victim type",
    "victim number",
    "effect on victim",
]

_LABEL_CSV = ", ".join(FIELD_LABELS)

MULTI_EVENT_INSTRUCTION = (
    f"Extract all events as a JSON array where each element is an object with fields: {_LABEL_CSV}. "
    f"Omit any field not mentioned. Reply with JSON only."
)
