#modified template
TEMPLATE: list[tuple[str, str]] = [
    ("event type", "incident_type"),
    ("date", "incident_date"),
    ("country", "incident_location_country"),
    ("city", "incident_location_city"),
    ("event stage", "incident_stage_of_execution"),
    ("weapon", "incident_instrument_id"),
    ("weapon type", "incident_instrument_type"),
    ("perp. category", "perp_incident_category"),
    ("perp. individual", "perp_individual_id"),
    ("perp. org.", "perp_organization_id"),
    ("perp. confidence", "perp_organization_confidence"),
    ("phys target", "phys_tgt_id"),
    ("phys target type", "phys_tgt_type"),
    ("phys target number", "phys_tgt_number"),
    ("phys target effect", "phys_tgt_effect_of_incident"),
    ("victim name", "hum_tgt_name"),
    ("victim description", "hum_tgt_description"),
    ("victim type", "hum_tgt_type"),
    ("victim number", "hum_tgt_number"),
    ("victim effect", "hum_tgt_effect_of_incident"),
   
]

EXTRACTIVE_TEMPLATE: list[tuple[str, str]] = [
    ("weapon", "incident_instrument_id"),
    ("perpetrator individual", "perp_individual_id"),
    ("perpetrator organization", "perp_organization_id"),
    ("physical target", "phys_tgt_id"),
    ("victim name", "hum_tgt_name"),
    ("victim description", "hum_tgt_description"),
]


SIMPLIFIED_TEMPLATE: list[tuple[str, str]] = [
    ("event type", "incident_type"),
    ("weapon", "incident_instrument_id"),
    ("perpetrator individual", "perp_individual_id"),
    ("perpetrator organization", "perp_organization_id"),
    ("physical target", "phys_tgt_id"),
    ("victim name", "hum_tgt_name"),
]

SPLIT_NAME_MAP: dict[str, str] = {
    "train": "train",
    "validation": "val",
    "test": "test"
}



def get_template(template_type: str) -> list[tuple[str, str]]:
    if template_type == "extractive":
        return EXTRACTIVE_TEMPLATE
    else:
        return TEMPLATE