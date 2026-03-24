from dataclasses import dataclass
from typing import Any


ALLOWED_FIELD_TYPES = {
    "arabic_text",
    "date",
    "time",
    "syp_price",
    "usd_price",
}

ALLOWED_OCR_ENGINES = {
    "paddle",
    "tesseract",
}

ALLOWED_PREPROCESS_MODES = {
    "binary",
    "adaptive",
    "contrast",
    "soft",
}

REQUIRED_FIELD_IDS = {
    "day",
    "date",
    "time",
    "k21_syp_sell",
    "k21_syp_buy",
    "k21_usd_sell",
    "k21_usd_buy",
    "k18_syp_sell",
    "k18_syp_buy",
    "k18_usd_sell",
    "k18_usd_buy",
}


@dataclass
class BlueprintValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_box(field_id: str, box: dict, errors: list[str]):
    if not isinstance(box, dict):
        errors.append(f"field '{field_id}': box must be an object")
        return

    for key in ("x1", "y1", "x2", "y2"):
        if key not in box:
            errors.append(f"field '{field_id}': missing box.{key}")
            continue
        if not _is_number(box[key]):
            errors.append(f"field '{field_id}': box.{key} must be numeric")
            continue
        if not (0.0 <= float(box[key]) <= 1.0):
            errors.append(f"field '{field_id}': box.{key} must be between 0.0 and 1.0")

    if all(k in box and _is_number(box[k]) for k in ("x1", "y1", "x2", "y2")):
        x1, y1, x2, y2 = map(float, (box["x1"], box["y1"], box["x2"], box["y2"]))
        if x1 >= x2:
            errors.append(f"field '{field_id}': box.x1 must be less than box.x2")
        if y1 >= y2:
            errors.append(f"field '{field_id}': box.y1 must be less than box.y2")


def _validate_string_list(
    field_id: str,
    key: str,
    values: Any,
    allowed: set[str],
    errors: list[str],
):
    if not isinstance(values, list) or not values:
        errors.append(f"field '{field_id}': {key} must be a non-empty list")
        return

    for item in values:
        if not isinstance(item, str):
            errors.append(f"field '{field_id}': {key} items must be strings")
            continue
        if item not in allowed:
            errors.append(
                f"field '{field_id}': invalid {key} value '{item}', allowed={sorted(allowed)}"
            )


def validate_blueprint(blueprint: dict) -> BlueprintValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(blueprint, dict):
        return BlueprintValidationResult(
            ok=False,
            errors=["blueprint must be a JSON object"],
            warnings=[],
        )

    required_top = [
        "template_name",
        "version",
        "is_active",
        "reference_size",
        "alignment",
        "validation",
        "fields",
    ]
    for key in required_top:
        if key not in blueprint:
            errors.append(f"missing top-level key '{key}'")

    if errors:
        return BlueprintValidationResult(ok=False, errors=errors, warnings=warnings)

    if not isinstance(blueprint["template_name"], str) or not blueprint["template_name"].strip():
        errors.append("template_name must be a non-empty string")

    if not isinstance(blueprint["version"], int) or blueprint["version"] < 1:
        errors.append("version must be an integer >= 1")

    if not isinstance(blueprint["is_active"], bool):
        errors.append("is_active must be a boolean")

    ref = blueprint["reference_size"]
    if not isinstance(ref, dict):
        errors.append("reference_size must be an object")
    else:
        for key in ("width", "height"):
            if key not in ref:
                errors.append(f"reference_size missing '{key}'")
            elif not isinstance(ref[key], int) or ref[key] <= 0:
                errors.append(f"reference_size.{key} must be a positive integer")

    alignment = blueprint["alignment"]
    if not isinstance(alignment, dict):
        errors.append("alignment must be an object")
    else:
        if "enabled" in alignment and not isinstance(alignment["enabled"], bool):
            errors.append("alignment.enabled must be a boolean")
        if "mode" in alignment:
            if alignment["mode"] not in {"homography", "resize_only"}:
                errors.append("alignment.mode must be 'homography' or 'resize_only'")

    validation = blueprint["validation"]
    expected_validation_keys = [
        "usd_hard_min", "usd_hard_max",
        "usd_expected_min", "usd_expected_max",
        "syp_hard_min", "syp_hard_max",
        "syp_expected_min", "syp_expected_max",
        "min_18k_to_21k_ratio", "max_18k_to_21k_ratio",
    ]
    if not isinstance(validation, dict):
        errors.append("validation must be an object")
    else:
        for key in expected_validation_keys:
            if key not in validation:
                errors.append(f"validation missing '{key}'")
            elif not _is_number(validation[key]):
                errors.append(f"validation.{key} must be numeric")

        if all(k in validation and _is_number(validation[k]) for k in expected_validation_keys):
            if float(validation["usd_hard_min"]) >= float(validation["usd_hard_max"]):
                errors.append("validation usd_hard_min must be less than usd_hard_max")
            if float(validation["usd_expected_min"]) >= float(validation["usd_expected_max"]):
                errors.append("validation usd_expected_min must be less than usd_expected_max")
            if float(validation["syp_hard_min"]) >= float(validation["syp_hard_max"]):
                errors.append("validation syp_hard_min must be less than syp_hard_max")
            if float(validation["syp_expected_min"]) >= float(validation["syp_expected_max"]):
                errors.append("validation syp_expected_min must be less than syp_expected_max")
            if float(validation["min_18k_to_21k_ratio"]) >= float(validation["max_18k_to_21k_ratio"]):
                errors.append("validation min_18k_to_21k_ratio must be less than max_18k_to_21k_ratio")

    fields = blueprint["fields"]
    if not isinstance(fields, list) or not fields:
        errors.append("fields must be a non-empty list")
        return BlueprintValidationResult(ok=False, errors=errors, warnings=warnings)

    seen_ids: set[str] = set()
    present_required_ids: set[str] = set()

    for index, field in enumerate(fields):
        if not isinstance(field, dict):
            errors.append(f"fields[{index}] must be an object")
            continue

        for key in ("id", "label", "type", "required", "box", "ocr_engines", "preprocess_modes"):
            if key not in field:
                errors.append(f"fields[{index}] missing '{key}'")

        field_id = field.get("id")
        if not isinstance(field_id, str) or not field_id.strip():
            errors.append(f"fields[{index}].id must be a non-empty string")
            continue

        if field_id in seen_ids:
            errors.append(f"duplicate field id '{field_id}'")
            continue
        seen_ids.add(field_id)

        if field_id in REQUIRED_FIELD_IDS:
            present_required_ids.add(field_id)

        if not isinstance(field.get("label"), str) or not field["label"].strip():
            errors.append(f"field '{field_id}': label must be a non-empty string")

        if field.get("type") not in ALLOWED_FIELD_TYPES:
            errors.append(
                f"field '{field_id}': invalid type '{field.get('type')}', allowed={sorted(ALLOWED_FIELD_TYPES)}"
            )

        if not isinstance(field.get("required"), bool):
            errors.append(f"field '{field_id}': required must be boolean")

        _validate_box(field_id, field.get("box"), errors)
        _validate_string_list(field_id, "ocr_engines", field.get("ocr_engines"), ALLOWED_OCR_ENGINES, errors)
        _validate_string_list(field_id, "preprocess_modes", field.get("preprocess_modes"), ALLOWED_PREPROCESS_MODES, errors)

        if "psm" in field and field["psm"] is not None:
            if not isinstance(field["psm"], int) or field["psm"] < 3 or field["psm"] > 13:
                errors.append(f"field '{field_id}': psm must be an integer between 3 and 13")

        if "char_whitelist" in field and field["char_whitelist"] is not None:
            if not isinstance(field["char_whitelist"], str):
                errors.append(f"field '{field_id}': char_whitelist must be a string or null")

    missing_required_ids = REQUIRED_FIELD_IDS - present_required_ids
    if missing_required_ids:
        errors.append(f"missing required field ids: {sorted(missing_required_ids)}")

    for field in fields:
        if not isinstance(field, dict):
            continue
        field_id = field.get("id", "?")
        box = field.get("box")
        if isinstance(box, dict) and all(k in box and _is_number(box[k]) for k in ("x1", "y1", "x2", "y2")):
            area = (float(box["x2"]) - float(box["x1"])) * (float(box["y2"]) - float(box["y1"]))
            if area < 0.002:
                warnings.append(f"field '{field_id}' has a very small crop area")
            if area > 0.20:
                warnings.append(f"field '{field_id}' has a very large crop area")

    return BlueprintValidationResult(
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
