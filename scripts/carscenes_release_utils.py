#!/usr/bin/env python3
from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from carscenes_annotation_taxonomy import (
    FIELD_PATH_TO_TAXONOMY_PATH,
    load_annotation_taxonomy,
    normalize_list_values,
    normalize_record_to_taxonomy,
    normalize_scalar_value,
    taxonomy_options_for_field,
)


SOURCE_NAMES = ("Argoverse1", "Cityscapes", "KITTI", "nuScenes")
TAXONOMY = load_annotation_taxonomy()
BENCHMARK_LIST_DEFAULTS = {
    "LaneInformation.SpecialLanes": ["NoSpecialLanes"],
    "TrafficSigns.TrafficSignsTypes": ["NoTrafficSigns"],
    "Vehicles.InMotion": [],
    "Vehicles.States": [],
    "Vehicles.VehicleTypes": [],
    "Pedestrians": ["NoPed"],
    "Visibility.SpecificImpairments": ["NoImpairments"],
}
WRAPPER_FILES = {"split.py", "unifiedAnalysis.json"}

SCALAR_MAPS = {
    "TimeOfDay": {
        "Evening": "Dusk/Dawn",
        "Overcast": "Unknown",
    },
    "Weather": {
        "NoWeatherCondition": "Unknown",
    },
    "Directionality": {
        "Two-Way": "Two-Way",
        "One-Way": "One-Way",
        "Multi-directional": "Multi-directional",
        "Multi-directional with median": "Multi-directional with median",
    },
    "CameraCondition": {
        "Camera glare": "Glare",
        "Camera Glare": "Glare",
    },
    "Visibility.General": {
        "Impaired": "Reduced",
    },
    "LaneInformation.NumberOfLanes": {
        "Single": "One",
        "1": "One",
        "Two": "Two",
        "2": "Two",
        "None": "NoLane",
        "Multiple": "MultipleLanes",
    },
    "LaneInformation.LaneMarkings": {
        "Lane Not Clearly Visible": "LaneNotClearlyVisible",
        "Not Visible": "NotVisible",
        "Visible": "LaneVisible",
        "Present": "LaneVisible",
        "Clearly Visible": "LaneVisible",
        "Partially Visible": "LaneNotClearlyVisible",
        "Faded": "LaneNotClearlyVisible",
    },
    "TrafficSigns.TrafficSignsVisibility": {
        "Clear": "Visible",
        "Good": "Visible",
        "NoTrafficSigns": "SignNotVisible",
        "Not Applicable": "SignNotVisible",
        "Not Indicated": "Unknown",
        "Not Clearly Visible": "NotClearlyVisible",
    },
    "TrafficSigns.TrafficLightState": {
        "In Transition(NoState)": "InTransition",
        "In Transition(TwoState)": "InTransition",
        "Red Left Arrow": "Red",
        "Not Visible": "Unknown",
        "None": "Unknown",
    },
    "Vehicles.TotalNumber": {
        "One": "Few",
        "2": "Few",
        "3": "Few",
        "Multiple": "MultipleVehicles",
        "Several": "MultipleVehicles",
    },
}

LIST_MAPS = {
    "Vehicles.InMotion": {
        "1": "True",
    },
    "Visibility.SpecificImpairments": {
        "Obstructed by Rain": "ObstructedByRain",
        "Camera glare": "CameraGlare",
        "Glare": "CameraGlare",
        "Sun Glare": "CameraGlare",
        "RainOnLens": "ObstructedByRain",
        "Shadows and Vegetation": "Shadows",
    },
    "Vehicles.VehicleTypes": {
        "Cars": "Car",
        "Vans": "Van",
        "Commerical Truck": "Commercial Truck",
        "Commercial Vehicles": "Commercial Vehicle",
        "Commercial Truck": "Commercial Vehicle",
        "  Sedan": "Sedan",
    },
    "Pedestrians": {
        "One": "Few",
        "Single": "Few",
        "Not Present": "NoPed",
    },
    "Bicyclist.Presence": {
        "BicyclistPresent": "True",
        "Visible on Road": "True",
    },
    "Bicyclist.Location": {
        "Sidewalk": "On Sidewalk",
        "Crossing Street": "Crossing Road",
        "Crosswalk": "In Crosswalk",
        "Crossing": "Crossing Road",
    },
}


@dataclass
class ReleaseArtifacts:
    records: list[dict[str, Any]]
    issues: dict[str, Any]
    schema: dict[str, Any]
    splits: dict[str, list[str]]


def _clean_text(value: Any) -> str:
    text = str(value).replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_score(value: str, seed: int = 42) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _path_get(obj: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _normalize_scalar(path: str, value: Any) -> Any:
    return normalize_scalar_value(path, value, TAXONOMY)


def _normalize_list(path: str, values: Iterable[Any]) -> list[Any]:
    normalized = normalize_list_values(path, list(values) if isinstance(values, (list, tuple, set)) else values, TAXONOMY)
    return normalized or copy.deepcopy(BENCHMARK_LIST_DEFAULTS.get(path, []))


def _normalize_special_lanes(values: Iterable[Any], raw_lane_info: dict[str, Any]) -> list[str]:
    base_values = list(values) if isinstance(values, (list, tuple, set)) else [values]
    if raw_lane_info.get("RoadWork") == "True":
        base_values.append("Road Work")
    if raw_lane_info.get("TrafficCones") == "True":
        base_values.append("Traffic Cones Blocking Parts of the Road")
    if raw_lane_info.get("ConstructionZone") == "True":
        base_values.append("Construction Barriers")
    if raw_lane_info.get("TemporaryMarkings") == "True":
        base_values.append("Construction Cones")
    normalized = _normalize_list("LaneInformation.SpecialLanes", base_values)
    return normalized or ["NoSpecialLanes"]


def _normalize_vehicle_types(raw: dict[str, Any]) -> list[str]:
    traffic_signs = raw.get("TrafficSigns", {}) if isinstance(raw.get("TrafficSigns"), dict) else {}
    vehicles = raw.get("Vehicles", {}) if isinstance(raw.get("Vehicles"), dict) else {}
    values: list[Any] = []
    if isinstance(traffic_signs.get("VehicleTypes"), list):
        values.extend(traffic_signs["VehicleTypes"])
    if isinstance(vehicles.get("VehicleTypes"), list):
        values.extend(vehicles["VehicleTypes"])
    return _normalize_list("Vehicles.VehicleTypes", values or [])


def _normalize_visibility_specific(raw_visibility: dict[str, Any]) -> list[str]:
    values = raw_visibility.get("SpecificImpairments")
    if not isinstance(values, list):
        values = ["NoImpairments"]
    return _normalize_list("Visibility.SpecificImpairments", values)


def _normalize_pedestrians(raw: dict[str, Any]) -> list[str]:
    value = raw.get("Pedestrians")
    if isinstance(value, list):
        return _normalize_list("Pedestrians", value)
    if value is None:
        return ["NoPed"]
    return _normalize_list("Pedestrians", [value])


def _normalize_bicyclist(raw: dict[str, Any]) -> dict[str, Any] | None:
    bicyclist = raw.get("Bicyclist")
    if not isinstance(bicyclist, dict):
        return None
    presence = _normalize_scalar("Bicyclist.Presence", bicyclist.get("Presence", "False"))
    location = bicyclist.get("Location")
    position = bicyclist.get("Position")
    if location is None and position is not None:
        location = position
    normalized: dict[str, Any] = {"Presence": presence}
    if location is not None:
        normalized["Location"] = _normalize_scalar("Bicyclist.Location", location)
    return normalized


def _normalize_traffic_light_state(raw: dict[str, Any]) -> str | None:
    traffic_signs = raw.get("TrafficSigns", {}) if isinstance(raw.get("TrafficSigns"), dict) else {}
    if "TrafficLightState" in raw:
        return _normalize_scalar("TrafficSigns.TrafficLightState", raw.get("TrafficLightState"))
    if isinstance(raw.get("TrafficLights"), dict):
        lights = raw["TrafficLights"]
        if "State" in lights:
            return _normalize_scalar("TrafficSigns.TrafficLightState", lights.get("State"))
        if "TrafficLightState" in lights:
            return _normalize_scalar("TrafficSigns.TrafficLightState", lights.get("TrafficLightState"))
    if "TrafficLightState" in traffic_signs:
        return _normalize_scalar("TrafficSigns.TrafficLightState", traffic_signs.get("TrafficLightState"))
    return None


def canonicalize_raw_label(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    lane_info = raw.get("LaneInformation", {}) if isinstance(raw.get("LaneInformation"), dict) else {}
    traffic_signs = raw.get("TrafficSigns", {}) if isinstance(raw.get("TrafficSigns"), dict) else {}
    vehicles = raw.get("Vehicles", {}) if isinstance(raw.get("Vehicles"), dict) else {}
    ego = raw.get("Ego-Vehicle", {}) if isinstance(raw.get("Ego-Vehicle"), dict) else {}
    visibility = raw.get("Visibility", {}) if isinstance(raw.get("Visibility"), dict) else {}

    traffic_light_state = _normalize_traffic_light_state(raw)
    if "VehicleTypes" in traffic_signs and "VehicleTypes" not in vehicles:
        warnings.append("moved TrafficSigns.VehicleTypes to Vehicles.VehicleTypes")

    record = {
        "Scene": raw.get("Scene"),
        "TimeOfDay": raw.get("TimeOfDay"),
        "Weather": raw.get("Weather"),
        "RoadConditions": raw.get("RoadConditions"),
        "LaneInformation": {
            "NumberOfLanes": lane_info.get("NumberOfLanes"),
            "LaneMarkings": lane_info.get("LaneMarkings"),
            "SpecialLanes": _normalize_special_lanes(lane_info.get("SpecialLanes", []), lane_info),
        },
        "TrafficSigns": {
            "TrafficSignsTypes": traffic_signs.get("TrafficSignsTypes"),
            "TrafficSignsVisibility": traffic_signs.get("TrafficSignsVisibility"),
        },
        "Vehicles": {
            "TotalNumber": vehicles.get("TotalNumber"),
            "VehicleTypes": _normalize_vehicle_types(raw),
            "InMotion": vehicles.get("InMotion", []),
            "States": vehicles.get("States", []),
        },
        "Pedestrians": raw.get("Pedestrians"),
        "Directionality": raw.get("Directionality"),
        "Visibility": {
            "General": visibility.get("General"),
            "SpecificImpairments": visibility.get("SpecificImpairments"),
        },
        "Ego-Vehicle": {
            "Direction": ego.get("Direction", ego.get("EgoDirection")),
            "Maneuver": ego.get("Maneuver", ego.get("EgoManeuver")),
        },
        "CameraCondition": raw.get("CameraCondition"),
        "Severity": raw.get("Severity", 0),
    }

    if traffic_light_state is not None:
        record["TrafficSigns"]["TrafficLightState"] = traffic_light_state

    bicyclist = _normalize_bicyclist(raw)
    if bicyclist is not None:
        record["_auxiliary"] = {"Bicyclist": bicyclist}
    if isinstance(raw.get("Animals"), dict):
        record["Animals"] = raw["Animals"]
    if isinstance(raw.get("Policeman"), dict):
        record["Policeman"] = raw["Policeman"]

    if not isinstance(record["Severity"], int):
        try:
            record["Severity"] = int(record["Severity"])
        except Exception:
            warnings.append("severity_coerced_to_zero")
            record["Severity"] = 0

    record = normalize_record_to_taxonomy(record, TAXONOMY)
    auxiliary: dict[str, Any] = {}
    for key in ("Bicyclist", "Animals", "Policeman"):
        value = record.pop(key, None)
        if isinstance(value, dict):
            auxiliary[key] = value
    if auxiliary:
        record["_auxiliary"] = auxiliary

    return record, warnings


def _iter_image_files(dataset_root: Path) -> dict[str, dict[str, Path]]:
    image_index: dict[str, dict[str, Path]] = {}
    for source in SOURCE_NAMES:
        source_root = dataset_root / "train" / "images" / source
        image_index[source] = {path.name: path for path in source_root.rglob("*") if path.is_file()}
    return image_index


def _iter_label_files(dataset_root: Path) -> dict[str, dict[str, Path]]:
    label_index: dict[str, dict[str, Path]] = {}
    for source in SOURCE_NAMES:
        source_root = dataset_root / "train" / "labels" / source
        per_source: dict[str, Path] = {}
        for path in source_root.rglob("*.json"):
            if path.name in WRAPPER_FILES:
                continue
            per_source[path.name[:-5]] = path
        label_index[source] = per_source
    return label_index


def _source_target_sizes() -> dict[str, dict[str, int]]:
    return {
        "Argoverse1": {"gold": 25, "silver": 125},
        "Cityscapes": {"gold": 25, "silver": 125},
        "KITTI": {"gold": 25, "silver": 125},
        "nuScenes": {"gold": 25, "silver": 125},
    }


def load_canonical_records(dataset_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_index = _iter_image_files(dataset_root)
    label_index = _iter_label_files(dataset_root)

    records: list[dict[str, Any]] = []
    issues: dict[str, Any] = {
        "missing_labels": [],
        "extra_labels": [],
        "wrapper_files": [],
        "normalization_warnings": [],
        "source_summary": {},
    }

    for source in SOURCE_NAMES:
        images = image_index[source]
        labels = label_index[source]
        missing = sorted(set(images) - set(labels))
        extra = sorted(set(labels) - set(images))
        issues["missing_labels"].extend(
            {"source": source, "image": name} for name in missing
        )
        issues["extra_labels"].extend(
            {"source": source, "label": name} for name in extra
        )
        issues["source_summary"][source] = {
            "images": len(images),
            "labels": len(labels),
            "missing_labels": len(missing),
            "extra_labels": len(extra),
        }

        for name in sorted(set(images) & set(labels)):
            image_path = images[name]
            label_path = labels[name]
            raw = json.loads(label_path.read_text(encoding="utf-8"))
            canonical, warnings = canonicalize_raw_label(raw)
            record_id = f"carscenes-v1:{source}:{name}"
            canonical.update(
                {
                    "_id": record_id,
                    "_source": source,
                    "_image_relpath": str(image_path.relative_to(dataset_root)),
                    "_raw_label_relpath": str(label_path.relative_to(dataset_root)),
                    "_record_hash": _stable_hash(record_id),
                }
            )
            if warnings:
                issues["normalization_warnings"].append(
                    {"id": record_id, "warnings": warnings}
                )
            records.append(canonical)

    return sorted(records, key=lambda item: item["_id"]), issues


def _severity_bucket(severity: int) -> str:
    if severity <= 3:
        return "1-3"
    if severity <= 6:
        return "4-6"
    return "7-10"


def has_vru(record: dict[str, Any]) -> bool:
    if any(item != "NoPed" for item in record.get("Pedestrians", [])):
        return True
    bicyclist = _path_get(record, "_auxiliary.Bicyclist.Presence")
    return bicyclist == "True"


def has_traffic_light(record: dict[str, Any]) -> bool:
    value = _path_get(record, "TrafficSigns.TrafficLightState")
    return value not in {None, "Unknown"}


def adverse_conditions(record: dict[str, Any]) -> bool:
    if record.get("TimeOfDay") in {"Nighttime", "Dusk/Dawn"}:
        return True
    if record.get("Visibility", {}).get("General") != "Good":
        return True
    impairments = set(record.get("Visibility", {}).get("SpecificImpairments", []))
    if impairments and impairments != {"NoImpairments"}:
        return True
    return record.get("Weather") in {"Rainy", "Cloudy", "Overcast", "Unknown"}


def scenario_score(record: dict[str, Any], seed: int = 42) -> tuple[float, float]:
    score = 0.0
    score += 1.0 if adverse_conditions(record) else 0.0
    score += 1.0 if has_vru(record) else 0.0
    score += 1.0 if has_traffic_light(record) else 0.0
    score += {"1-3": 0.0, "4-6": 0.4, "7-10": 0.8}[_severity_bucket(record["Severity"])]
    return (score, _stable_score(record["_id"], seed))


def _sample_bucket(records: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    ranked = sorted(records, key=lambda item: scenario_score(item, seed), reverse=True)
    return ranked[:target]


def build_splits(records: list[dict[str, Any]], seed: int = 42) -> dict[str, list[str]]:
    targets = _source_target_sizes()
    by_source: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        by_source[record["_source"]].append(record)

    gold_ids: list[str] = []
    silver_ids: list[str] = []
    agreement_ids: list[str] = []

    for source, source_records in by_source.items():
        severity_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for record in source_records:
            severity_groups[_severity_bucket(record["Severity"])].append(record)

        gold_quota = targets[source]["gold"]
        planned = {"1-3": max(1, gold_quota // 3), "4-6": max(1, math.ceil(gold_quota / 3)), "7-10": max(1, gold_quota // 4)}
        selected: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        for bucket in ("7-10", "4-6", "1-3"):
            bucket_records = [item for item in severity_groups[bucket] if item["_id"] not in used_ids]
            take = min(planned[bucket], len(bucket_records))
            sampled = _sample_bucket(bucket_records, take, seed)
            selected.extend(sampled)
            used_ids.update(item["_id"] for item in sampled)
        remaining = [item for item in source_records if item["_id"] not in used_ids]
        if len(selected) < gold_quota:
            top_up = _sample_bucket(remaining, gold_quota - len(selected), seed)
            selected.extend(top_up)
            used_ids.update(item["_id"] for item in top_up)
        selected = sorted(selected[:gold_quota], key=lambda item: item["_id"])
        gold_ids.extend(item["_id"] for item in selected)

        remaining_after_gold = [item for item in source_records if item["_id"] not in used_ids]
        silver_ranked = sorted(
            remaining_after_gold,
            key=lambda item: scenario_score(item, seed),
            reverse=True,
        )
        silver_choice = sorted(
            silver_ranked[: targets[source]["silver"]],
            key=lambda item: item["_id"],
        )
        silver_ids.extend(item["_id"] for item in silver_choice)

        for bucket in ("7-10", "4-6", "1-3"):
            bucket_gold = [item for item in selected if _severity_bucket(item["Severity"]) == bucket]
            agreement_ids.extend(item["_id"] for item in sorted(bucket_gold, key=lambda item: item["_id"])[:2])

    agreement_ids = sorted(set(agreement_ids))[:25]
    if len(agreement_ids) < 25:
        filler = [
            record["_id"]
            for record in sorted(records, key=lambda item: scenario_score(item, seed), reverse=True)
            if record["_id"] in set(gold_ids) and record["_id"] not in set(agreement_ids)
        ]
        agreement_ids.extend(filler[: 25 - len(agreement_ids)])
    agreement_ids = sorted(set(agreement_ids))[:25]
    taken = set(gold_ids) | set(silver_ids)
    train_ids = sorted(record["_id"] for record in records if record["_id"] not in taken)

    return {
        "train": sorted(gold_ids and train_ids or train_ids),
        "silver-dev-500": sorted(silver_ids),
        "gold-test-100": sorted(gold_ids),
        "gold-agreement-25": agreement_ids,
    }


def _collect_field_enums(records: list[dict[str, Any]]) -> dict[str, set[Any]]:
    enums: dict[str, set[Any]] = collections.defaultdict(set)

    def visit(prefix: str, value: Any) -> None:
        if prefix.startswith("_"):
            return
        if isinstance(value, dict):
            for key, inner in value.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                visit(new_prefix, inner)
            return
        if isinstance(value, list):
            for item in value:
                enums[prefix].add(item)
            return
        enums[prefix].add(value)

    for record in records:
        visit("", record)
    return enums


def build_schema(records: list[dict[str, Any]]) -> dict[str, Any]:
    enums = _collect_field_enums(records)

    def enum(path: str) -> list[Any]:
        if path in FIELD_PATH_TO_TAXONOMY_PATH:
            taxonomy_values = taxonomy_options_for_field(path, TAXONOMY)
            if taxonomy_values:
                return taxonomy_values
        values = enums.get(path, set())
        return sorted(values, key=lambda item: str(item))

    return {
        "name": "carscenes-v1",
        "version": "1.0.0",
        "required_top_level": [
            "_id",
            "_source",
            "_image_relpath",
            "_raw_label_relpath",
            "_record_hash",
            "Scene",
            "TimeOfDay",
            "Weather",
            "RoadConditions",
            "LaneInformation",
            "TrafficSigns",
            "Vehicles",
            "Pedestrians",
            "Directionality",
            "Visibility",
            "Ego-Vehicle",
            "CameraCondition",
            "Severity",
        ],
        "properties": {
            "Scene": {"type": "string", "enum": enum("Scene")},
            "TimeOfDay": {"type": "string", "enum": enum("TimeOfDay")},
            "Weather": {"type": "string", "enum": enum("Weather")},
            "RoadConditions": {"type": "string", "enum": enum("RoadConditions")},
            "Directionality": {"type": "string", "enum": enum("Directionality")},
            "CameraCondition": {"type": "string", "enum": enum("CameraCondition")},
            "Severity": {"type": "integer", "minimum": 0, "maximum": 10},
            "LaneInformation": {
                "type": "object",
                "required": ["NumberOfLanes", "LaneMarkings", "SpecialLanes"],
                "properties": {
                    "NumberOfLanes": {"type": "string", "enum": enum("LaneInformation.NumberOfLanes")},
                    "LaneMarkings": {"type": "string", "enum": enum("LaneInformation.LaneMarkings")},
                    "SpecialLanes": {"type": "array", "items": {"type": "string", "enum": enum("LaneInformation.SpecialLanes")}},
                },
            },
            "TrafficSigns": {
                "type": "object",
                "required": ["TrafficSignsTypes", "TrafficSignsVisibility"],
                "properties": {
                    "TrafficSignsTypes": {"type": "array", "items": {"type": "string", "enum": enum("TrafficSigns.TrafficSignsTypes")}},
                    "TrafficSignsVisibility": {"type": "string", "enum": enum("TrafficSigns.TrafficSignsVisibility")},
                    "TrafficLightState": {"type": "string", "enum": enum("TrafficSigns.TrafficLightState")},
                },
            },
            "Vehicles": {
                "type": "object",
                "required": ["TotalNumber", "VehicleTypes", "InMotion", "States"],
                "properties": {
                    "TotalNumber": {"type": "string", "enum": enum("Vehicles.TotalNumber")},
                    "VehicleTypes": {"type": "array", "items": {"type": "string", "enum": enum("Vehicles.VehicleTypes")}},
                    "InMotion": {"type": "array", "items": {"type": "string", "enum": enum("Vehicles.InMotion")}},
                    "States": {"type": "array", "items": {"type": "string", "enum": enum("Vehicles.States")}},
                },
            },
            "Pedestrians": {"type": "array", "items": {"type": "string", "enum": enum("Pedestrians")}},
            "Visibility": {
                "type": "object",
                "required": ["General", "SpecificImpairments"],
                "properties": {
                    "General": {"type": "string", "enum": enum("Visibility.General")},
                    "SpecificImpairments": {"type": "array", "items": {"type": "string", "enum": enum("Visibility.SpecificImpairments")}},
                },
            },
            "Ego-Vehicle": {
                "type": "object",
                "required": ["Direction", "Maneuver"],
                "properties": {
                    "Direction": {"type": "string", "enum": enum("Ego-Vehicle.Direction")},
                    "Maneuver": {"type": "string", "enum": enum("Ego-Vehicle.Maneuver")},
                },
            },
        },
        "benchmark_fields": {
            "scalar": [
                "Scene",
                "TimeOfDay",
                "Weather",
                "RoadConditions",
                "Directionality",
                "CameraCondition",
                "Visibility.General",
                "LaneInformation.NumberOfLanes",
                "LaneInformation.LaneMarkings",
                "TrafficSigns.TrafficSignsVisibility",
                "TrafficSigns.TrafficLightState",
                "Vehicles.TotalNumber",
                "Ego-Vehicle.Direction",
                "Ego-Vehicle.Maneuver",
                "Severity",
            ],
            "list": [
                "LaneInformation.SpecialLanes",
                "TrafficSigns.TrafficSignsTypes",
                "Vehicles.VehicleTypes",
                "Vehicles.InMotion",
                "Vehicles.States",
                "Pedestrians",
                "Visibility.SpecificImpairments",
            ],
        },
    }


def validate_record(record: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in schema["required_top_level"]:
        if key not in record:
            errors.append(f"missing_top_level:{key}")

    def validate_value(prefix: str, value: Any, spec: dict[str, Any]) -> None:
        if spec["type"] == "string":
            if not isinstance(value, str):
                errors.append(f"type:{prefix}:expected_string")
                return
            if spec.get("enum") and value not in spec["enum"]:
                errors.append(f"enum:{prefix}:{value}")
        elif spec["type"] == "integer":
            if not isinstance(value, int):
                errors.append(f"type:{prefix}:expected_integer")
                return
            if value < spec.get("minimum", value) or value > spec.get("maximum", value):
                errors.append(f"range:{prefix}:{value}")
        elif spec["type"] == "array":
            if not isinstance(value, list):
                errors.append(f"type:{prefix}:expected_array")
                return
            for item in value:
                validate_value(prefix, item, spec["items"])
        elif spec["type"] == "object":
            if not isinstance(value, dict):
                errors.append(f"type:{prefix}:expected_object")
                return
            for required_key in spec.get("required", []):
                if required_key not in value:
                    errors.append(f"missing:{prefix}.{required_key}")
            for key, inner_spec in spec.get("properties", {}).items():
                if key in value:
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    validate_value(new_prefix, value[key], inner_spec)

    for key, spec in schema["properties"].items():
        if key in record:
            validate_value(key, record[key], spec)
    return errors


def summarize_records(
    records: list[dict[str, Any]],
    issues: dict[str, Any],
    splits: dict[str, list[str]],
) -> dict[str, Any]:
    by_source = collections.Counter(record["_source"] for record in records)
    severity = collections.Counter(_severity_bucket(record["Severity"]) for record in records)
    split_counts = {name: len(values) for name, values in splits.items()}
    return {
        "release": "carscenes-v1",
        "records": len(records),
        "source_counts": dict(by_source),
        "severity_bucket_counts": dict(severity),
        "split_counts": split_counts,
        "issues": {
            "missing_labels": len(issues["missing_labels"]),
            "extra_labels": len(issues["extra_labels"]),
            "normalization_actions": len(issues["normalization_warnings"]),
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_release(dataset_root: Path, seed: int = 42) -> ReleaseArtifacts:
    records, issues = load_canonical_records(dataset_root)
    splits = build_splits(records, seed=seed)
    schema = build_schema(records)
    return ReleaseArtifacts(records=records, issues=issues, schema=schema, splits=splits)
