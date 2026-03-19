#!/usr/bin/env python3
from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_TAXONOMY_PATH = "configs/carscenes_annotation_taxonomy.json"

FIELD_PATH_TO_TAXONOMY_PATH = {
    "Scene": "Scene",
    "TimeOfDay": "TimeOfDay",
    "Weather": "Weather",
    "RoadConditions": "RoadConditions",
    "LaneInformation.NumberOfLanes": "LaneInformation.NumberOfLanes",
    "LaneInformation.LaneMarkings": "LaneInformation.LaneMarkings",
    "LaneInformation.SpecialLanes": "LaneInformation.SpecialLanes",
    "TrafficSigns.TrafficSignsTypes": "TrafficSigns.Types",
    "TrafficSigns.TrafficSignsVisibility": "TrafficSigns.Visibility",
    "TrafficSigns.TrafficLightState": "TrafficSigns.TrafficLightState",
    "Vehicles.TotalNumber": "Vehicles.TotalNumber",
    "Vehicles.InMotion": "Vehicles.InMotion",
    "Vehicles.VehicleTypes": "Vehicles.VehicleTypes",
    "Vehicles.States": "Vehicles.States",
    "Pedestrians": "Pedestrians",
    "Policeman.Presence": "Policeman.Presence",
    "Policeman.States": "Policeman.States",
    "Bicyclist.Presence": "Bicyclist.Presence",
    "Bicyclist.Location": "Bicyclist.Location",
    "Animals.Type": "Animals.Type",
    "Animals.State": "Animals.State",
    "Directionality": "Directionality",
    "Ego-Vehicle.Direction": "Ego-Vehicle.Direction",
    "Ego-Vehicle.Maneuver": "Ego-Vehicle.Maneuver",
    "Visibility.General": "Visibility.General",
    "Visibility.SpecificImpairments": "Visibility.SpecificImpairments",
    "CameraCondition": "CameraCondition",
    "Severity": "Severity",
}

SCALAR_FIELDS = {
    "Scene",
    "TimeOfDay",
    "Weather",
    "RoadConditions",
    "LaneInformation.NumberOfLanes",
    "LaneInformation.LaneMarkings",
    "TrafficSigns.TrafficSignsVisibility",
    "TrafficSigns.TrafficLightState",
    "Vehicles.TotalNumber",
    "Policeman.Presence",
    "Bicyclist.Presence",
    "Bicyclist.Location",
    "Animals.Type",
    "Animals.State",
    "Directionality",
    "Ego-Vehicle.Direction",
    "Ego-Vehicle.Maneuver",
    "Visibility.General",
    "CameraCondition",
    "Severity",
}

LIST_FIELDS = {
    "LaneInformation.SpecialLanes",
    "TrafficSigns.TrafficSignsTypes",
    "Vehicles.InMotion",
    "Vehicles.VehicleTypes",
    "Vehicles.States",
    "Pedestrians",
    "Policeman.States",
    "Visibility.SpecificImpairments",
}

LIST_DEFAULTS = {
    "LaneInformation.SpecialLanes": ["NoSpecialLanes"],
    "TrafficSigns.TrafficSignsTypes": ["NoTrafficSigns"],
    "Vehicles.InMotion": [],
    "Vehicles.VehicleTypes": [],
    "Vehicles.States": [],
    "Pedestrians": ["NoPed"],
    "Policeman.States": ["NoPolicemanState"],
    "Visibility.SpecificImpairments": ["NoImpairments"],
}

SCALAR_DEFAULTS = {
    "Scene": "Unknown",
    "TimeOfDay": "Daytime",
    "Weather": "Not Indicated",
    "RoadConditions": "Dry",
    "LaneInformation.NumberOfLanes": "NoLane",
    "LaneInformation.LaneMarkings": "LaneNotVisible",
    "TrafficSigns.TrafficSignsVisibility": "SignNotIndicated",
    "Vehicles.TotalNumber": "NoVehicle",
    "Policeman.Presence": "False",
    "Bicyclist.Presence": "False",
    "Bicyclist.Location": "NoBicyclistLocation",
    "Animals.Type": "NoAnimal",
    "Animals.State": "NoAnimalState",
    "Directionality": "Unknown",
    "Ego-Vehicle.Direction": "EgoStopped",
    "Ego-Vehicle.Maneuver": "EgoWaiting",
    "Visibility.General": "Good",
    "CameraCondition": "Clear",
}

OPTIONAL_SCALAR_FIELDS = {"TrafficSigns.TrafficLightState"}

SCENE_PRIORITY = [
    "Intersection",
    "School Zone",
    "Construction",
    "Parking Area",
    "Under Bridge",
    "Highway",
    "Residential",
    "Commercial",
    "Industrial",
    "Rural",
    "Suburban",
    "Urban",
    "Unknown",
]


def _normalized_key(value: Any) -> str:
    text = str(value).replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[\[\]\(\),/_-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _dedupe_preserve_order(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    output: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _build_alias_map(pairs: dict[str, str | None]) -> dict[str, str | None]:
    return {_normalized_key(key): value for key, value in pairs.items()}


SCALAR_ALIASES = {
    "Scene": _build_alias_map(
        {
            "Bridge": "Under Bridge",
            "Roundabout": "Intersection",
            "Narrow Residential Road": "Residential",
            "Pedestrian Zone": "Commercial",
        }
    ),
    "TimeOfDay": _build_alias_map(
        {
            "Evening": "Dusk/Dawn",
            "Overcast": "Daytime",
            "Night": "Nighttime",
            "Day": "Daytime",
            "Unknown": "Daytime",
        }
    ),
    "Weather": _build_alias_map(
        {
            "Unknown": "Not Indicated",
            "NoWeatherCondition": "Not Indicated",
        }
    ),
    "RoadConditions": _build_alias_map(
        {
            "Unknown": "Dry",
        }
    ),
    "LaneInformation.NumberOfLanes": _build_alias_map(
        {
            "Single": "1",
            "One": "1",
            "Two": "2",
            "Three": "3",
            "Four": "4",
            "Five": "5",
            "Multiple": "MultipleLanes",
            "None": "NoLane",
        }
    ),
    "LaneInformation.LaneMarkings": _build_alias_map(
        {
            "Visible": "LaneVisible",
            "Present": "LaneVisible",
            "Clearly Visible": "LaneVisible",
            "Not Visible": "LaneNotVisible",
            "Lane Not Visible": "LaneNotVisible",
            "NotVisible": "LaneNotVisible",
            "Not Visible / Unknown": "LaneNotVisible",
            "Lane Not Clearly Visible": "LaneNotClearlyVisible",
            "Not Clearly Visible": "LaneNotClearlyVisible",
            "Partially Visible": "LaneNotClearlyVisible",
            "Faded": "LaneNotClearlyVisible",
        }
    ),
    "TrafficSigns.TrafficSignsVisibility": _build_alias_map(
        {
            "Visible": "SignVisible",
            "Clear": "SignVisible",
            "Good": "SignVisible",
            "Not Clearly Visible": "SignNotClearlyVisible",
            "NotClearlyVisible": "SignNotClearlyVisible",
            "NoTrafficSigns": "SignNotVisible",
            "Not Applicable": "SignNotVisible",
            "Unknown": "SignNotIndicated",
            "Not Indicated": "SignNotIndicated",
        }
    ),
    "TrafficSigns.TrafficLightState": _build_alias_map(
        {
            "InTransition": "In Transition(NoState)",
            "Red Left Arrow": "Red",
            "Unknown": None,
            "None": None,
            "Not Visible": None,
        }
    ),
    "Vehicles.TotalNumber": _build_alias_map(
        {
            "Single": "One",
            "1": "One",
            "2": "Few",
            "3": "Few",
            "Multiple": "MultipleVehicles",
            "Several": "MultipleVehicles",
        }
    ),
    "Policeman.Presence": _build_alias_map({"Present": "True", "Not Present": "False"}),
    "Bicyclist.Presence": _build_alias_map(
        {"BicyclistPresent": "True", "Visible on Road": "True", "Present": "True", "Not Present": "False"}
    ),
    "Bicyclist.Location": _build_alias_map(
        {
            "Sidewalk": "On Sidewalk",
            "Crossing Street": "Crossing Road",
            "Crosswalk": "Crossing Road",
            "Crossing": "Crossing Road",
            "No Location": "NoBicyclistLocation",
        }
    ),
    "Animals.Type": _build_alias_map({"None": "NoAnimal", "NoAnimalType": "NoAnimal"}),
    "Animals.State": _build_alias_map({"None": "NoAnimalState", "NoState": "NoAnimalState"}),
    "Directionality": _build_alias_map(
        {
            "Multi-directional": "Multi-directional with median",
            "Multidirectional": "Multi-directional with median",
        }
    ),
    "Ego-Vehicle.Direction": _build_alias_map(
        {
            "Forward": "EgoForward",
            "Stopped": "EgoStopped",
            "Making a Left Turn": "EgoMaking a Left Turn",
            "Making a Right Turn": "EgoMaking a Right Turn",
            "Stopped at Intersection": "EgoStopped at Intersection",
            "Stopped at Stop Sign": "EgoStopped at Stop Sign",
            "Approaching Intersection": "EgoApproaching Intersection",
            "Approaching Roundabout": "EgoApproaching Roundabout",
            "Entering Roundabout": "EgoEntering Roundabout",
            "Exiting Roundabout": "EgoExiting Roundabout",
            "Exiting Highway": "EgoExiting Highway",
            "Parked": "EgoParked",
            "Right": "EgoMaking a Right Turn",
            "Left": "EgoMaking a Left Turn",
            "EgoMoving": "EgoForward",
            "EgoRight": "EgoMaking a Right Turn",
            "EgoLeft": "EgoMaking a Left Turn",
            "EgoStopped at Traffic Light": "EgoStopped at Intersection",
            "EgoStopped at Crosswalk": "EgoStopped",
            "EgoMerging Left": "EgoMerging",
            "EgoMerging Right": "EgoMerging",
            "EgoIn Queue": "EgoStopped",
            "EgoTurning": "EgoForward",
            "EgoStopped in Parking Space": "EgoParked",
            "EgoUnknown": None,
            "Unknown": None,
        }
    ),
    "Ego-Vehicle.Maneuver": _build_alias_map(
        {
            "Moving": "EgoMoving",
            "Waiting": "EgoWaiting",
            "Turning Right": "EgoTurning Right",
            "Turning right": "EgoTurning Right",
            "Turning Left": "EgoTurning Left",
            "Turning left": "EgoTurning Left",
            "Turning": "EgoTurning",
            "In Queue": "EgoIn Queue",
            "Approaching Intersection": "EgoMoving",
            "Stopped at Intersection": "EgoFullStopped",
            "Stopped": "EgoFullStopped",
            "Stopping": "EgoSlowing Down",
            "Following": "EgoFollowing",
            "Yielding": "EgoYielding",
            "Slowing Down": "EgoSlowing Down",
            "EgoSlowing": "EgoSlowing Down",
            "EgoProceeding": "EgoProceeding through Intersection",
            "EgoMovingForward": "EgoMoving",
            "NoManeuver": "EgoWaiting",
            "EgoStopped": "EgoFullStopped",
            "EgoStopped at Traffic Light": "EgoFullStopped",
            "EgoStopped at Intersection": "EgoFullStopped",
            "EgoApproaching Intersection": "EgoMoving",
            "EgoTurningRight": "EgoTurning Right",
            "EgoTurningLeft": "EgoTurning Left",
            "EgoUnknown": None,
            "Unknown": None,
        }
    ),
    "Visibility.General": _build_alias_map(
        {
            "Impaired": "Reduced",
            "Unknown": "Good",
        }
    ),
    "CameraCondition": _build_alias_map(
        {
            "Camera glare": "Glare",
            "Camera Glare": "Glare",
            "Overexposed": "Glare",
            "Unknown": "Clear",
        }
    ),
}

LIST_ALIASES = {
    "LaneInformation.SpecialLanes": _build_alias_map(
        {
            "Bus Lane": "Taxi and Bus Lane",
            "Tram Lane": "Tram Tracks",
            "Tram Track": "Tram Tracks",
            "Exit Lane": "Forward Only Lane",
            "Turn Lane": "Center Lane",
            "Pedestrian Walkway": "Crosswalk",
            "None": "NoSpecialLanes",
            "No Lane": "NoSpecialLanes",
            "Divided": None,
            "Exit Sign": None,
        }
    ),
    "TrafficSigns.TrafficSignsTypes": _build_alias_map(
        {
            "NoSign": "NoTrafficSigns",
            "One Way": "One Way Sign",
            "No Entry": "No Entry Sign",
            "Parking": "Parking Sign",
            "Directional Sign": "Directional Arrow",
            "Taxi and Bus Lane": "Bus Lane",
            "No Turn on Red": "No Turn On Red",
            "Road Work Sign": "Road Work",
            "Road Work Warning": "Road Work Ahead",
            "Road Work Ahead Sign": "Road Work Ahead",
            "Bike Lane": "Bike Lane Sign",
            "Pedestrian Crossing Sign": "Pedestrian Crossing",
            "Keep Right": "Keep Right Sign",
            "Bus Stop": "Bus Stop Sign",
            "Gas Station": "Gas Station Sign",
            "Hospital": "Hospital Sign",
            "Roundabout": "Roundabout Sign",
            "Information": "Information Sign",
            "Warning Sign": "Information Sign",
            "Stop for Pedestrian": "Stop for Pedestrians",
        }
    ),
    "Vehicles.InMotion": _build_alias_map({"1": "True", "0": "False"}),
    "Vehicles.VehicleTypes": _build_alias_map(
        {
            "Car": "Cars",
            "Vans": "Van",
            "Commercial Truck": "Commercial Vehicle",
            "Commercial Vehicles": "Commercial Vehicle",
            "Commerical Truck": "Commercial Vehicle",
            "NoVehicleType": None,
            "Bicycle": None,
            "Trailer": "Semi-Trailer Truck",
            "Camper": "RV",
            "Smart Car": "Compact Car",
            "Compact": "Compact Car",
        }
    ),
    "Vehicles.States": _build_alias_map(
        {
            "NoVehicleState": None,
            "Stopping": "Stopped",
            "Making a Left Turn": "Turning Left",
            "Making a Right Turn": "Turning Right",
            "Moving": "In Motion",
            "Waiting at Traffic Light": "Stopped at Traffic Light",
            "Loading/Unloading": "Loading",
        }
    ),
    "Pedestrians": _build_alias_map(
        {
            "Few": "MultiplePed",
            "One": "MultiplePed",
            "Single": "MultiplePed",
            "Present": "MultiplePed",
            "Visible": "MultiplePed",
            "Visible on Crosswalk": "Crossing Street",
            "Waiting at Crosswalk": "Crossing Street",
            "Crosswalk": "Crossing Street",
            "Present on Sidewalk": "Visible on Sidewalk",
            "Walking on Sidewalk": "Visible on Sidewalk",
            "Standing on Sidewalk": "Visible on Sidewalk",
            "Walking": "Visible on Sidewalk",
            "Not Present": "NoPed",
        }
    ),
    "Policeman.States": _build_alias_map({"None": "NoPolicemanState", "NoState": "NoPolicemanState"}),
    "Visibility.SpecificImpairments": _build_alias_map(
        {
            "Camera glare": "Camera Glare",
            "CameraGlare": "Camera Glare",
            "Glare": "Camera Glare",
            "Sun Glare": "Camera Glare",
            "Obstructed by Rain": "Obstructed by Rain",
            "ObstructedByRain": "Obstructed by Rain",
            "RainOnLens": "Obstructed by Rain",
            "Shadows and Vegetation": "Shadows",
        }
    ),
}


def load_annotation_taxonomy(path: str | Path = DEFAULT_TAXONOMY_PATH) -> dict[str, Any]:
    taxonomy_path = Path(path)
    if not taxonomy_path.exists():
        return {}
    return json.loads(taxonomy_path.read_text(encoding="utf-8"))


def _path_get(obj: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _path_set(obj: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _path_delete(obj: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return
        cur = cur[part]
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)


def taxonomy_options_for_field(field_path: str, taxonomy: dict[str, Any]) -> list[Any]:
    taxonomy_path = FIELD_PATH_TO_TAXONOMY_PATH.get(field_path)
    if not taxonomy_path or not taxonomy:
        return []
    value = _path_get(taxonomy, taxonomy_path, [])
    if isinstance(value, list):
        return copy.deepcopy(value)
    return []


def _allowed_lookup(field_path: str, taxonomy: dict[str, Any]) -> dict[str, Any]:
    return {_normalized_key(option): option for option in taxonomy_options_for_field(field_path, taxonomy)}


def _parse_listish(value: Any) -> list[Any] | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(parsed, list):
        return parsed
    return None


def _flatten_values(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, list):
        flattened: list[Any] = []
        for value in values:
            flattened.extend(_flatten_values(value))
        return flattened
    parsed = _parse_listish(values)
    if parsed is not None:
        return _flatten_values(parsed)
    return [values]


def _normalize_scene_candidate(value: str) -> str:
    parsed = _parse_listish(value)
    if parsed:
        options = [str(item) for item in parsed]
        for candidate in SCENE_PRIORITY:
            if candidate in options:
                return candidate
        return options[0] if options else value
    normalized_key = _normalized_key(value)
    if "bridge" in normalized_key:
        return "Under Bridge"
    if "roundabout" in normalized_key:
        return "Intersection"
    if "residential" in normalized_key:
        return "Residential"
    if "commercial" in normalized_key or "pedestrian zone" in normalized_key:
        return "Commercial"
    if "parking" in normalized_key:
        return "Parking Area"
    if "construction" in normalized_key or "work zone" in normalized_key:
        return "Construction"
    if "school" in normalized_key:
        return "School Zone"
    if "industrial" in normalized_key:
        return "Industrial"
    if "rural" in normalized_key:
        return "Rural"
    if "suburban" in normalized_key:
        return "Suburban"
    if "highway" in normalized_key:
        return "Highway"
    if "intersection" in normalized_key:
        return "Intersection"
    if "urban" in normalized_key:
        return "Urban"
    return value


def _normalize_traffic_light_candidate(value: Any, taxonomy: dict[str, Any]) -> str | None:
    flattened = [str(item) for item in _flatten_values(value) if item is not None]
    if not flattened:
        return None
    normalized_values = []
    for item in flattened:
        mapped = SCALAR_ALIASES["TrafficSigns.TrafficLightState"].get(_normalized_key(item), item)
        if mapped is None:
            continue
        lookup = _allowed_lookup("TrafficSigns.TrafficLightState", taxonomy)
        canonical = lookup.get(_normalized_key(mapped))
        if canonical:
            normalized_values.append(canonical)
            continue
        if mapped in {"Red", "Green", "Yellow"}:
            normalized_values.append(mapped)
    normalized_values = _dedupe_preserve_order(normalized_values)
    if len(normalized_values) >= 2:
        return "In Transition(TwoState)"
    if normalized_values:
        return normalized_values[0]
    return None


def _normalize_scalar_candidate(field_path: str, value: Any, taxonomy: dict[str, Any]) -> Any:
    if value is None:
        return None
    if field_path == "Severity":
        try:
            return int(value)
        except Exception:
            return None
    if field_path == "TrafficSigns.TrafficLightState":
        return _normalize_traffic_light_candidate(value, taxonomy)
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if field_path == "Scene":
        value = _normalize_scene_candidate(value)
    else:
        parsed = _parse_listish(value)
        if parsed:
            value = str(parsed[0]) if parsed else value
    allowed_lookup = _allowed_lookup(field_path, taxonomy)
    direct = allowed_lookup.get(_normalized_key(value))
    if direct is not None:
        return direct
    mapped = SCALAR_ALIASES.get(field_path, {}).get(_normalized_key(value), value)
    if mapped is None:
        return None
    direct = allowed_lookup.get(_normalized_key(mapped))
    if direct is not None:
        return direct
    if field_path == "Directionality" and _normalized_key(mapped) == "multi directional":
        return "Multi-directional with median"
    if field_path in OPTIONAL_SCALAR_FIELDS:
        return None
    return mapped


def _normalize_list_item(field_path: str, value: Any, taxonomy: dict[str, Any]) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    allowed_lookup = _allowed_lookup(field_path, taxonomy)
    direct = allowed_lookup.get(_normalized_key(value))
    if direct is not None:
        return direct
    mapped = LIST_ALIASES.get(field_path, {}).get(_normalized_key(value), value)
    if mapped is None:
        return None
    if field_path == "LaneInformation.SpecialLanes":
        key = _normalized_key(mapped)
        if "left turn" in key:
            mapped = "Left Turn Only Lane"
        elif "right turn" in key:
            mapped = "Right Turn Only Lane"
    if field_path == "TrafficSigns.TrafficSignsTypes":
        key = _normalized_key(mapped)
        if "railroad crossing" in key:
            mapped = "Railroad Crossing"
        elif "bike lane" in key:
            mapped = "Bike Lane Sign"
        elif "road closed" in key:
            mapped = "Road Closed"
        elif "school zone" in key:
            mapped = "School Zone"
    direct = allowed_lookup.get(_normalized_key(mapped))
    if direct is not None:
        return direct
    return None


def normalize_scalar_value(field_path: str, value: Any, taxonomy: dict[str, Any] | None = None) -> Any:
    taxonomy = taxonomy or load_annotation_taxonomy()
    normalized = _normalize_scalar_candidate(field_path, value, taxonomy)
    if normalized is not None:
        return normalized
    return SCALAR_DEFAULTS.get(field_path)


def normalize_list_values(field_path: str, values: Any, taxonomy: dict[str, Any] | None = None) -> list[Any]:
    taxonomy = taxonomy or load_annotation_taxonomy()
    normalized: list[Any] = []
    for value in _flatten_values(values):
        mapped = _normalize_list_item(field_path, value, taxonomy)
        if mapped is None:
            continue
        normalized.append(mapped)
    normalized = _dedupe_preserve_order(normalized)
    if field_path == "LaneInformation.SpecialLanes" and len(normalized) > 1:
        normalized = [item for item in normalized if item != "NoSpecialLanes"]
    if field_path == "TrafficSigns.TrafficSignsTypes" and len(normalized) > 1:
        normalized = [item for item in normalized if item != "NoTrafficSigns"]
    if field_path == "Pedestrians" and len(normalized) > 1:
        normalized = [item for item in normalized if item != "NoPed"]
    if field_path == "Visibility.SpecificImpairments" and len(normalized) > 1:
        normalized = [item for item in normalized if item != "NoImpairments"]
    if not normalized:
        return copy.deepcopy(LIST_DEFAULTS.get(field_path, []))
    return normalized


def _infer_ego_direction(direction: Any, maneuver: Any) -> str:
    if isinstance(direction, str):
        return direction
    if maneuver == "EgoTurning Right":
        return "EgoMaking a Right Turn"
    if maneuver == "EgoTurning Left":
        return "EgoMaking a Left Turn"
    if maneuver == "EgoMerging":
        return "EgoMerging"
    if maneuver in {"EgoFullStopped", "EgoWaiting", "EgoIn Queue", "EgoYielding", "EgoStopped in Parking Space"}:
        return "EgoStopped"
    return "EgoForward"


def _infer_ego_maneuver(direction: Any, maneuver: Any) -> str:
    if isinstance(maneuver, str):
        return maneuver
    if direction == "EgoMaking a Right Turn":
        return "EgoTurning Right"
    if direction == "EgoMaking a Left Turn":
        return "EgoTurning Left"
    if direction == "EgoMerging":
        return "EgoMerging"
    if isinstance(direction, str) and direction.startswith("EgoStopped"):
        return "EgoFullStopped"
    return "EgoMoving"


def normalize_record_to_taxonomy(record: dict[str, Any], taxonomy: dict[str, Any] | None = None) -> dict[str, Any]:
    taxonomy = taxonomy or load_annotation_taxonomy()
    normalized = copy.deepcopy(record)

    for field_path in SCALAR_FIELDS:
        value = _path_get(normalized, field_path)
        if value is None and field_path not in SCALAR_DEFAULTS and field_path not in OPTIONAL_SCALAR_FIELDS:
            continue
        candidate = _normalize_scalar_candidate(field_path, value, taxonomy)
        if candidate is None:
            if field_path in OPTIONAL_SCALAR_FIELDS:
                _path_delete(normalized, field_path)
                continue
            candidate = SCALAR_DEFAULTS.get(field_path)
        if candidate is None:
            _path_delete(normalized, field_path)
            continue
        _path_set(normalized, field_path, candidate)

    for field_path in LIST_FIELDS:
        value = _path_get(normalized, field_path)
        _path_set(normalized, field_path, normalize_list_values(field_path, value, taxonomy))

    if _path_get(normalized, "Visibility.SpecificImpairments") is None:
        _path_set(normalized, "Visibility.SpecificImpairments", ["NoImpairments"])

    total_number = _path_get(normalized, "Vehicles.TotalNumber")
    if total_number == "NoVehicle":
        _path_set(normalized, "Vehicles.VehicleTypes", [])
        _path_set(normalized, "Vehicles.InMotion", [])
        _path_set(normalized, "Vehicles.States", [])

    traffic_sign_types = _path_get(normalized, "TrafficSigns.TrafficSignsTypes", [])
    if not traffic_sign_types:
        _path_set(normalized, "TrafficSigns.TrafficSignsTypes", ["NoTrafficSigns"])
        traffic_sign_types = ["NoTrafficSigns"]
    if "NoTrafficSigns" in traffic_sign_types:
        _path_set(normalized, "TrafficSigns.TrafficSignsVisibility", "SignNotVisible")
        _path_delete(normalized, "TrafficSigns.TrafficLightState")
    elif _path_get(normalized, "TrafficSigns.TrafficSignsVisibility") is None:
        _path_set(normalized, "TrafficSigns.TrafficSignsVisibility", "SignNotIndicated")

    direction = _path_get(normalized, "Ego-Vehicle.Direction")
    maneuver = _path_get(normalized, "Ego-Vehicle.Maneuver")
    _path_set(normalized, "Ego-Vehicle.Direction", _infer_ego_direction(direction, maneuver))
    _path_set(normalized, "Ego-Vehicle.Maneuver", _infer_ego_maneuver(direction, maneuver))

    if isinstance(normalized.get("Bicyclist"), dict):
        presence = normalize_scalar_value("Bicyclist.Presence", normalized["Bicyclist"].get("Presence"), taxonomy)
        location = normalize_scalar_value("Bicyclist.Location", normalized["Bicyclist"].get("Location"), taxonomy)
        normalized["Bicyclist"] = {
            "Presence": presence or "False",
            "Location": location if presence == "True" else "NoBicyclistLocation",
        }

    if isinstance(normalized.get("Policeman"), dict):
        presence = normalize_scalar_value("Policeman.Presence", normalized["Policeman"].get("Presence"), taxonomy)
        states = normalize_list_values("Policeman.States", normalized["Policeman"].get("States"), taxonomy)
        normalized["Policeman"] = {
            "Presence": presence or "False",
            "States": states if presence == "True" else ["NoPolicemanState"],
        }

    if isinstance(normalized.get("Animals"), dict):
        animal_type = normalize_scalar_value("Animals.Type", normalized["Animals"].get("Type"), taxonomy)
        animal_state = normalize_scalar_value("Animals.State", normalized["Animals"].get("State"), taxonomy)
        normalized["Animals"] = {
            "Type": animal_type or "NoAnimal",
            "State": animal_state if animal_type and animal_type != "NoAnimal" else "NoAnimalState",
        }

    return normalized


def normalize_record_for_agreement(record: dict[str, Any]) -> dict[str, Any]:
    return normalize_record_to_taxonomy(record)
