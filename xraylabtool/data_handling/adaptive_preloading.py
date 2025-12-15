"""
Adaptive cache pre-loading based on usage patterns.

This module implements intelligent cache warming that learns from user behavior
and automatically adjusts pre-loading strategies to maximize cache hit rates.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# Thread-safe adaptive preloading state
_adaptive_lock = threading.RLock()

# Adaptive preloading configuration
_adaptive_config = {
    "learning_window_hours": 24,  # How long to track usage patterns
    "prediction_threshold": 0.7,  # Confidence threshold for predictions
    "max_preload_elements": 20,  # Maximum elements to preload
    "adaptation_interval_minutes": 30,  # How often to update predictions
    "min_usage_frequency": 0.1,  # Minimum usage frequency to consider
    "pattern_weight_decay": 0.95,  # How much to decay old patterns
}

# Usage pattern tracking for adaptive learning
_usage_history: dict[str, Any] = {
    "element_access_times": defaultdict(list),  # Element -> [timestamps]
    "compound_sequences": deque(maxlen=1000),  # Recent compound calculation sequences
    "session_patterns": defaultdict(list),  # Session -> [elements used]
    "time_based_patterns": defaultdict(
        lambda: defaultdict(int)
    ),  # Hour -> {element: count}
    "element_associations": defaultdict(
        lambda: defaultdict(int)
    ),  # Element -> {associated_element: count}
    "last_adaptation": datetime.now(),
    "predictions": {},  # Current predictions
    "preload_success_rate": 0.0,  # How well predictions work
}

# Active preloading state
_active_preloads: set[str] = set()


def record_element_access(element: str, _context: str = "calculation") -> None:
    """
    Record an element access for adaptive learning.

    Args:
        element: Element symbol that was accessed
        context: Context of the access (calculation, warming, etc.)
    """
    with _adaptive_lock:
        current_time = time.time()
        _usage_history["element_access_times"][element].append(current_time)

        # Keep only recent access times
        cutoff_time = current_time - (_adaptive_config["learning_window_hours"] * 3600)
        _usage_history["element_access_times"][element] = [
            t
            for t in _usage_history["element_access_times"][element]
            if t > cutoff_time
        ]

        # Update time-based patterns
        current_hour = datetime.now().hour
        _usage_history["time_based_patterns"][current_hour][element] += 1


def record_compound_calculation(formula: str, elements: list[str]) -> None:
    """
    Record a compound calculation for sequence analysis.

    Args:
        formula: Chemical formula that was calculated
        elements: List of elements in the compound
    """
    with _adaptive_lock:
        # Record the sequence
        _usage_history["compound_sequences"].append(
            {"formula": formula, "elements": elements, "timestamp": time.time()}
        )

        # Update element associations
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements):
                if i != j:
                    _usage_history["element_associations"][elem1][elem2] += 1


def calculate_element_usage_frequency(element: str) -> float:
    """
    Calculate usage frequency for an element (accesses per hour).

    Args:
        element: Element symbol

    Returns:
        Usage frequency in accesses per hour
    """
    with _adaptive_lock:
        access_times = _usage_history["element_access_times"].get(element, [])
        if not access_times:
            return 0.0

        # Calculate frequency over the learning window
        learning_window_seconds = _adaptive_config["learning_window_hours"] * 3600
        frequency = len(access_times) / (learning_window_seconds / 3600)
        return frequency


def predict_next_elements(
    current_elements: list[str], max_predictions: int = 10
) -> list[tuple[str, float]]:
    """
    Predict which elements are likely to be needed next.

    Args:
        current_elements: Elements currently being used
        max_predictions: Maximum number of predictions to return

    Returns:
        List of (element, confidence) tuples sorted by confidence
    """
    with _adaptive_lock:
        predictions: dict[str, float] = defaultdict(float)

        # Weight 1: Element associations
        for current_elem in current_elements:
            associations = _usage_history["element_associations"].get(current_elem, {})
            total_associations = sum(associations.values())

            if total_associations > 0:
                for associated_elem, count in associations.items():
                    if associated_elem not in current_elements:
                        confidence = (count / total_associations) * 0.4  # 40% weight
                        predictions[associated_elem] += confidence

        # Weight 2: Recent usage frequency
        for element, _access_times in _usage_history["element_access_times"].items():
            if element not in current_elements:
                frequency = calculate_element_usage_frequency(element)
                if frequency >= _adaptive_config["min_usage_frequency"]:
                    confidence = min(frequency / 10.0, 1.0) * 0.3  # 30% weight
                    predictions[element] += confidence

        # Weight 3: Time-based patterns
        current_hour = datetime.now().hour
        hour_patterns = _usage_history["time_based_patterns"].get(current_hour, {})
        total_hour_accesses = sum(hour_patterns.values())

        if total_hour_accesses > 0:
            for element, count in hour_patterns.items():
                if element not in current_elements:
                    confidence = (count / total_hour_accesses) * 0.2  # 20% weight
                    predictions[element] += confidence

        # Weight 4: Recent compound sequences
        recent_sequences = list(_usage_history["compound_sequences"])[-50:]  # Last 50
        sequence_elements = set()
        for seq in recent_sequences:
            sequence_elements.update(seq["elements"])

        for element in sequence_elements:
            if element not in current_elements:
                confidence = 0.1  # 10% weight for being in recent sequences
                predictions[element] += confidence

        # Sort by confidence and return top predictions
        sorted_predictions = sorted(
            predictions.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_predictions[:max_predictions]


def update_adaptive_strategy() -> dict[str, Any]:
    """
    Update the adaptive preloading strategy based on learned patterns.

    Returns:
        Dictionary with updated strategy information
    """
    with _adaptive_lock:
        current_time = datetime.now()

        # Check if it's time for adaptation
        time_since_last = current_time - _usage_history["last_adaptation"]
        if time_since_last.total_seconds() < (
            _adaptive_config["adaptation_interval_minutes"] * 60
        ):
            return {
                "status": "no_update_needed",
                "next_update_in_minutes": _adaptive_config[
                    "adaptation_interval_minutes"
                ],
            }

        # Analyze recent patterns
        recent_elements = []
        cutoff_time = time.time() - (1 * 3600)  # Last hour

        for element, access_times in _usage_history["element_access_times"].items():
            recent_accesses = [t for t in access_times if t > cutoff_time]
            if recent_accesses:
                recent_elements.append(element)

        # Generate predictions for frequently used elements
        predictions = {}
        for element in recent_elements:
            next_elements = predict_next_elements([element], max_predictions=5)
            predictions[element] = next_elements

        # Update global predictions
        _usage_history["predictions"] = predictions
        _usage_history["last_adaptation"] = current_time

        # Calculate adaptation confidence
        total_predictions = sum(len(pred) for pred in predictions.values())
        avg_confidence = 0.0
        if total_predictions > 0:
            total_confidence = sum(
                conf for pred_list in predictions.values() for _, conf in pred_list
            )
            avg_confidence = total_confidence / total_predictions

        return {
            "status": "updated",
            "recent_elements": recent_elements,
            "predictions": predictions,
            "avg_confidence": avg_confidence,
            "adaptation_time": current_time.isoformat(),
            "next_update_in_minutes": _adaptive_config["adaptation_interval_minutes"],
        }


def get_adaptive_preload_recommendations(
    current_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Get adaptive preloading recommendations based on current context.

    Args:
        current_context: Current calculation context (elements, compounds, etc.)

    Returns:
        Dictionary with preloading recommendations
    """
    with _adaptive_lock:
        if current_context is None:
            current_context = {}

        current_elements = current_context.get("elements", [])
        current_context.get("compounds", [])

        # Get predictions for current elements
        all_predictions: dict[str, float] = defaultdict(float)

        # Predictions based on current elements
        for element in current_elements:
            element_predictions = predict_next_elements([element], max_predictions=8)
            for pred_element, confidence in element_predictions:
                all_predictions[pred_element] += confidence

        # General high-frequency elements
        for element, _access_times in _usage_history["element_access_times"].items():
            if element not in current_elements:
                frequency = calculate_element_usage_frequency(element)
                if frequency >= _adaptive_config["min_usage_frequency"]:
                    all_predictions[element] += frequency * 0.1

        # Filter by confidence threshold
        confident_predictions = [
            (element, confidence)
            for element, confidence in all_predictions.items()
            if confidence >= _adaptive_config["prediction_threshold"]
        ]

        # Sort and limit recommendations
        confident_predictions.sort(key=lambda x: x[1], reverse=True)
        recommendations = confident_predictions[
            : _adaptive_config["max_preload_elements"]
        ]

        return {
            "recommended_elements": [elem for elem, conf in recommendations],
            "element_confidences": dict(recommendations),
            "total_recommendations": len(recommendations),
            "avg_confidence": (
                sum(conf for _, conf in recommendations) / len(recommendations)
                if recommendations
                else 0.0
            ),
            "prediction_basis": {
                "current_elements": current_elements,
                "learning_window_hours": _adaptive_config["learning_window_hours"],
                "min_confidence": _adaptive_config["prediction_threshold"],
            },
        }


def execute_adaptive_preloading(
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute adaptive preloading based on current patterns.

    Args:
        context: Current calculation context

    Returns:
        Dictionary with preloading results
    """
    from xraylabtool.data_handling.atomic_cache import warm_cache_for_compounds

    recommendations = get_adaptive_preload_recommendations(context)
    recommended_elements = recommendations["recommended_elements"]

    if not recommended_elements:
        return {
            "status": "no_recommendations",
            "elements_preloaded": [],
            "success_rate": 0.0,
        }

    # Create synthetic compounds for warming (single-element "compounds")
    synthetic_formulas = recommended_elements  # Treat elements as formulas

    try:
        start_time = time.perf_counter()

        # Execute warming
        warming_result = warm_cache_for_compounds(
            synthetic_formulas,
            include_similar=False,  # Don't expand for adaptive preloading
            include_family=False,  # Keep it focused
            timing_info=True,
        )

        end_time = time.perf_counter()
        preload_time_ms = (end_time - start_time) * 1000.0

        # Track which elements were successfully preloaded
        with _adaptive_lock:
            _active_preloads.update(warming_result["elements_warmed"])

        return {
            "status": "completed",
            "elements_preloaded": warming_result["elements_warmed"],
            "success_rate": warming_result["success_rate"],
            "preload_time_ms": preload_time_ms,
            "recommendations_used": recommendations,
            "warming_details": warming_result,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "elements_preloaded": [],
            "success_rate": 0.0,
        }


def measure_preload_effectiveness(actual_elements_used: list[str]) -> dict[str, Any]:
    """
    Measure how effective the adaptive preloading was.

    Args:
        actual_elements_used: Elements that were actually used in calculations

    Returns:
        Dictionary with effectiveness metrics
    """
    with _adaptive_lock:
        if not _active_preloads:
            return {
                "hit_rate": 0.0,
                "preloaded_count": 0,
                "used_count": 0,
                "effectiveness": "no_preloads",
            }

        # Calculate hit rate
        hits = len(set(actual_elements_used) & _active_preloads)
        total_used = len(actual_elements_used)
        hit_rate = (hits / total_used) if total_used > 0 else 0.0

        # Calculate precision (how many preloaded elements were actually used)
        precision = (hits / len(_active_preloads)) if _active_preloads else 0.0

        # Update success rate tracking
        _usage_history["preload_success_rate"] = hit_rate

        effectiveness_metrics = {
            "hit_rate": hit_rate,
            "precision": precision,
            "preloaded_count": len(_active_preloads),
            "used_count": total_used,
            "hits": hits,
            "effectiveness": (
                "excellent"
                if hit_rate > 0.8
                else "good"
                if hit_rate > 0.6
                else "fair"
                if hit_rate > 0.4
                else "poor"
            ),
        }

        # Clear active preloads for next measurement
        _active_preloads.clear()

        return effectiveness_metrics


def get_adaptive_statistics() -> dict[str, Any]:
    """
    Get comprehensive statistics about adaptive preloading performance.

    Returns:
        Dictionary with adaptive preloading statistics
    """
    with _adaptive_lock:
        total_tracked_elements = len(_usage_history["element_access_times"])
        total_accesses = sum(
            len(access_times)
            for access_times in _usage_history["element_access_times"].values()
        )

        # Calculate average usage frequency
        avg_frequency = 0.0
        if total_tracked_elements > 0:
            total_frequency = sum(
                calculate_element_usage_frequency(element)
                for element in _usage_history["element_access_times"]
            )
            avg_frequency = total_frequency / total_tracked_elements

        # Get top elements by frequency
        top_elements = sorted(
            [
                (elem, calculate_element_usage_frequency(elem))
                for elem in _usage_history["element_access_times"]
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        return {
            "learning_window_hours": _adaptive_config["learning_window_hours"],
            "total_tracked_elements": total_tracked_elements,
            "total_accesses": total_accesses,
            "avg_usage_frequency": avg_frequency,
            "top_elements": top_elements,
            "compound_sequences_tracked": len(_usage_history["compound_sequences"]),
            "last_adaptation": _usage_history["last_adaptation"].isoformat(),
            "current_preload_success_rate": _usage_history["preload_success_rate"],
            "active_preloads": len(_active_preloads),
            "configuration": _adaptive_config.copy(),
        }


def reset_adaptive_learning() -> None:
    """Reset all adaptive learning data for testing or new sessions."""
    with _adaptive_lock:
        global _usage_history, _active_preloads

        _usage_history = {
            "element_access_times": defaultdict(list),
            "compound_sequences": deque(maxlen=1000),
            "session_patterns": defaultdict(list),
            "time_based_patterns": defaultdict(lambda: defaultdict(int)),
            "element_associations": defaultdict(lambda: defaultdict(int)),
            "last_adaptation": datetime.now(),
            "predictions": {},
            "preload_success_rate": 0.0,
        }

        _active_preloads = set()


def configure_adaptive_preloading(**config_updates) -> dict[str, Any]:
    """
    Update adaptive preloading configuration.

    Args:
        **config_updates: Configuration parameters to update

    Returns:
        Updated configuration dictionary
    """
    with _adaptive_lock:
        for key, value in config_updates.items():
            if key in _adaptive_config:
                _adaptive_config[key] = value

        return _adaptive_config.copy()
