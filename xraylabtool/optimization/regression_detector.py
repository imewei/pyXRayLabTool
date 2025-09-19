"""
Performance regression detection framework for XRayLabTool optimization.

This module provides automated performance regression detection with statistical
analysis, threshold monitoring, and alerting capabilities for maintaining
calculation speed targets.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np


@dataclass
class PerformanceMetric:
    """Individual performance measurement."""

    name: str
    value: float
    unit: str
    timestamp: datetime
    context: dict[str, Any]
    test_parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "test_parameters": self.test_parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerformanceMetric":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            value=data["value"],
            unit=data["unit"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            context=data["context"],
            test_parameters=data["test_parameters"],
        )


@dataclass
class RegressionAlert:
    """Performance regression alert information."""

    metric_name: str
    current_value: float
    baseline_value: float
    regression_percentage: float
    severity: str  # 'warning', 'critical'
    detection_timestamp: datetime
    context: dict[str, Any]

    def __str__(self) -> str:
        """Human-readable alert description."""
        return (
            f"[{self.severity.upper()}] Performance regression in {self.metric_name}: "
            f"current={self.current_value:.3f}, baseline={self.baseline_value:.3f}, "
            f"regression={self.regression_percentage:.1f}%"
        )


@dataclass
class RegressionThresholds:
    """Configurable regression detection thresholds."""

    warning_percentage: float = 10.0  # 10% regression triggers warning
    critical_percentage: float = 25.0  # 25% regression triggers critical alert
    minimum_samples: int = 5  # Minimum samples needed for reliable baseline
    baseline_window_days: int = 7  # Days to look back for baseline calculation
    statistical_confidence: float = 0.95  # Confidence level for statistical tests

    def validate(self) -> None:
        """Validate threshold configuration."""
        if self.warning_percentage >= self.critical_percentage:
            raise ValueError("Warning percentage must be less than critical percentage")
        if self.minimum_samples < 3:
            raise ValueError("Minimum samples must be at least 3")
        if not 0.5 <= self.statistical_confidence <= 0.99:
            raise ValueError("Statistical confidence must be between 0.5 and 0.99")


class PerformanceRegressionDetector:
    """
    Automated performance regression detection system.

    Features:
    - Historical performance data tracking
    - Statistical baseline calculation
    - Configurable regression thresholds
    - Multi-level alerting (warning/critical)
    - Trend analysis and prediction
    - Export/import capabilities for CI/CD integration
    """

    def __init__(
        self,
        data_file: Path | None = None,
        thresholds: RegressionThresholds | None = None,
    ):
        """
        Initialize regression detector.

        Args:
            data_file: Path to performance data storage file
            thresholds: Regression detection thresholds
        """
        self.data_file = data_file or Path("performance_history.json")
        self.thresholds = thresholds or RegressionThresholds()
        self.thresholds.validate()

        # Performance metrics storage
        self.metrics_history: dict[str, list[PerformanceMetric]] = defaultdict(list)
        self.baselines: dict[str, float] = {}
        self.alerts: list[RegressionAlert] = []

        # Load existing data
        self._load_history()

    def record_metric(
        self,
        name: str,
        value: float,
        unit: str,
        context: dict[str, Any] | None = None,
        test_parameters: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a performance metric measurement.

        Args:
            name: Metric name (e.g., 'calculations_per_second')
            value: Measured value
            unit: Unit of measurement (e.g., 'calc/sec', 'seconds', 'MB')
            context: Additional context (material type, energy range, etc.)
            test_parameters: Test-specific parameters
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            context=context or {},
            test_parameters=test_parameters or {},
        )

        self.metrics_history[name].append(metric)
        self._update_baseline(name)
        self._save_history()

    def check_regression(
        self,
        metric_name: str,
        current_value: float,
        context: dict[str, Any] | None = None,
    ) -> RegressionAlert | None:
        """
        Check for performance regression against baseline.

        Args:
            metric_name: Name of metric to check
            current_value: Current measured value
            context: Additional context for the measurement

        Returns:
            RegressionAlert if regression detected, None otherwise
        """
        if metric_name not in self.baselines:
            return None

        baseline = self.baselines[metric_name]

        # Calculate regression percentage
        # For performance metrics, lower values are worse (regression)
        if baseline > 0:
            regression_pct = ((baseline - current_value) / baseline) * 100
        else:
            return None

        # Check if regression exceeds thresholds
        severity = None
        if regression_pct >= self.thresholds.critical_percentage:
            severity = "critical"
        elif regression_pct >= self.thresholds.warning_percentage:
            severity = "warning"

        if severity:
            alert = RegressionAlert(
                metric_name=metric_name,
                current_value=current_value,
                baseline_value=baseline,
                regression_percentage=regression_pct,
                severity=severity,
                detection_timestamp=datetime.now(),
                context=context or {},
            )
            self.alerts.append(alert)
            return alert

        return None

    def get_baseline(self, metric_name: str) -> float | None:
        """Get current baseline for a metric."""
        return self.baselines.get(metric_name)

    def get_recent_alerts(self, hours: int = 24) -> list[RegressionAlert]:
        """Get alerts from the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.detection_timestamp >= cutoff]

    def get_trend_analysis(self, metric_name: str, days: int = 30) -> dict[str, Any]:
        """
        Analyze performance trend for a metric.

        Args:
            metric_name: Metric to analyze
            days: Number of days to analyze

        Returns:
            Dictionary with trend analysis results
        """
        if metric_name not in self.metrics_history:
            return {"error": "No data available for metric"}

        cutoff = datetime.now() - timedelta(days=days)
        recent_metrics = [
            m for m in self.metrics_history[metric_name] if m.timestamp >= cutoff
        ]

        if len(recent_metrics) < 2:
            return {"error": "Insufficient data for trend analysis"}

        values = [m.value for m in recent_metrics]
        timestamps = [m.timestamp.timestamp() for m in recent_metrics]

        # Linear regression for trend
        x = np.array(timestamps)
        y = np.array(values)

        if len(x) > 1:
            slope, _intercept = np.polyfit(x, y, 1)
            trend_direction = (
                "improving" if slope > 0 else "degrading" if slope < 0 else "stable"
            )
        else:
            slope = 0
            y[0] if len(y) > 0 else 0
            trend_direction = "stable"

        return {
            "metric_name": metric_name,
            "sample_count": len(recent_metrics),
            "mean_value": statistics.mean(values),
            "median_value": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "min_value": min(values),
            "max_value": max(values),
            "trend_slope": slope,
            "trend_direction": trend_direction,
            "baseline_value": self.baselines.get(metric_name),
            "current_vs_baseline_pct": (
                (
                    (values[-1] - self.baselines[metric_name])
                    / self.baselines[metric_name]
                    * 100
                )
                if metric_name in self.baselines and self.baselines[metric_name] > 0
                else None
            ),
        }

    def export_for_ci(self, output_file: Path | None = None) -> dict[str, Any]:
        """
        Export performance data for CI/CD integration.

        Args:
            output_file: Optional file to write JSON report

        Returns:
            Performance report dictionary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "baselines": self.baselines.copy(),
            "recent_alerts": [
                {
                    "metric": alert.metric_name,
                    "severity": alert.severity,
                    "regression_pct": alert.regression_percentage,
                    "current_value": alert.current_value,
                    "baseline_value": alert.baseline_value,
                }
                for alert in self.get_recent_alerts(24)
            ],
            "metrics_summary": {},
        }

        # Add summary for each metric
        for metric_name in self.metrics_history:
            trend = self.get_trend_analysis(metric_name, days=7)
            if "error" not in trend:
                report["metrics_summary"][metric_name] = {
                    "current_baseline": self.baselines.get(metric_name),
                    "recent_mean": trend["mean_value"],
                    "trend_direction": trend["trend_direction"],
                    "sample_count": trend["sample_count"],
                }

        if output_file:
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2)

        return report

    def _update_baseline(self, metric_name: str) -> None:
        """Update baseline calculation for a metric."""
        if metric_name not in self.metrics_history:
            return

        # Get recent measurements within baseline window
        cutoff = datetime.now() - timedelta(days=self.thresholds.baseline_window_days)
        recent_metrics = [
            m for m in self.metrics_history[metric_name] if m.timestamp >= cutoff
        ]

        if len(recent_metrics) < self.thresholds.minimum_samples:
            return

        # Calculate baseline as median of recent measurements
        # Median is more robust to outliers than mean
        values = [m.value for m in recent_metrics]
        self.baselines[metric_name] = statistics.median(values)

    def _load_history(self) -> None:
        """Load performance history from file."""
        if not self.data_file.exists():
            return

        try:
            with open(self.data_file) as f:
                data = json.load(f)

            # Load metrics history
            for metric_name, metric_list in data.get("metrics_history", {}).items():
                self.metrics_history[metric_name] = [
                    PerformanceMetric.from_dict(m) for m in metric_list
                ]

            # Load baselines
            self.baselines = data.get("baselines", {})

            # Load recent alerts
            for alert_data in data.get("alerts", []):
                alert = RegressionAlert(
                    metric_name=alert_data["metric_name"],
                    current_value=alert_data["current_value"],
                    baseline_value=alert_data["baseline_value"],
                    regression_percentage=alert_data["regression_percentage"],
                    severity=alert_data["severity"],
                    detection_timestamp=datetime.fromisoformat(
                        alert_data["detection_timestamp"]
                    ),
                    context=alert_data["context"],
                )
                self.alerts.append(alert)

        except (json.JSONDecodeError, KeyError, ValueError):
            # If file is corrupted, start fresh
            self.metrics_history = defaultdict(list)
            self.baselines = {}
            self.alerts = []

    def _save_history(self) -> None:
        """Save performance history to file."""
        # Limit history size to prevent unbounded growth
        max_metrics_per_type = 1000
        for metric_name in self.metrics_history:
            if len(self.metrics_history[metric_name]) > max_metrics_per_type:
                # Keep most recent metrics
                self.metrics_history[metric_name] = self.metrics_history[metric_name][
                    -max_metrics_per_type:
                ]

        # Limit alerts history
        max_alerts = 500
        if len(self.alerts) > max_alerts:
            self.alerts = self.alerts[-max_alerts:]

        data = {
            "metrics_history": {
                name: [m.to_dict() for m in metrics]
                for name, metrics in self.metrics_history.items()
            },
            "baselines": self.baselines,
            "alerts": [
                {
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "baseline_value": alert.baseline_value,
                    "regression_percentage": alert.regression_percentage,
                    "severity": alert.severity,
                    "detection_timestamp": alert.detection_timestamp.isoformat(),
                    "context": alert.context,
                }
                for alert in self.alerts
            ],
        }

        # Ensure directory exists
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=2)


# Global instance for easy access
_global_detector: PerformanceRegressionDetector | None = None


def get_global_detector() -> PerformanceRegressionDetector:
    """Get or create global regression detector instance."""
    global _global_detector
    if _global_detector is None:
        _global_detector = PerformanceRegressionDetector()
    return _global_detector


def record_performance_metric(
    name: str,
    value: float,
    unit: str,
    context: dict[str, Any] | None = None,
    test_parameters: dict[str, Any] | None = None,
) -> None:
    """Convenience function to record metric using global detector."""
    detector = get_global_detector()
    detector.record_metric(name, value, unit, context, test_parameters)


def check_for_regression(
    metric_name: str, current_value: float, context: dict[str, Any] | None = None
) -> RegressionAlert | None:
    """Convenience function to check regression using global detector."""
    detector = get_global_detector()
    return detector.check_regression(metric_name, current_value, context)
