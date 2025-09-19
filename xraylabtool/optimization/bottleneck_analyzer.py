"""
Bottleneck identification and analysis system for XRayLabTool optimization.

This module provides comprehensive profiling and analysis tools to identify
performance bottlenecks in calculation pipelines, memory allocation patterns,
and vectorization opportunities.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
import cProfile
from dataclasses import asdict, dataclass
from functools import wraps
import io
import json
from pathlib import Path
import pstats
import time
from typing import Any

# Try to import line_profiler if available
try:
    from line_profiler import LineProfiler

    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

import builtins

from xraylabtool.optimization.memory_profiler import MemoryProfiler


@dataclass
class FunctionProfile:
    """Profile data for a single function."""

    function_name: str
    file_path: str
    line_number: int
    total_time: float
    cumulative_time: float
    call_count: int
    time_per_call: float
    percentage_of_total: float
    is_builtin: bool


@dataclass
class LineProfile:
    """Line-by-line profiling data."""

    line_number: int
    hits: int
    time_per_hit: float
    total_time: float
    percentage: float
    line_contents: str


@dataclass
class MemoryBottleneck:
    """Memory allocation bottleneck information."""

    location: str
    allocation_size: float  # MB
    allocation_count: int
    percentage_of_total: float
    growth_rate: float  # MB/sec
    is_frequent: bool


@dataclass
class VectorizationOpportunity:
    """Identified vectorization opportunity."""

    function_name: str
    file_path: str
    line_range: tuple[int, int]
    loop_type: str  # 'for_loop', 'while_loop', 'nested_loop'
    estimated_benefit: str  # 'high', 'medium', 'low'
    current_pattern: str
    suggested_optimization: str
    complexity_score: int


@dataclass
class BottleneckAnalysisReport:
    """Comprehensive bottleneck analysis report."""

    timestamp: str
    analysis_duration: float
    function_bottlenecks: list[FunctionProfile]
    line_bottlenecks: list[LineProfile]
    memory_bottlenecks: list[MemoryBottleneck]
    vectorization_opportunities: list[VectorizationOpportunity]
    recommendations: list[str]
    summary_stats: dict[str, Any]


class BottleneckAnalyzer:
    """
    Comprehensive bottleneck identification and analysis system.

    Features:
    - Function-level profiling with cProfile
    - Line-by-line profiling when line_profiler available
    - Memory allocation bottleneck detection
    - Vectorization opportunity identification
    - Performance pattern analysis
    - Actionable optimization recommendations
    """

    def __init__(self, enable_line_profiling: bool = True):
        """
        Initialize bottleneck analyzer.

        Args:
            enable_line_profiling: Enable line-by-line profiling if available
        """
        self.enable_line_profiling = enable_line_profiling and LINE_PROFILER_AVAILABLE
        self.memory_profiler = MemoryProfiler()
        self.profiles = {}
        self.line_profiles = {}

    def profile_function(self, func: Callable) -> Callable:
        """
        Decorator to profile a specific function.

        Args:
            func: Function to profile

        Returns:
            Wrapped function with profiling
        """

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            profiler = cProfile.Profile()
            line_profiler = None

            # Check if there's already an active profiler
            import sys

            enable_line_profiling = False  # Disable line profiling to avoid conflicts

            # Only enable if explicitly safe (no other monitoring active)
            if (
                self.enable_line_profiling
                and hasattr(sys, "monitoring")
                and not sys.monitoring.get_tool(sys.monitoring.PROFILER_ID)
            ):
                enable_line_profiling = True

            if enable_line_profiling:
                try:
                    line_profiler = LineProfiler()
                    line_profiler.add_function(func)
                    line_profiler.enable_by_count()
                except ValueError as e:
                    if "Another profiling tool is already active" in str(e):
                        # Gracefully fallback to function-level profiling only
                        enable_line_profiling = False
                        line_profiler = None
                    else:
                        raise

            try:
                try:
                    profiler.enable()
                    start_time = time.perf_counter()

                    result = func(*args, **kwargs)

                    end_time = time.perf_counter()
                    profiler.disable()
                except ValueError as e:
                    if "Another profiling tool is already active" in str(e):
                        # Fallback: just measure execution time
                        start_time = time.perf_counter()
                        result = func(*args, **kwargs)
                        end_time = time.perf_counter()
                    else:
                        raise

                if enable_line_profiling and line_profiler:
                    try:
                        line_profiler.disable_by_count()
                    except (ValueError, AttributeError):
                        # Ignore line profiler errors during cleanup
                        pass

                # Store profiling data
                profile_name = f"{func.__module__}.{func.__name__}"
                self.profiles[profile_name] = {
                    "profiler": profiler,
                    "duration": end_time - start_time,
                    "args_info": self._get_args_info(args, kwargs),
                }

                if enable_line_profiling and line_profiler:
                    self.line_profiles[profile_name] = line_profiler

                return result

            except Exception as e:
                with suppress(builtins.BaseException):
                    profiler.disable()
                if enable_line_profiling and line_profiler:
                    with suppress(builtins.BaseException):
                        line_profiler.disable_by_count()
                raise e

        return wrapper

    @contextmanager
    def profile_operation(
        self, operation_name: str, enable_memory_tracking: bool = True, **context
    ) -> Iterator[None]:
        """
        Context manager for profiling an operation.

        Args:
            operation_name: Name of the operation being profiled
            enable_memory_tracking: Whether to track memory allocations
            **context: Additional context information
        """
        profiler = cProfile.Profile()
        line_profiler = None

        # Check if there's already an active profiler
        import sys

        enable_line_profiling = False  # Disable line profiling to avoid conflicts

        # Only enable if explicitly safe (no other monitoring active)
        if (
            self.enable_line_profiling
            and hasattr(sys, "monitoring")
            and not sys.monitoring.get_tool(sys.monitoring.PROFILER_ID)
        ):
            enable_line_profiling = True

        if enable_line_profiling:
            try:
                line_profiler = LineProfiler()
                line_profiler.enable_by_count()
            except ValueError as e:
                if "Another profiling tool is already active" in str(e):
                    # Gracefully fallback to function-level profiling only
                    enable_line_profiling = False
                    line_profiler = None
                else:
                    raise

        memory_context = None
        if enable_memory_tracking:
            memory_context = self.memory_profiler.profile_operation(
                operation_name, **context
            )

        try:
            if memory_context:
                memory_context.__enter__()

            try:
                profiler.enable()
                start_time = time.perf_counter()

                yield

                end_time = time.perf_counter()
                profiler.disable()
            except ValueError as e:
                if "Another profiling tool is already active" in str(e):
                    # Fallback: just measure execution time
                    start_time = time.perf_counter()
                    yield
                    end_time = time.perf_counter()
                    # Create a dummy profiler for consistency
                    profiler = None
                else:
                    raise

            if enable_line_profiling and line_profiler:
                try:
                    line_profiler.disable_by_count()
                except (ValueError, AttributeError):
                    # Ignore line profiler errors during cleanup
                    pass

            # Store profiling data
            self.profiles[operation_name] = {
                "profiler": profiler,
                "duration": end_time - start_time,
                "context": context,
            }

            if enable_line_profiling and line_profiler:
                self.line_profiles[operation_name] = line_profiler

        finally:
            if memory_context:
                memory_context.__exit__(None, None, None)

    def analyze_function_bottlenecks(
        self, profile_name: str, top_n: int = 20
    ) -> list[FunctionProfile]:
        """
        Analyze function-level bottlenecks from profiling data.

        Args:
            profile_name: Name of the profile to analyze
            top_n: Number of top bottlenecks to return

        Returns:
            List of function bottlenecks sorted by impact
        """
        if profile_name not in self.profiles:
            return []

        profiler = self.profiles[profile_name]["profiler"]
        total_time = self.profiles[profile_name]["duration"]

        # Handle case where profiler couldn't be used
        if profiler is None:
            return []

        # Capture profiler output
        s = io.StringIO()
        try:
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats("cumulative")
        except (TypeError, ValueError):
            # Profiler data is invalid or empty
            return []

        bottlenecks = []
        for func, (cc, _nc, tt, ct, _callers) in ps.stats.items():
            filename, line_num, func_name = func

            # Skip built-in functions for now (focus on our code)
            is_builtin = filename.startswith("<") and filename.endswith(">")

            time_per_call = tt / cc if cc > 0 else 0
            percentage = (ct / total_time * 100) if total_time > 0 else 0

            bottleneck = FunctionProfile(
                function_name=func_name,
                file_path=filename,
                line_number=line_num,
                total_time=tt,
                cumulative_time=ct,
                call_count=cc,
                time_per_call=time_per_call,
                percentage_of_total=percentage,
                is_builtin=is_builtin,
            )
            bottlenecks.append(bottleneck)

        # Sort by cumulative time and return top N
        bottlenecks.sort(key=lambda x: x.cumulative_time, reverse=True)
        return bottlenecks[:top_n]

    def analyze_line_bottlenecks(
        self, profile_name: str, top_n: int = 20
    ) -> list[LineProfile]:
        """
        Analyze line-by-line bottlenecks if line profiling is available.

        Args:
            profile_name: Name of the profile to analyze
            top_n: Number of top line bottlenecks to return

        Returns:
            List of line bottlenecks sorted by impact
        """
        if not self.enable_line_profiling or profile_name not in self.line_profiles:
            return []

        line_profiler = self.line_profiles[profile_name]

        # Capture line profiler output
        s = io.StringIO()
        line_profiler.print_stats(stream=s)

        # Parse line profiler output
        lines = s.getvalue().split("\n")
        line_bottlenecks = []

        for line in lines:
            if "Line #" in line and "Hits" in line:
                continue  # Header line

            if line.strip() and not line.startswith(" "):
                line.strip()
                continue

            # Parse line profile data
            parts = line.split()
            if len(parts) >= 6 and parts[0].isdigit():
                try:
                    line_num = int(parts[0])
                    hits = int(parts[1]) if parts[1].isdigit() else 0
                    time_per_hit = (
                        float(parts[2]) if parts[2].replace(".", "").isdigit() else 0
                    )
                    total_time = (
                        float(parts[3]) if parts[3].replace(".", "").isdigit() else 0
                    )
                    percentage = (
                        float(parts[4]) if parts[4].replace(".", "").isdigit() else 0
                    )
                    line_contents = " ".join(parts[5:])

                    line_bottleneck = LineProfile(
                        line_number=line_num,
                        hits=hits,
                        time_per_hit=time_per_hit,
                        total_time=total_time,
                        percentage=percentage,
                        line_contents=line_contents,
                    )
                    line_bottlenecks.append(line_bottleneck)
                except (ValueError, IndexError):
                    continue

        # Sort by total time and return top N
        line_bottlenecks.sort(key=lambda x: x.total_time, reverse=True)
        return line_bottlenecks[:top_n]

    def analyze_memory_bottlenecks(self) -> list[MemoryBottleneck]:
        """
        Analyze memory allocation bottlenecks.

        Returns:
            List of memory bottlenecks sorted by impact
        """
        if (
            not hasattr(self.memory_profiler, "allocation_profiles")
            or not self.memory_profiler.allocation_profiles
        ):
            return []

        bottlenecks = []
        total_allocations = sum(
            profile.total_allocated_mb
            for profile in self.memory_profiler.allocation_profiles.values()
        )

        for location, profile in self.memory_profiler.allocation_profiles.items():
            # Calculate growth rate (approximate)
            growth_rate = 0
            if profile.peak_memory_mb > 0 and hasattr(profile, "duration"):
                growth_rate = profile.peak_memory_mb / getattr(profile, "duration", 1)

            percentage = (
                (profile.total_allocated_mb / total_allocations * 100)
                if total_allocations > 0
                else 0
            )
            is_frequent = (
                profile.allocation_count > 100
            )  # Threshold for frequent allocations

            bottleneck = MemoryBottleneck(
                location=location,
                allocation_size=profile.total_allocated_mb,
                allocation_count=profile.allocation_count,
                percentage_of_total=percentage,
                growth_rate=growth_rate,
                is_frequent=is_frequent,
            )
            bottlenecks.append(bottleneck)

        # Sort by allocation size
        bottlenecks.sort(key=lambda x: x.allocation_size, reverse=True)
        return bottlenecks

    def identify_vectorization_opportunities(
        self, source_paths: list[Path]
    ) -> list[VectorizationOpportunity]:
        """
        Identify potential vectorization opportunities in source code.

        Args:
            source_paths: List of source file paths to analyze

        Returns:
            List of vectorization opportunities
        """
        opportunities = []

        for path in source_paths:
            if not path.exists() or path.suffix != ".py":
                continue

            try:
                with open(path) as f:
                    lines = f.readlines()

                opportunities.extend(self._analyze_file_for_vectorization(path, lines))

            except Exception:
                # Skip files that can't be read
                continue

        return opportunities

    def _analyze_file_for_vectorization(
        self, file_path: Path, lines: list[str]
    ) -> list[VectorizationOpportunity]:
        """Analyze a single file for vectorization opportunities."""
        opportunities = []

        # Patterns that suggest vectorization opportunities
        loop_patterns = [
            (r"for\s+\w+\s+in\s+range\s*\(\s*len\s*\(.*\)\s*\)", "for_loop", "high"),
            (r"for\s+\w+\s+in\s+.*:", "for_loop", "medium"),
            (r"while\s+.*:", "while_loop", "low"),
        ]

        array_operation_patterns = [
            (r".*\[.*\]\s*[\+\-\*\/]\s*.*\[.*\]", "element_wise_ops", "high"),
            (r"math\.\w+\s*\(.*\[.*\]\)", "math_functions", "medium"),
            (r"np\..*\s*\(.*\)", "numpy_usage", "low"),  # Already using NumPy
        ]

        import re

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Check for loop patterns
            for pattern, loop_type, benefit in loop_patterns:
                if re.search(pattern, line_stripped):
                    # Look ahead for nested operations
                    context_lines = lines[i : min(i + 10, len(lines))]
                    complexity_score = self._calculate_complexity_score(context_lines)

                    # Check if this looks like a vectorizable operation
                    if any(
                        "*" in l or "+" in l or "-" in l or "/" in l
                        for l in context_lines[1:6]
                    ):
                        opportunity = VectorizationOpportunity(
                            function_name=self._extract_function_name(lines, i),
                            file_path=str(file_path),
                            line_range=(i + 1, min(i + 10, len(lines))),
                            loop_type=loop_type,
                            estimated_benefit=benefit,
                            current_pattern=line_stripped,
                            suggested_optimization=self._suggest_vectorization(
                                line_stripped, context_lines
                            ),
                            complexity_score=complexity_score,
                        )
                        opportunities.append(opportunity)

            # Check for array operation patterns
            for pattern, op_type, benefit in array_operation_patterns:
                if re.search(pattern, line_stripped):
                    opportunity = VectorizationOpportunity(
                        function_name=self._extract_function_name(lines, i),
                        file_path=str(file_path),
                        line_range=(i + 1, i + 1),
                        loop_type=op_type,
                        estimated_benefit=benefit,
                        current_pattern=line_stripped,
                        suggested_optimization=self._suggest_array_optimization(
                            line_stripped
                        ),
                        complexity_score=1,
                    )
                    opportunities.append(opportunity)

        return opportunities

    def _extract_function_name(self, lines: list[str], line_index: int) -> str:
        """Extract the function name containing the given line."""
        # Look backwards for function definition
        for i in range(line_index, max(0, line_index - 50), -1):
            line = lines[i].strip()
            if line.startswith("def "):
                import re

                match = re.match(r"def\s+(\w+)\s*\(", line)
                if match:
                    return match.group(1)
        return "unknown_function"

    def _calculate_complexity_score(self, context_lines: list[str]) -> int:
        """Calculate complexity score for a code block."""
        score = 0
        for line in context_lines:
            stripped = line.strip()
            # Count mathematical operations
            score += (
                stripped.count("+")
                + stripped.count("-")
                + stripped.count("*")
                + stripped.count("/")
            )
            # Count function calls
            score += stripped.count("(")
            # Count array indexing
            score += stripped.count("[")
            # Nested loops increase complexity
            if "for " in stripped or "while " in stripped:
                score += 3
        return score

    def _suggest_vectorization(self, line: str, context: list[str]) -> str:
        """Suggest vectorization optimization for a loop."""
        if "range(len(" in line:
            return (
                "Replace with NumPy broadcasting: np.array operations on entire arrays"
            )
        elif "for " in line and any("*" in l or "+" in l for l in context):
            return "Use NumPy vectorized operations instead of explicit loops"
        else:
            return "Consider NumPy array operations for better performance"

    def _suggest_array_optimization(self, line: str) -> str:
        """Suggest optimization for array operations."""
        if "math." in line:
            return "Replace math.* functions with np.* equivalents for array operations"
        elif "[" in line and "]" in line:
            return "Consider NumPy broadcasting to eliminate element-wise indexing"
        else:
            return "Use NumPy vectorized functions where possible"

    def generate_comprehensive_report(
        self, profile_name: str, source_paths: list[Path] | None = None
    ) -> BottleneckAnalysisReport:
        """
        Generate comprehensive bottleneck analysis report.

        Args:
            profile_name: Name of the profile to analyze
            source_paths: Optional source paths for vectorization analysis

        Returns:
            Comprehensive analysis report
        """
        start_time = time.perf_counter()

        # Analyze function bottlenecks
        function_bottlenecks = self.analyze_function_bottlenecks(profile_name)

        # Analyze line bottlenecks
        line_bottlenecks = self.analyze_line_bottlenecks(profile_name)

        # Analyze memory bottlenecks
        memory_bottlenecks = self.analyze_memory_bottlenecks()

        # Identify vectorization opportunities
        vectorization_opportunities = []
        if source_paths:
            vectorization_opportunities = self.identify_vectorization_opportunities(
                source_paths
            )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            function_bottlenecks,
            line_bottlenecks,
            memory_bottlenecks,
            vectorization_opportunities,
        )

        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(
            function_bottlenecks, memory_bottlenecks, vectorization_opportunities
        )

        analysis_duration = time.perf_counter() - start_time

        return BottleneckAnalysisReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            analysis_duration=analysis_duration,
            function_bottlenecks=function_bottlenecks,
            line_bottlenecks=line_bottlenecks,
            memory_bottlenecks=memory_bottlenecks,
            vectorization_opportunities=vectorization_opportunities,
            recommendations=recommendations,
            summary_stats=summary_stats,
        )

    def _generate_recommendations(
        self,
        function_bottlenecks: list[FunctionProfile],
        line_bottlenecks: list[LineProfile],
        memory_bottlenecks: list[MemoryBottleneck],
        vectorization_opportunities: list[VectorizationOpportunity],
    ) -> list[str]:
        """Generate actionable optimization recommendations."""
        recommendations = []

        # Function-level recommendations
        if function_bottlenecks:
            top_function = function_bottlenecks[0]
            if top_function.percentage_of_total > 30:
                recommendations.append(
                    f"Focus optimization on {top_function.function_name} - consumes"
                    f" {top_function.percentage_of_total:.1f}% of total execution time"
                )

        # Memory recommendations
        high_memory_bottlenecks = [
            mb for mb in memory_bottlenecks if mb.percentage_of_total > 20
        ]
        if high_memory_bottlenecks:
            recommendations.append(
                f"Address memory allocation in {high_memory_bottlenecks[0].location} -"
                f" accounts for {high_memory_bottlenecks[0].percentage_of_total:.1f}%"
                " of allocations"
            )

        # Vectorization recommendations
        high_value_vectorization = [
            vo for vo in vectorization_opportunities if vo.estimated_benefit == "high"
        ]
        if high_value_vectorization:
            recommendations.append(
                "High-impact vectorization opportunity in"
                f" {high_value_vectorization[0].function_name} -"
                f" {high_value_vectorization[0].suggested_optimization}"
            )

        # Line-level recommendations
        if line_bottlenecks:
            top_line = line_bottlenecks[0]
            if top_line.percentage > 10:
                recommendations.append(
                    f"Optimize line {top_line.line_number}:"
                    f" {top_line.line_contents.strip()[:50]}... - consumes"
                    f" {top_line.percentage:.1f}% of execution time"
                )

        if not recommendations:
            recommendations.append(
                "No significant bottlenecks identified. Consider micro-optimizations."
            )

        return recommendations

    def _calculate_summary_stats(
        self,
        function_bottlenecks: list[FunctionProfile],
        memory_bottlenecks: list[MemoryBottleneck],
        vectorization_opportunities: list[VectorizationOpportunity],
    ) -> dict[str, Any]:
        """Calculate summary statistics for the analysis."""
        stats = {
            "total_functions_analyzed": len(function_bottlenecks),
            "total_memory_bottlenecks": len(memory_bottlenecks),
            "total_vectorization_opportunities": len(vectorization_opportunities),
            "high_impact_vectorization_count": len(
                [
                    vo
                    for vo in vectorization_opportunities
                    if vo.estimated_benefit == "high"
                ]
            ),
            "top_function_time_percentage": (
                function_bottlenecks[0].percentage_of_total
                if function_bottlenecks
                else 0
            ),
            "total_memory_allocation_mb": sum(
                mb.allocation_size for mb in memory_bottlenecks
            ),
        }
        return stats

    def save_report(self, report: BottleneckAnalysisReport, output_file: Path) -> None:
        """Save bottleneck analysis report to file."""
        # Convert dataclasses to dictionaries for JSON serialization
        report_dict = {
            "timestamp": report.timestamp,
            "analysis_duration": report.analysis_duration,
            "function_bottlenecks": [asdict(fb) for fb in report.function_bottlenecks],
            "line_bottlenecks": [asdict(lb) for lb in report.line_bottlenecks],
            "memory_bottlenecks": [asdict(mb) for mb in report.memory_bottlenecks],
            "vectorization_opportunities": [
                asdict(vo) for vo in report.vectorization_opportunities
            ],
            "recommendations": report.recommendations,
            "summary_stats": report.summary_stats,
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(report_dict, f, indent=2)

    def _get_args_info(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        """Extract argument information for profiling context."""
        info = {"arg_count": len(args), "kwarg_count": len(kwargs)}

        # Try to extract useful information about arguments
        for i, arg in enumerate(args[:3]):  # Only first 3 args
            if hasattr(arg, "__len__") and not isinstance(arg, str):
                info[f"arg_{i}_length"] = len(arg)
            elif isinstance(arg, (int, float)):
                info[f"arg_{i}_value"] = arg

        return info
