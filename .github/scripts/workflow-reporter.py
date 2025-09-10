#!/usr/bin/env python3
"""
Enhanced workflow monitoring and reporting script.
Provides detailed analysis and recommendations for GitHub Actions workflows.
"""

import datetime
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional


class WorkflowReporter:
    """Comprehensive workflow analysis and reporting."""

    def __init__(self, days: int = 7):
        """Initialize the reporter with analysis period."""
        self.days = days
        self.workflows_data = []
        self.analysis_results = {}

    def fetch_workflow_data(self) -> bool:
        """Fetch workflow run data from GitHub CLI."""
        try:
            print(f"📊 Fetching workflow data for last {self.days} days...")

            # Fetch recent workflow runs
            result = subprocess.run(
                [
                    "gh",
                    "run",
                    "list",
                    "--limit",
                    "200",
                    "--json",
                    "status,conclusion,workflowName,createdAt,updatedAt,durationMs,url,headBranch",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            self.workflows_data = json.loads(result.stdout)

            # Filter by date range
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=self.days)
            filtered_data = []

            for run in self.workflows_data:
                created_at = datetime.datetime.fromisoformat(
                    run["createdAt"].replace("Z", "+00:00")
                )
                if created_at >= cutoff_date:
                    filtered_data.append(run)

            self.workflows_data = filtered_data
            print(f"✅ Fetched {len(self.workflows_data)} workflow runs")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to fetch workflow data: {e}")
            return False
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse workflow data: {e}")
            return False

    def analyze_workflow_performance(self) -> Dict[str, Any]:
        """Analyze workflow performance metrics."""
        print("📈 Analyzing workflow performance...")

        workflow_stats = defaultdict(
            lambda: {
                "total": 0,
                "success": 0,
                "failure": 0,
                "cancelled": 0,
                "durations": [],
                "branches": defaultdict(int),
                "recent_failures": [],
            }
        )

        for run in self.workflows_data:
            workflow = run["workflowName"]
            conclusion = run["conclusion"] or "running"
            duration = run.get("durationMs", 0)
            branch = run.get("headBranch", "unknown")

            stats = workflow_stats[workflow]
            stats["total"] += 1
            stats["branches"][branch] += 1

            if duration and duration > 0:
                stats["durations"].append(duration)

            if conclusion == "success":
                stats["success"] += 1
            elif conclusion == "failure":
                stats["failure"] += 1
                stats["recent_failures"].append(
                    {"url": run["url"], "created": run["createdAt"], "branch": branch}
                )
            elif conclusion == "cancelled":
                stats["cancelled"] += 1

        # Calculate derived metrics
        analysis = {}
        for workflow, stats in workflow_stats.items():
            total = stats["total"]
            if total == 0:
                continue

            success_rate = (stats["success"] / total) * 100
            failure_rate = (stats["failure"] / total) * 100

            # Calculate average duration
            avg_duration = 0
            if stats["durations"]:
                avg_duration = sum(stats["durations"]) / len(stats["durations"])

            analysis[workflow] = {
                "total_runs": total,
                "success_rate": round(success_rate, 1),
                "failure_rate": round(failure_rate, 1),
                "avg_duration_minutes": (
                    round(avg_duration / 60000, 1) if avg_duration else 0
                ),
                "health_status": self._get_health_status(success_rate),
                "primary_branch": (
                    max(stats["branches"].items(), key=lambda x: x[1])[0]
                    if stats["branches"]
                    else "unknown"
                ),
                "recent_failure_count": len(stats["recent_failures"]),
                "recent_failures": stats["recent_failures"][:3],  # Last 3 failures
            }

        self.analysis_results = analysis
        return analysis

    def _get_health_status(self, success_rate: float) -> str:
        """Determine workflow health status."""
        if success_rate >= 90:
            return "excellent"
        elif success_rate >= 80:
            return "good"
        elif success_rate >= 60:
            return "warning"
        else:
            return "critical"

    def generate_failure_patterns(self) -> Dict[str, Any]:
        """Analyze failure patterns and common issues."""
        print("🔍 Analyzing failure patterns...")

        failure_patterns = {
            "by_workflow": Counter(),
            "by_branch": Counter(),
            "by_time_period": defaultdict(int),
            "common_issues": [],
        }

        for run in self.workflows_data:
            if run["conclusion"] == "failure":
                workflow = run["workflowName"]
                branch = run.get("headBranch", "unknown")
                created = datetime.datetime.fromisoformat(
                    run["createdAt"].replace("Z", "+00:00")
                )

                failure_patterns["by_workflow"][workflow] += 1
                failure_patterns["by_branch"][branch] += 1

                # Group by hour of day
                hour_key = f"{created.hour:02d}:00"
                failure_patterns["by_time_period"][hour_key] += 1

        # Identify common failure times
        if failure_patterns["by_time_period"]:
            peak_failure_time = max(
                failure_patterns["by_time_period"].items(), key=lambda x: x[1]
            )
            failure_patterns["peak_failure_time"] = peak_failure_time

        return failure_patterns

    def generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        print("💡 Generating recommendations...")

        recommendations = []

        for workflow, stats in self.analysis_results.items():
            if stats["health_status"] in ["warning", "critical"]:
                recommendations.append(
                    f"🔧 **{workflow}**: Success rate is {stats['success_rate']}% - "
                    f"investigate recent failures and add retry mechanisms"
                )

            if stats["avg_duration_minutes"] > 30:
                recommendations.append(
                    f"⚡ **{workflow}**: Average runtime is {stats['avg_duration_minutes']} minutes - "
                    f"consider optimizing with better caching or parallel jobs"
                )

            if stats["recent_failure_count"] >= 3:
                recommendations.append(
                    f"🚨 **{workflow}**: {stats['recent_failure_count']} recent failures - "
                    f"requires immediate attention"
                )

        # General recommendations
        if not recommendations:
            recommendations.append("✅ All workflows are performing well!")
        else:
            recommendations.extend(
                [
                    "📋 **General Recommendations:**",
                    "- Add retry mechanisms for network-dependent steps",
                    "- Implement better error handling and logging",
                    "- Use caching to reduce build times",
                    "- Consider matrix build optimization",
                    "- Monitor workflow performance regularly",
                ]
            )

        return recommendations

    def generate_markdown_report(self) -> str:
        """Generate a comprehensive markdown report."""
        print("📝 Generating markdown report...")

        report_lines = [
            f"# Workflow Analysis Report",
            f"*Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC*",
            f"*Analysis period: Last {self.days} days*",
            "",
            "## 📊 Workflow Performance Summary",
            "",
        ]

        # Performance table
        report_lines.extend(
            [
                "| Workflow | Runs | Success Rate | Avg Duration | Health Status |",
                "|----------|------|--------------|--------------|---------------|",
            ]
        )

        for workflow, stats in sorted(self.analysis_results.items()):
            health_icon = {
                "excellent": "🟢",
                "good": "🟡",
                "warning": "🟠",
                "critical": "🔴",
            }.get(stats["health_status"], "❓")

            report_lines.append(
                f"| {workflow} | {stats['total_runs']} | {stats['success_rate']}% | "
                f"{stats['avg_duration_minutes']}m | {health_icon} {stats['health_status']} |"
            )

        # Failure analysis
        failure_patterns = self.generate_failure_patterns()

        report_lines.extend(["", "## 🚨 Failure Analysis", ""])

        if failure_patterns["by_workflow"]:
            report_lines.extend(["### Most Failing Workflows", ""])
            for workflow, count in failure_patterns["by_workflow"].most_common(5):
                report_lines.append(f"- **{workflow}**: {count} failures")

        # Recommendations
        recommendations = self.generate_recommendations()
        report_lines.extend(["", "## 💡 Recommendations", ""])
        report_lines.extend(recommendations)

        # Recent failures detail
        report_lines.extend(["", "## 🔍 Recent Failures Detail", ""])

        for workflow, stats in self.analysis_results.items():
            if stats["recent_failures"]:
                report_lines.extend([f"### {workflow}", ""])
                for failure in stats["recent_failures"]:
                    report_lines.append(
                        f"- [{failure['created']}]({failure['url']}) on `{failure['branch']}`"
                    )
                report_lines.append("")

        return "\n".join(report_lines)

    def save_json_report(self, filename: str = "workflow_analysis.json"):
        """Save detailed analysis as JSON."""
        report_data = {
            "generated_at": datetime.datetime.now().isoformat(),
            "analysis_period_days": self.days,
            "total_runs_analyzed": len(self.workflows_data),
            "workflow_analysis": self.analysis_results,
            "failure_patterns": self.generate_failure_patterns(),
            "recommendations": self.generate_recommendations(),
        }

        with open(filename, "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"💾 JSON report saved to {filename}")

    def run_analysis(self) -> bool:
        """Run complete workflow analysis."""
        print("🚀 Starting comprehensive workflow analysis...")

        if not self.fetch_workflow_data():
            return False

        self.analyze_workflow_performance()

        # Generate reports
        markdown_report = self.generate_markdown_report()

        # Save reports
        with open("workflow_analysis_report.md", "w") as f:
            f.write(markdown_report)
        print("📄 Markdown report saved to workflow_analysis_report.md")

        self.save_json_report()

        # Print summary to console
        print("\n" + "=" * 50)
        print("📋 WORKFLOW ANALYSIS SUMMARY")
        print("=" * 50)

        for workflow, stats in self.analysis_results.items():
            status_icon = {
                "excellent": "🟢",
                "good": "🟡",
                "warning": "🟠",
                "critical": "🔴",
            }.get(stats["health_status"], "❓")

            print(
                f"{status_icon} {workflow:30} | {stats['success_rate']:5.1f}% success | {stats['total_runs']:2d} runs"
            )

        print("\n💡 Run 'cat workflow_analysis_report.md' for detailed recommendations")
        return True


def main():
    """Main function."""
    days = int(os.getenv("ANALYSIS_DAYS", "7"))

    reporter = WorkflowReporter(days=days)

    try:
        success = reporter.run_analysis()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
