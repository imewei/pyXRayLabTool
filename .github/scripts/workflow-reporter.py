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
from pathlib import Path
from typing import Any


class WorkflowReporter:
    """Comprehensive workflow analysis and reporting."""

    def __init__(self, days: int = 7):
        """Initialize the reporter with analysis period."""
        self.days = days
        self.workflows_data = []
        self.analysis_results = {}
        self.repo_context = None
        self.fallback_mode = False

    def verify_github_cli_auth(self) -> bool:
        """Verify GitHub CLI authentication and permissions."""
        try:
            print("🔐 Verifying GitHub CLI authentication...")

            # Check if gh is authenticated
            subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                check=True,
            )

            print("✅ GitHub CLI authenticated successfully")

            # Get repository context
            repo_result = subprocess.run(
                ["gh", "repo", "view", "--json", "nameWithOwner"],
                capture_output=True,
                text=True,
                check=True,
            )

            repo_data = json.loads(repo_result.stdout)
            self.repo_context = repo_data["nameWithOwner"]
            print(f"📂 Repository context: {self.repo_context}")

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ GitHub CLI authentication failed: {e}")
            print("🔄 Switching to fallback analysis mode...")
            self.fallback_mode = True
            return False
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse repository data: {e}")
            print("🔄 Switching to fallback analysis mode...")
            self.fallback_mode = True
            return False

    def fetch_workflow_data(self) -> bool:
        """Fetch workflow run data from GitHub CLI."""
        if self.fallback_mode:
            print("⚠️ Skipping workflow data fetch due to authentication issues")
            return False

        try:
            print(f"📊 Fetching workflow data for last {self.days} days...")

            # Build command with repository context if available
            cmd = ["gh", "run", "list", "--limit", "200"]

            if self.repo_context:
                cmd.extend(["--repo", self.repo_context])

            json_fields = (
                "status,conclusion,workflowName,createdAt,updatedAt,"
                "durationMs,url,headBranch"
            )
            cmd.extend(["--json", json_fields])

            # Fetch recent workflow runs
            result = subprocess.run(
                cmd,
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
            print("🔄 Switching to fallback analysis mode...")
            self.fallback_mode = True
            return False
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse workflow data: {e}")
            print("🔄 Switching to fallback analysis mode...")
            self.fallback_mode = True
            return False

    def analyze_workflow_performance(self) -> dict[str, Any]:
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

    def generate_failure_patterns(self) -> dict[str, Any]:
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

    def fallback_analysis(self) -> dict[str, Any]:
        """Perform basic analysis when API access is unavailable."""
        print("🔧 Running fallback workflow analysis...")

        # Analyze workflow files directly
        workflow_files = []
        workflow_dir = Path(".github/workflows")

        if workflow_dir.exists():
            for workflow_file in workflow_dir.glob("*.yml"):
                try:
                    with open(workflow_file) as f:
                        content = f.read()

                    analysis = {
                        "name": workflow_file.name,
                        "path": str(workflow_file),
                        "has_timeout": "timeout-minutes" in content,
                        "has_retry": any(
                            retry_term in content
                            for retry_term in ["retry", "nick-fields/retry"]
                        ),
                        "has_caching": "actions/cache" in content,
                        "has_continue_on_error": "continue-on-error: true" in content,
                        "triggers": self._extract_triggers(content),
                        "jobs_count": content.count("jobs:"),
                    }

                    workflow_files.append(analysis)

                except Exception as e:
                    print(f"⚠️ Error analyzing {workflow_file}: {e}")

        # Generate basic health assessment
        fallback_results = {
            "total_workflow_files": len(workflow_files),
            "workflows_with_timeout": sum(
                1 for w in workflow_files if w["has_timeout"]
            ),
            "workflows_with_retry": sum(1 for w in workflow_files if w["has_retry"]),
            "workflows_with_caching": sum(
                1 for w in workflow_files if w["has_caching"]
            ),
            "workflow_details": workflow_files,
        }

        self.analysis_results = {
            "fallback_analysis": fallback_results,
            "mode": "fallback",
        }

        return fallback_results

    def _extract_triggers(self, content: str) -> list[str]:
        """Extract workflow triggers from content."""
        triggers = []
        lines = content.split("\n")
        in_on_section = False

        for line in lines:
            line = line.strip()
            if line.startswith("on:"):
                in_on_section = True
                continue
            elif in_on_section:
                if line and not line.startswith(" ") and not line.startswith("-"):
                    break
                elif line.startswith("- ") or ":" in line:
                    trigger = line.replace("- ", "").split(":")[0].strip()
                    if trigger:
                        triggers.append(trigger)

        return triggers

    def _generate_fallback_recommendations(self) -> list[str]:
        """Generate recommendations for fallback analysis mode."""
        fallback_data = self.analysis_results["fallback_analysis"]
        recommendations = [
            "🔄 **Fallback Analysis Mode Active**",
            "⚠️ GitHub CLI authentication failed - performing static workflow analysis",
            "",
            "📊 **Workflow File Analysis:**",
            f"- Total workflow files: {fallback_data['total_workflow_files']}",
            f"- With timeout settings: {fallback_data['workflows_with_timeout']}",
            f"- With retry mechanisms: {fallback_data['workflows_with_retry']}",
            f"- With caching: {fallback_data['workflows_with_caching']}",
            "",
            "💡 **Recommendations based on static analysis:**",
        ]

        # Analyze missing features
        missing_timeout = (
            fallback_data["total_workflow_files"]
            - fallback_data["workflows_with_timeout"]
        )
        missing_retry = (
            fallback_data["total_workflow_files"]
            - fallback_data["workflows_with_retry"]
        )
        missing_cache = (
            fallback_data["total_workflow_files"]
            - fallback_data["workflows_with_caching"]
        )

        if missing_timeout > 0:
            recommendations.append(
                f"- 🕐 {missing_timeout} workflows missing timeout settings"
            )

        if missing_retry > 0:
            recommendations.append(
                f"- 🔄 {missing_retry} workflows could benefit from retry mechanisms"
            )

        if missing_cache > 0:
            recommendations.append(
                f"- 💾 {missing_cache} workflows could benefit from caching"
            )

        recommendations.extend(
            [
                "",
                "🔧 **To fix GitHub CLI authentication issues:**",
                "- Verify GITHUB_TOKEN has 'actions:read' permissions",
                "- Check if repository context is correctly set",
                "- Ensure GitHub CLI is properly authenticated",
                "- Consider using a personal access token with broader permissions",
            ]
        )

        return recommendations

    def generate_recommendations(self) -> list[str]:
        """Generate actionable recommendations based on analysis."""
        print("💡 Generating recommendations...")

        recommendations = []

        # Handle fallback mode recommendations
        if self.fallback_mode and "fallback_analysis" in self.analysis_results:
            return self._generate_fallback_recommendations()

        for workflow, stats in self.analysis_results.items():
            if stats["health_status"] in ["warning", "critical"]:
                recommendations.append(
                    f"🔧 **{workflow}**: Success rate is {stats['success_rate']}% - "
                    f"investigate recent failures and add retry mechanisms"
                )

            if stats["avg_duration_minutes"] > 30:
                duration = stats["avg_duration_minutes"]
                recommendations.append(
                    f"⚡ **{workflow}**: Average runtime is {duration} minutes - "
                    f"consider optimizing with better caching or parallel jobs"
                )

            if stats["recent_failure_count"] >= 3:
                failure_count = stats["recent_failure_count"]
                recommendations.append(
                    f"🚨 **{workflow}**: {failure_count} recent failures - "
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

    def _generate_fallback_markdown_report(self, report_lines: list[str]) -> str:
        """Generate markdown report for fallback mode."""
        fallback_data = self.analysis_results["fallback_analysis"]

        report_lines.extend(
            [
                "## 🔄 Fallback Analysis Mode",
                "",
                "⚠️ **GitHub CLI authentication failed - performing "
                "static workflow analysis**",
                "",
                "## 📊 Workflow File Analysis",
                "",
                f"- **Total workflow files:** {fallback_data['total_workflow_files']}",
                f"- **With timeout settings:** "
                f"{fallback_data['workflows_with_timeout']}",
                f"- **With retry mechanisms:** {fallback_data['workflows_with_retry']}",
                f"- **With caching:** {fallback_data['workflows_with_caching']}",
                "",
                "## 📋 Workflow Details",
                "",
                "| File | Timeout | Retry | Caching | Triggers |",
                "|------|---------|-------|---------|----------|",
            ]
        )

        for workflow in fallback_data["workflow_details"]:
            timeout_icon = "✅" if workflow["has_timeout"] else "❌"
            retry_icon = "✅" if workflow["has_retry"] else "❌"
            cache_icon = "✅" if workflow["has_caching"] else "❌"
            triggers = ", ".join(workflow["triggers"][:3])  # Limit to first 3 triggers
            if len(workflow["triggers"]) > 3:
                triggers += "..."

            line = (
                f"| {workflow['name']} | {timeout_icon} | {retry_icon} | "
                f"{cache_icon} | {triggers} |"
            )
            report_lines.append(line)

        # Add recommendations
        recommendations = self.generate_recommendations()
        report_lines.extend(["", "## 💡 Recommendations", ""])
        report_lines.extend(recommendations)

        return "\n".join(report_lines)

    def generate_markdown_report(self) -> str:
        """Generate a comprehensive markdown report."""
        print("📝 Generating markdown report...")

        report_lines = [
            "# Workflow Analysis Report",
            f"*Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC*",
            f"*Analysis period: Last {self.days} days*",
            "",
        ]

        # Handle fallback mode differently
        if self.fallback_mode:
            return self._generate_fallback_markdown_report(report_lines)

        report_lines.extend(
            [
                "## 📊 Workflow Performance Summary",
                "",
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

            table_row = (
                f"| {workflow} | {stats['total_runs']} | {stats['success_rate']}% | "
                f"{stats['avg_duration_minutes']}m | {health_icon} {stats['health_status']} |"
            )
            report_lines.append(table_row)

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

        # First verify GitHub CLI authentication
        auth_success = self.verify_github_cli_auth()

        if auth_success:
            # Try to fetch workflow data
            data_success = self.fetch_workflow_data()
            if data_success:
                self.analyze_workflow_performance()
            else:
                # If data fetch fails, run fallback analysis
                self.fallback_analysis()
        else:
            # Run fallback analysis if authentication fails
            self.fallback_analysis()

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

        if self.fallback_mode:
            fallback_data = self.analysis_results["fallback_analysis"]
            print("🔄 **Fallback Analysis Mode Active**")
            print(f"📊 Total workflow files: {fallback_data['total_workflow_files']}")
            print(
                f"🕐 With timeout settings: {fallback_data['workflows_with_timeout']}"
            )
            print(f"🔄 With retry mechanisms: {fallback_data['workflows_with_retry']}")
            print(f"💾 With caching: {fallback_data['workflows_with_caching']}")
        else:
            for workflow, stats in self.analysis_results.items():
                status_icon = {
                    "excellent": "🟢",
                    "good": "🟡",
                    "warning": "🟠",
                    "critical": "🔴",
                }.get(stats["health_status"], "❓")

                success_rate = stats["success_rate"]
                total_runs = stats["total_runs"]
                print(
                    f"{status_icon} {workflow:30} | {success_rate:5.1f}% "
                    f"success | {total_runs:2d} runs"
                )

        print("\n💡 Run 'cat workflow_analysis_report.md' for detailed recommendations")
        return True


def main():
    """Run main workflow analysis function."""
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
