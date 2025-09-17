"""
Git integration for codebase cleanup operations.

This module provides Git repository analysis capabilities to help make
informed decisions about file removal based on version control history,
tracking status, and change patterns.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    import git
    from git import Repo, InvalidGitRepositoryError, GitCommandError

    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

    # Create mock classes for when GitPython is not available
    class Repo:
        pass

    class InvalidGitRepositoryError(Exception):
        pass

    class GitCommandError(Exception):
        pass


logger = logging.getLogger(__name__)


@dataclass
class FileHistory:
    """File history information from Git."""

    file_path: Path
    first_commit_date: Optional[datetime]
    last_commit_date: Optional[datetime]
    commit_count: int
    authors: Set[str]
    is_tracked: bool
    is_staged: bool
    is_modified: bool


@dataclass
class AgeAnalysis:
    """File age analysis results."""

    days_since_creation: int
    days_since_last_commit: int
    days_since_last_modification: int
    is_stale: bool
    staleness_score: float  # 0.0 (fresh) to 1.0 (very stale)


class GitChangeAnalyzer:
    """
    Analyzer for Git repository changes and file tracking status.

    This class provides insights into file history, tracking status,
    and change patterns to inform cleanup decisions.
    """

    def __init__(self, repo_path: Union[str, Path], stale_threshold_days: int = 90):
        """
        Initialize the Git analyzer.

        Args:
            repo_path: Path to the Git repository
            stale_threshold_days: Days after which files are considered stale
        """
        self.repo_path = Path(repo_path).resolve()
        self.stale_threshold_days = stale_threshold_days
        self.repo = None

        if not GIT_AVAILABLE:
            logger.warning("GitPython not available - Git integration disabled")
            return

        try:
            self.repo = Repo(self.repo_path)
            logger.info(f"Initialized Git analyzer for repository: {self.repo_path}")
        except InvalidGitRepositoryError:
            logger.warning(f"Not a valid Git repository: {self.repo_path}")
            self.repo = None
        except Exception as e:
            logger.error(f"Failed to initialize Git repository: {e}")
            self.repo = None

    def is_git_available(self) -> bool:
        """Check if Git integration is available and working."""
        return GIT_AVAILABLE and self.repo is not None

    def get_file_history(self, file_path: Path) -> FileHistory:
        """
        Get comprehensive history information for a file.

        Args:
            file_path: Path to the file to analyze

        Returns:
            FileHistory with Git metadata
        """
        if not self.is_git_available():
            return self._create_fallback_history(file_path)

        try:
            relative_path = self._get_relative_path(file_path)

            # Get commit history for the file
            commits = list(self.repo.iter_commits(paths=str(relative_path)))

            # Get file status information
            is_tracked = not self._is_untracked(relative_path)
            is_staged = self._is_staged(relative_path)
            is_modified = self._is_modified(relative_path)

            if commits:
                # Extract commit information
                first_commit = commits[-1]  # Oldest commit
                last_commit = commits[0]  # Most recent commit

                first_commit_date = datetime.fromtimestamp(first_commit.committed_date)
                last_commit_date = datetime.fromtimestamp(last_commit.committed_date)

                authors = {commit.author.name for commit in commits}

                return FileHistory(
                    file_path=file_path,
                    first_commit_date=first_commit_date,
                    last_commit_date=last_commit_date,
                    commit_count=len(commits),
                    authors=authors,
                    is_tracked=is_tracked,
                    is_staged=is_staged,
                    is_modified=is_modified,
                )
            else:
                # File exists but has no commit history
                return FileHistory(
                    file_path=file_path,
                    first_commit_date=None,
                    last_commit_date=None,
                    commit_count=0,
                    authors=set(),
                    is_tracked=is_tracked,
                    is_staged=is_staged,
                    is_modified=is_modified,
                )

        except Exception as e:
            logger.error(f"Failed to get history for {file_path}: {e}")
            return self._create_fallback_history(file_path)

    def get_untracked_files(self, directory: Optional[Path] = None) -> List[Path]:
        """
        Get list of untracked files in the repository.

        Args:
            directory: Specific directory to check (None for entire repo)

        Returns:
            List of untracked file paths
        """
        if not self.is_git_available():
            return []

        try:
            untracked_files = []

            # Get untracked files from Git
            for item in self.repo.untracked_files:
                file_path = self.repo_path / item

                # Filter by directory if specified
                if directory is None or self._is_in_directory(file_path, directory):
                    untracked_files.append(file_path)

            return untracked_files

        except Exception as e:
            logger.error(f"Failed to get untracked files: {e}")
            return []

    def analyze_file_age(self, file_path: Path) -> AgeAnalysis:
        """
        Analyze the age characteristics of a file.

        Args:
            file_path: Path to the file to analyze

        Returns:
            AgeAnalysis with staleness information
        """
        now = datetime.now()

        # Get file system modification time
        try:
            fs_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            days_since_fs_modification = (now - fs_mtime).days
        except (OSError, IOError):
            days_since_fs_modification = 0

        if not self.is_git_available():
            return AgeAnalysis(
                days_since_creation=days_since_fs_modification,
                days_since_last_commit=days_since_fs_modification,
                days_since_last_modification=days_since_fs_modification,
                is_stale=days_since_fs_modification > self.stale_threshold_days,
                staleness_score=min(
                    1.0, days_since_fs_modification / self.stale_threshold_days
                ),
            )

        history = self.get_file_history(file_path)

        # Calculate age metrics
        if history.first_commit_date:
            days_since_creation = (now - history.first_commit_date).days
        else:
            days_since_creation = days_since_fs_modification

        if history.last_commit_date:
            days_since_last_commit = (now - history.last_commit_date).days
        else:
            days_since_last_commit = days_since_fs_modification

        # Determine staleness
        is_stale = (
            days_since_last_commit > self.stale_threshold_days
            and not history.is_modified
            and not history.is_staged
        )

        # Calculate staleness score (0.0 = fresh, 1.0 = very stale)
        staleness_factors = [
            days_since_last_commit / self.stale_threshold_days,
            0.0 if history.is_modified else 0.2,  # Modified files are fresher
            (
                0.0 if history.commit_count > 1 else 0.3
            ),  # Files with history are more important
        ]
        staleness_score = min(1.0, sum(staleness_factors) / len(staleness_factors))

        return AgeAnalysis(
            days_since_creation=days_since_creation,
            days_since_last_commit=days_since_last_commit,
            days_since_last_modification=days_since_fs_modification,
            is_stale=is_stale,
            staleness_score=staleness_score,
        )

    def get_ignored_patterns(self) -> List[str]:
        """
        Get patterns from .gitignore that might indicate safe-to-remove files.

        Returns:
            List of gitignore patterns
        """
        if not self.is_git_available():
            return []

        patterns = []
        gitignore_path = self.repo_path / ".gitignore"

        try:
            if gitignore_path.exists():
                content = gitignore_path.read_text(encoding="utf-8")
                for line in content.split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
        except Exception as e:
            logger.error(f"Failed to read .gitignore: {e}")

        return patterns

    def find_large_files_in_history(
        self, size_threshold_mb: float = 10.0, max_files: int = 50
    ) -> List[Tuple[Path, float]]:
        """
        Find large files that might have been accidentally committed.

        Args:
            size_threshold_mb: Minimum file size to report
            max_files: Maximum number of files to return

        Returns:
            List of (file_path, size_mb) tuples
        """
        if not self.is_git_available():
            return []

        large_files = []

        try:
            # Check files in the current working tree
            for root, dirs, files in os.walk(self.repo_path):
                # Skip .git directory
                if ".git" in dirs:
                    dirs.remove(".git")

                for filename in files:
                    file_path = Path(root) / filename
                    try:
                        size_bytes = file_path.stat().st_size
                        size_mb = size_bytes / (1024 * 1024)

                        if size_mb >= size_threshold_mb:
                            large_files.append((file_path, size_mb))

                    except (OSError, IOError):
                        continue

            # Sort by size (largest first) and limit results
            large_files.sort(key=lambda x: x[1], reverse=True)
            return large_files[:max_files]

        except Exception as e:
            logger.error(f"Failed to find large files: {e}")
            return []

    def check_file_safety_by_git_status(self, file_path: Path) -> Dict[str, any]:
        """
        Check file safety based on Git status and history.

        Args:
            file_path: Path to the file to check

        Returns:
            Dictionary with safety assessment
        """
        if not self.is_git_available():
            return {
                "safe_to_remove": False,
                "confidence": 0.0,
                "reasons": ["Git not available"],
                "git_status": "unknown",
            }

        history = self.get_file_history(file_path)
        age_analysis = self.analyze_file_age(file_path)

        reasons = []
        confidence = 0.5
        safe_to_remove = False

        # Untracked files are generally safer to remove
        if not history.is_tracked:
            reasons.append("File is not tracked by Git")
            confidence += 0.3
            safe_to_remove = True

        # Modified or staged files should be preserved
        if history.is_modified or history.is_staged:
            reasons.append("File has uncommitted changes")
            confidence = 0.1
            safe_to_remove = False

        # Files with no commit history might be temporary
        elif history.commit_count == 0:
            reasons.append("File has no commit history")
            confidence += 0.2

        # Old, unchanged files might be artifacts
        elif age_analysis.is_stale and not history.is_modified:
            reasons.append(
                f"File is stale ({age_analysis.days_since_last_commit} days old)"
            )
            confidence += 0.2

        # Files with extensive history are more important
        if history.commit_count > 10:
            reasons.append(
                f"File has extensive history ({history.commit_count} commits)"
            )
            confidence -= 0.2

        # Multiple authors indicate collaborative work
        if len(history.authors) > 2:
            reasons.append(f"File has multiple authors ({len(history.authors)})")
            confidence -= 0.1

        return {
            "safe_to_remove": safe_to_remove and confidence > 0.6,
            "confidence": max(0.0, min(1.0, confidence)),
            "reasons": reasons,
            "git_status": {
                "tracked": history.is_tracked,
                "modified": history.is_modified,
                "staged": history.is_staged,
                "commit_count": history.commit_count,
                "days_since_last_commit": age_analysis.days_since_last_commit,
            },
        }

    def _get_relative_path(self, file_path: Path) -> Path:
        """Get path relative to repository root."""
        try:
            return file_path.relative_to(self.repo_path)
        except ValueError:
            # File is outside repository
            return file_path

    def _is_untracked(self, relative_path: Path) -> bool:
        """Check if file is untracked."""
        return str(relative_path) in self.repo.untracked_files

    def _is_staged(self, relative_path: Path) -> bool:
        """Check if file is staged for commit."""
        try:
            # Check if file is in the index
            staged_files = [item.a_path for item in self.repo.index.diff("HEAD")]
            return str(relative_path) in staged_files
        except Exception:
            return False

    def _is_modified(self, relative_path: Path) -> bool:
        """Check if file is modified in working directory."""
        try:
            # Check if file is modified
            modified_files = [item.a_path for item in self.repo.index.diff(None)]
            return str(relative_path) in modified_files
        except Exception:
            return False

    def _is_in_directory(self, file_path: Path, directory: Path) -> bool:
        """Check if file is within the specified directory."""
        try:
            file_path.relative_to(directory)
            return True
        except ValueError:
            return False

    def _create_fallback_history(self, file_path: Path) -> FileHistory:
        """Create fallback history when Git is not available."""
        try:
            stat = file_path.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)

            return FileHistory(
                file_path=file_path,
                first_commit_date=mtime,
                last_commit_date=mtime,
                commit_count=0,
                authors=set(),
                is_tracked=False,
                is_staged=False,
                is_modified=False,
            )
        except (OSError, IOError):
            return FileHistory(
                file_path=file_path,
                first_commit_date=None,
                last_commit_date=None,
                commit_count=0,
                authors=set(),
                is_tracked=False,
                is_staged=False,
                is_modified=False,
            )
