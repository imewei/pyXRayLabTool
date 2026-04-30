"""Tests for interfaces/completion_v2/cache.py module.

Tests performance caching system for shell completion data and operations.
"""

import json
from pathlib import Path
import time
from unittest.mock import Mock, patch

import pytest

from xraylabtool.interfaces.completion_v2.cache import CompletionCache


class TestCompletionCache:
    """Tests for CompletionCache class."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Create temporary cache directory for testing."""
        return tmp_path / "cache"

    @pytest.fixture
    def cache(self, cache_dir):
        """Create CompletionCache instance with temp directory."""
        return CompletionCache(cache_dir)

    def test_init_creates_cache_directory(self, cache_dir):
        """Test that cache initialization creates cache directory."""
        cache = CompletionCache(cache_dir)
        assert cache.cache_dir.exists()
        assert cache.cache_dir.is_dir()

    def test_default_cache_location(self):
        """Test that default cache location is user home."""
        cache = CompletionCache()
        assert cache.cache_dir.parent.name == ".xraylabtool"

    def test_timeout_defaults(self, cache):
        """Test that cache timeout defaults are set."""
        assert cache.default_timeout == 3600
        assert cache.command_cache_timeout == 86400
        assert cache.env_cache_timeout == 1800

    def test_get_cache_key_string(self, cache):
        """Test cache key generation for string data."""
        key = cache.get_cache_key("test_data")
        assert isinstance(key, str)
        assert len(key) > 0

    def test_get_cache_key_dict(self, cache):
        """Test cache key generation for dict data."""
        data = {"key": "value", "number": 42}
        key = cache.get_cache_key(data)
        assert isinstance(key, str)
        assert len(key) > 0

    def test_get_cache_key_deterministic(self, cache):
        """Test that cache key generation is deterministic."""
        data = {"test": "value"}
        key1 = cache.get_cache_key(data)
        key2 = cache.get_cache_key(data)
        assert key1 == key2

    def test_get_cache_key_different_for_different_data(self, cache):
        """Test that different data produces different keys."""
        key1 = cache.get_cache_key("data1")
        key2 = cache.get_cache_key("data2")
        assert key1 != key2

    def test_set_get_basic(self, cache):
        """Test basic set and get operations."""
        key = "test_key"
        value = {"data": "value"}

        cache.set(key, value)
        result = cache.get(key)

        assert result is not None
        assert result == value

    def test_get_nonexistent_key(self, cache):
        """Test getting non-existent key returns None."""
        result = cache.get("nonexistent")
        assert result is None

    def test_set_overwrites_existing(self, cache):
        """Test that set overwrites existing values."""
        key = "test_key"
        cache.set(key, "value1")
        cache.set(key, "value2")

        result = cache.get(key)
        assert result == "value2"

    def test_cache_with_timeout(self, cache):
        """Test that cache respects timeout."""
        key = "test_key"
        value = {"data": "value"}

        # Set with short timeout and get with timeout parameter
        cache.set(key, value)

        # Should be available immediately
        assert cache.get(key, timeout=3600) == value

        # Test with very short timeout - should be expired
        cache.set(key, value)
        time.sleep(0.1)
        result = cache.get(key, timeout=0.05)
        assert result is None  # Should be expired

    def test_clear_cache(self, cache):
        """Test clearing the cache."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_invalidate_single_key(self, cache):
        """Test invalidating a single cache key."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.invalidate("key1")

        assert cache.get("key1") is None
        assert cache.get("key2", timeout=3600) == "value2"

    def test_cache_with_json_serializable_data(self, cache):
        """Test caching JSON-serializable data."""
        data = {
            "list": [1, 2, 3],
            "dict": {"nested": True},
            "string": "value",
            "number": 42,
        }

        cache.set("json_data", data)
        result = cache.get("json_data")

        assert result == data

    def test_cache_persistence(self, cache_dir):
        """Test that cache persists between instances."""
        # Create and set value
        cache1 = CompletionCache(cache_dir)
        cache1.set("persistent", "value")

        # Create new instance with same directory
        cache2 = CompletionCache(cache_dir)
        result = cache2.get("persistent", timeout=3600)

        # Value should persist
        assert result == "value"  # Depends on implementation

    def test_cache_size_limits(self, cache):
        """Test cache behavior with large number of entries."""
        # Add many entries
        for i in range(100):
            cache.set(f"key_{i}", {"value": i})

        # All should be retrievable
        result = cache.get("key_0", timeout=3600)
        assert result == {"value": 0}

        # Test stats to ensure all entries exist
        stats = cache.get_cache_stats()
        assert stats["file_count"] == 100


class TestCompletionCacheIntegration:
    """Integration tests for cache with completion system."""

    def test_cache_different_timeouts(self, tmp_path):
        """Test cache respects different timeout types."""
        cache = CompletionCache(tmp_path / "cache")

        # Test with different timeout settings - use get() with timeout param
        cache.set("cmd", {"cmd": "calc"})
        cache.set("env", {"env": "test"})

        # Both should be available with appropriate timeouts
        assert cache.get("cmd", timeout=cache.command_cache_timeout) is not None
        assert cache.get("env", timeout=cache.env_cache_timeout) is not None

    def test_cache_with_list_data(self, tmp_path):
        """Test caching list data."""
        cache = CompletionCache(tmp_path / "cache")

        data = ["calc", "batch", "convert", "list"]
        cache.set("commands", data)

        result = cache.get("commands", timeout=3600)
        assert result == data
