"""Tests specifically for KDMA parsing functionality."""

import pytest
from align_utils.models import parse_alignment_target_id


def test_parse_alignment_target_id_single_kdma():
    """Test parsing single KDMA alignment targets."""

    # Standard format
    result = parse_alignment_target_id("ADEPT-June2025-merit-0.0")
    assert len(result) == 1
    assert result[0].kdma == "merit"
    assert result[0].value == 0.0

    # KDMA with underscore
    result = parse_alignment_target_id("ADEPT-June2025-personal_safety-0.5")
    assert len(result) == 1
    assert result[0].kdma == "personal_safety"
    assert result[0].value == 0.5

    # Short format (directory name style)
    result = parse_alignment_target_id("personal_safety-0.0")
    assert len(result) == 1
    assert result[0].kdma == "personal_safety"
    assert result[0].value == 0.0

    # Another standard case
    result = parse_alignment_target_id("ADEPT-June2025-affiliation-1.0")
    assert len(result) == 1
    assert result[0].kdma == "affiliation"
    assert result[0].value == 1.0


def test_parse_alignment_target_id_multi_kdma():
    """Test parsing multi-KDMA alignment targets."""

    # Two KDMAs without underscores in names
    result = parse_alignment_target_id("ADEPT-June2025-affiliation_merit-0.0_0.5")
    assert len(result) == 2

    # Sort by KDMA name for consistent comparison
    result = sorted(result, key=lambda x: x.kdma)
    assert result[0].kdma == "affiliation"
    assert result[0].value == 0.0
    assert result[1].kdma == "merit"
    assert result[1].value == 0.5

    # Three values
    result = parse_alignment_target_id("ADEPT-June2025-a_b_c-0.1_0.2_0.3")
    assert len(result) == 3
    result = sorted(result, key=lambda x: x.kdma)
    assert result[0].kdma == "a"
    assert result[0].value == 0.1
    assert result[1].kdma == "b"
    assert result[1].value == 0.2
    assert result[2].kdma == "c"
    assert result[2].value == 0.3


def test_parse_alignment_target_id_edge_cases():
    """Test edge cases and invalid inputs."""

    # Unaligned
    result = parse_alignment_target_id("unaligned")
    assert result == []

    # Empty string
    result = parse_alignment_target_id("")
    assert result == []

    # None
    result = parse_alignment_target_id(None)
    assert result == []

    # Invalid format - no hyphens
    result = parse_alignment_target_id("merit0.0")
    assert result == []

    # Invalid format - not enough parts
    result = parse_alignment_target_id("merit")
    assert result == []

    # Invalid format - missing value
    result = parse_alignment_target_id("ADEPT-June2025-merit")
    assert result == []

    # Invalid format - bad float value
    result = parse_alignment_target_id("ADEPT-June2025-merit-invalid")
    assert result == []

    # Mismatched KDMA names and values count (should not happen with new logic for single values)
    # This case would be handled properly now since single values treat the whole kdma_part as one name


def test_parse_alignment_target_id_regression_personal_safety():
    """Regression test for the specific personal_safety bug."""

    # This was the main failing case
    result = parse_alignment_target_id("personal_safety-0.0")
    assert len(result) == 1
    assert result[0].kdma == "personal_safety"
    assert result[0].value == 0.0

    # Longer format that was also failing
    result = parse_alignment_target_id("ADEPT-June2025-personal_safety-0.0")
    assert len(result) == 1
    assert result[0].kdma == "personal_safety"
    assert result[0].value == 0.0

    # Ensure other underscore-containing KDMA names work
    result = parse_alignment_target_id("some_other_kdma-1.0")
    assert len(result) == 1
    assert result[0].kdma == "some_other_kdma"
    assert result[0].value == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
