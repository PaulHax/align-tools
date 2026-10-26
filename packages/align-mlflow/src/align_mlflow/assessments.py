"""Identify feedback recorded by the episode score importer."""

import math
from typing import TypeGuard

from align_utils.open_world import RAW_LOG_FILE
from mlflow.entities import Assessment, Feedback

from .traces import SCORE_ASSESSMENT, SESSION_METADATA_KEY


def is_recorded_session_score(
    assessment: Assessment, session: str
) -> TypeGuard[Feedback]:
    """Match source feedback, including invalidated feedback retained as a tombstone."""
    return (
        isinstance(assessment, Feedback)
        and assessment.name == SCORE_ASSESSMENT
        and assessment.source.source_type == "CODE"
        and assessment.source.source_id == RAW_LOG_FILE
        and (assessment.metadata or {}).get(SESSION_METADATA_KEY) == session
        and not isinstance(assessment.value, bool)
        and isinstance(assessment.value, (int, float))
        and math.isfinite(assessment.value)
    )
