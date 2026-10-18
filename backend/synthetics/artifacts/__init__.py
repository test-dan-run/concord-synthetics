"""The data contract: one Pydantic model per file in runs/<run_id>/ (§3.2)."""

from .io import read_model, write_model
from .meeting import MeetingSpec
from .models import MeetingStatus, Script, ScriptValidation, SpeakerSplits, StageStatus, Turn
from .paths import MeetingPaths, RunPaths

__all__ = ["MeetingPaths", "MeetingSpec", "MeetingStatus", "RunPaths", "Script", "ScriptValidation", "SpeakerSplits",
           "StageStatus", "Turn", "read_model", "write_model"]
