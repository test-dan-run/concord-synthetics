"""The data contract: one Pydantic model per file in runs/<run_id>/ (§3.2)."""

from .io import read_model, write_model
from .labels import ManifestEntry, Timeline, Transcript
from .meeting import MeetingSpec
from .models import MeetingStatus, Script, ScriptValidation, SpeakerSplits, StageStatus, Turn
from .paths import MeetingPaths, RunPaths
from .voice import Synthesis, VoiceValidation

__all__ = ["ManifestEntry", "MeetingPaths", "MeetingSpec", "MeetingStatus", "RunPaths", "Script", "ScriptValidation",
           "SpeakerSplits", "StageStatus", "Synthesis", "Timeline", "Transcript", "Turn", "VoiceValidation",
           "read_model", "write_model"]
