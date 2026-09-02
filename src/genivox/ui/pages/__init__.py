"""Page widgets used by :class:`genivox.ui.MainWindow`."""

from genivox.ui.pages.experiments import ExperimentPage
from genivox.ui.pages.languages import MultilingualPage
from genivox.ui.pages.models import ModelManagerPage
from genivox.ui.pages.overview import OverviewPage
from genivox.ui.pages.synthesis import SynthesisPage
from genivox.ui.pages.training import TrainingPage
from genivox.ui.pages.voice_profile import VoiceProfilePage

__all__ = [
    "ExperimentPage",
    "ModelManagerPage",
    "MultilingualPage",
    "OverviewPage",
    "SynthesisPage",
    "TrainingPage",
    "VoiceProfilePage",
]
