import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .behavior_command import BehaviorCommand

class BehaviorCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the normal velocity command generator."""

    class_type: type = BehaviorCommand
    heading_command: bool = False  # --> we don't use heading command for normal velocity command.
    exclude_phase_offset: bool = True
    binary_phase: bool = False
    pacing_offset: bool = False
    balance_gait_distibution: bool = True
    gaitwise_curicula: bool = True
    @configclass
    class Ranges:
        body_height: tuple[float, float] = [0.0, 0.01] 
        gait_phase: tuple[float, float] = [0.0, 0.01]
        gait_frequency: tuple[float, float] = [2.0, 2.01]  
        gait_offset: tuple[float, float] = [0.0, 0.01] 
        gait_bound: tuple[float, float] = [0.0, 0.01]
        gai_duration: tuple[float, float] = [0.49, 0.5]
        footswing_height: tuple[float, float] = [0.06, 0.061]
        body_pitch: tuple[float, float] = [0.0, 0.01]
        body_roll: tuple[float, float] = [0.0, 0.01]
        aux_vel_x: tuple[float, float] = [0.0, 0.01]
        stance_width: tuple[float, float] = [0.0, 0.01]
        stance_length: tuple[float, float] = [0.0, 0.01]

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""