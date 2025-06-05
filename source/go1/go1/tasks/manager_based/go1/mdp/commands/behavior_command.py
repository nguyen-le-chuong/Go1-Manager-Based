# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommand, NUniformVelocityCommandCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import BehaviorCommandCfg


class BehaviorCommand(UniformVelocityCommand):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: BehaviorCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: BehaviorCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # check configuration
        self.body_height = torch.zeros(self.num_envs, device=self.device)
        self.gait_frequency = torch.zeros(self.num_envs, device=self.device)
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)
        self.gait_offset = torch.zeros(self.num_envs, device=self.device)
        self.gait_bound = torch.zeros(self.num_envs, device=self.device)
        self.gait_duration = torch.zeros(self.num_envs, device=self.device)
        self.footswing_height = torch.zeros(self.num_envs, device=self.device)
        self.body_pitch = torch.zeros(self.num_envs, device=self.device)
        self.body_roll = torch.zeros(self.num_envs, device=self.device)
        self.stand_width = torch.zeros(self.num_envs, device=self.device)
        self.stand_length = torch.zeros(self.num_envs, device=self.device)
        self.aux_reward_coeff = torch.zeros(self.num_envs, device=self.device)

        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_gait_frequency"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_gait_phase"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_gait_offset"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_gait_bound"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_gait_duration"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_footswing_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pitch"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_roll"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_stand_width"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_stand_length"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "NormalVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    def _resample_command(self, env_ids):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.body_height[env_ids] = r.uniform_(*self.cfg.ranges.body_height)
        self.gait_frequency[env_ids] = r.uniform_(*self.cfg.ranges.gait_frequency)
        self.gait_phase[env_ids] = r.uniform_(*self.cfg.ranges.gait_phase)
        self.gait_offset[env_ids] = r.uniform_(*self.cfg.ranges.gait_offset)
        self.gait_bound[env_ids] = r.uniform_(*self.cfg.ranges.gait_bound)
        self.gait_duration[env_ids] = r.uniform_(*self.cfg.ranges.gai_duration)
        self.footswing_height[env_ids] = r.uniform_(*self.cfg.ranges.footswing_height)
        self.body_pitch[env_ids] = r.uniform_(*self.cfg.ranges.body_pitch)
        self.body_roll[env_ids] = r.uniform_(*self.cfg.ranges.body_roll)
        self.stand_width[env_ids] = r.uniform_(*self.cfg.ranges.stance_width)
        self.stand_length[env_ids] = r.uniform_(*self.cfg.ranges.stance_length)
        self.aux_reward_coeff[env_ids] = r.uniform_(*self.cfg.ranges.aux_vel_x)


    def _update_command(self):
        """Sets velocity command to zero for standing envs."""
        # Enforce standing (i.e., zero velocity command) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.body_height[standing_env_ids] = 0.0
        self.gait_frequency[standing_env_ids] = 0.0
        self.gait_phase[standing_env_ids] = 0.0
        self.gait_offset[standing_env_ids] = 0.0
        self.gait_bound[standing_env_ids] = 0.0
        self.gait_duration[standing_env_ids] = 0.0
        self.footswing_height[standing_env_ids] = 0.0
        self.body_pitch[standing_env_ids] = 0.0
        self.body_roll[standing_env_ids] = 0.0
        self.stand_width[standing_env_ids] = 0.0
        self.stand_length[standing_env_ids] = 0.0
        self.aux_reward_coeff[standing_env_ids] = 0.0
        # update the command
