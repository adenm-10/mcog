# domains/contact/sac_clipped.py
"""SAC with the critic's TD target clipped to the analytically valid range.

Standard practice in HER implementations -- OpenAI baselines' `ddpg.py` does
`clip_by_value(r + gamma * target_Q, -clip_return, 0)` with
`clip_return = 1/(1-gamma)` for its `r in {-1, 0}` convention. Our bound is
100x tighter because ARRIVAL TERMINATES the episode here (gym_env.py's
`terminated = bool(arr.reached_interface) or ...`) while the Fetch tasks
baselines targets keep paying +1 every step you stay in tolerance. With a
one-shot terminal bonus and every other reward weight zeroed, `step_reward`
reduces to `goal_reward * arrived`, so Q* <= goal_reward exactly -- against
goal_reward/(1-gamma) = 1000 for the loose form, which would not have fired on
the divergence this exists to stop (worst observed Q was 539).

This changes no reward, only the regression target, so the pure-sparse setup is
preserved. `train()` is copied from SB3 2.9.0's SAC with one clamp added, the
same audit-against-the-installed-source pattern as her_buffer.py.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F


class TargetClippedSAC(SAC):
    """SAC whose TD target is clamped to [0, target_clip]. `None` disables the
    clamp, leaving `train()` numerically identical to SB3's."""

    def __init__(self, *args, target_clip: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_clip = None if target_clip is None else float(target_clip)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        clip_fracs = []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
                # THE ONE ADDED LINE. Clamped after the entropy term so the
                # bound holds on the quantity the critic actually regresses on.
                if self.target_clip is not None:
                    clip_fracs.append(
                        (target_q_values > self.target_clip).float().mean().item())
                    target_q_values = th.clamp(target_q_values, 0.0, self.target_clip)

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        # Fraction of targets the clamp actually bit on -- 0.0 throughout means
        # the run never needed it, which is itself the result.
        if clip_fracs:
            self.logger.record("train/target_clip_frac", np.mean(clip_fracs))

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params()
