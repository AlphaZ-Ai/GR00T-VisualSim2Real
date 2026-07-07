"""GRPO fine-tuning trainer for the vision (student) policy.

Implements the GRPO-based sim-to-real fine-tuning stage from
"Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer"
(Xue et al., arXiv:2512.01061) -- the third stage after (1) the privileged
teacher (PPO + staged-reset exploration) and (2) vision distillation (DAgger).

Paper formulation
-----------------
A batch of G rollouts {tau_i} is sampled from the current student policy pi_S,
each with a scalar trajectory return R_i (mainly binary task success plus light
behavior regularizers).

  Eq. (4)  group-relative advantage:
      A_hat_i = (R_i - mean(R)) / std(R)

  Eq. (5)  clipped surrogate, critic-free:
      L_GRPO(theta) = E_{i,t}[ min( r_{i,t} A_hat_i,
                                    clip(r_{i,t}, 1-eps, 1+eps) A_hat_i ) ]
      with r_{i,t} = pi_theta(a_{i,t}|o_{i,t}) / pi_old(a_{i,t}|o_{i,t}).

The clipped surrogate in TRLPPOTrainer._compute_ppo_loss already *is* Eq. (5)
(it multiplies the per-step ratio by the stored advantage and takes the clipped
min), so this subclass supplies the advantages and runs critic-free
(`value_model=None`, `algo.config.vf_coef: 0.0`).

How this maps onto the windowed rollout trainer
-----------------------------------------------
The paper's "G rollouts" are complete trajectories. This trainer collects
fixed-length windows of `num_steps_per_env` steps from `num_envs` parallel
envs, and episodes (~O(1000) steps) may span several windows. We therefore
treat each *completed episode* as one rollout tau_i:

  * R_i is the trajectory return of episode i, summed over the WHOLE episode.
    A per-env running sum is carried across window boundaries, so an episode
    that started in a previous window still gets its exact full return when it
    terminates. (gamma is configurable; the paper's trajectory return is
    undiscounted -> set `algo.config.gamma: 1.0`.)

  * The group for Eq. (4) is exactly the set of episodes that COMPLETED in the
    current batch (gathered across processes). This keeps the baseline fully
    on-policy: no rolling buffer of returns from stale policy iterations.

  * Steps whose episode has NOT yet terminated by the end of the window get
    ZERO advantage: they contribute nothing to the policy gradient this update
    (their episode's steps in the *next* window will carry the signal when it
    completes). This trades some sample efficiency for correctness -- no
    partial-return guesses enter Eq. (4).

  * Degenerate groups are a no-op by construction: if fewer than
    `_GRPO_MIN_GROUP` episodes completed, or std(R) ~ 0 (e.g. every episode
    failed identically), advantages are zeroed and a warning is logged.
    Exploration noise (`init_noise_std`) must be tuned so groups contain mixed
    outcomes -- GRPO has NO learning signal without within-group variance.

CRITICAL integration note (the bug that broke the previous attempt): the base
`_rollout_step` only computes returns/advantages inside its trailing
`if self.value_model is not None:` block. Critic-free GRPO runs with
`value_model=None`, so without the `_rollout_step` override below the
advantages buffer would silently stay all-zero and no learning would happen.
Telemetry is logged every rollout precisely so that failure mode can never
hide again.
"""

import copy

import torch
from loguru import logger

from gr00t.rl.trl.trainer.ppo_trainer_homie_api import TRLPPOTrainer


class TRLGRPOTrainerHomieAPI(TRLPPOTrainer):
    """Critic-free GRPO fine-tuning of the vision student (paper Eq. 4-5)."""

    # Minimum completed episodes for a usable group baseline (Eq. 4 needs a std).
    _GRPO_MIN_GROUP = 2
    # Below this, std(R) is considered degenerate (all outcomes identical).
    _GRPO_STD_EPS = 1e-6

    # ------------------------------------------------------------------ #
    # Checkpoint: distilled weights + FRESH exploration noise.           #
    # ------------------------------------------------------------------ #
    def load_checkpoint(self, checkpoint_path):
        """Load the distilled student checkpoint, then re-arm exploration noise.

        The actor's noise std is an nn.Parameter inside `policy_state_dict`, so
        loading a checkpoint distilled with a near-zero std (e.g. 0.001)
        silently overwrites the configured `init_noise_std` -- and
        `clamp_(max=max_noise_std)` only ever lowers it. Near-deterministic
        rollouts make every episode in a group end with (almost) the same
        return, so std(R) -> 0 and the GRPO gradient vanishes. Reset the std to
        the config value after loading: for GRPO the checkpoint provides the
        *weights*, the config owns the *exploration*.
        """
        checkpoint = super().load_checkpoint(checkpoint_path)
        init_noise_std = self.config.get("init_noise_std", None)
        policy = self.accelerator.unwrap_model(self.model).policy
        if init_noise_std is not None and hasattr(policy, "std"):
            with torch.no_grad():
                old = policy.std.detach().clone()
                # Per-dimension exploration noise. GRPO needs the return variance
                # within a group to come from the POLICY'S OWN action choices, not
                # environment randomness -- otherwise it credits actions for luck
                # and degrades the policy. But uniform noise on all dims makes the
                # near-deterministic distilled policy FALL: the noise lands on the
                # HOMIE locomotion-command dims (the first `homie_command_dim`
                # velocity commands driving the walking controller). So we split:
                #   * locomotion command dims -> `noise_std_locomotion` (tiny; a
                #     valid but near-deterministic distribution -> no fall-inducing
                #     gait perturbation). Must be > 0 (std=0 -> undefined log_prob).
                #   * everything else (upper body + fingers) -> `init_noise_std`
                #     (real manipulation exploration for GRPO's group contrast).
                n_loco = int(self.config.get("homie_command_dim", 0) or 0)
                loco_std = float(self.config.get("noise_std_locomotion", init_noise_std))
                std_vec = torch.full_like(policy.std.data, float(init_noise_std))
                if n_loco > 0:
                    std_vec[:n_loco] = loco_std
                policy.std.data[:] = std_vec
            logger.info(
                f"GRPO: reset actor noise std (checkpoint mean {old.mean().item():.4f}) "
                f"to per-dim [locomotion({n_loco})={loco_std}, manipulation="
                f"{init_noise_std}] for group exploration."
            )

        # Build the frozen KL-reference (the distilled student BEFORE fine-tuning).
        self._build_grpo_reference()
        return checkpoint

    # ------------------------------------------------------------------ #
    # KL-to-distilled-student anchor.                                     #
    # ------------------------------------------------------------------ #
    def _build_grpo_reference(self):
        """Snapshot the just-loaded policy as a frozen reference pi_ref.

        GRPO fine-tuning can drift a 32%-success policy into a 0% one. An
        analytic-Gaussian KL penalty D_KL(pi_theta || pi_ref) added to the loss
        keeps the policy anchored near the distilled student. pi_ref is a frozen
        deep copy of the *whole* wrapper (so it runs through the identical
        `_forward_model` path -- vision split/unsplit, homie concat), with its
        value model dropped and all parameters frozen. Skipped entirely when
        kl_ref_coef == 0 (no reference network -> no extra memory).
        """
        self._kl_ref_coef = float(self.config.get("kl_ref_coef", 0.0) or 0.0)
        self._grpo_ref_model = None
        if self._kl_ref_coef <= 0.0:
            logger.info("GRPO: kl_ref_coef=0 -> no KL-to-reference anchor.")
            return
        base = self.accelerator.unwrap_model(self.model)
        ref = copy.deepcopy(base)
        ref.value_model = None            # critic-free; ref only needs the actor
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        self._grpo_ref_model = ref
        n = sum(p.numel() for p in ref.policy.parameters())
        logger.info(
            f"GRPO: built frozen KL-reference (pi_ref) from the distilled student "
            f"({n/1e6:.1f}M actor params); kl_ref_coef={self._kl_ref_coef}."
        )

    def _kl_to_reference(self, forward_results, mb_rollout_data):
        """KL(pi_theta || pi_ref) as a mean-space anchor (padding-masked).

        pi_ref is forwarded on the SAME minibatch under no_grad, reusing the base
        `_forward_model` so its action *mean* is produced by the identical
        recurrent/vision path as the trainable policy.

        The reference width in the KL is a FIXED `kl_ref_sigma`, not the actor's
        own noise std: the distilled/re-armed std is ~1e-3, and a Gaussian KL with
        that in the denominator explodes (a 0.01 mean deviation -> KL ~50/dim),
        hard-freezing the policy. With a fixed width the KL reduces to a smooth,
        well-scaled anchor on the mean action: ||mu_new - mu_ref||^2 / (2 s^2),
        summed over the policy's own action dims (homie dims are frozen/shared).
        """
        policy_results = forward_results["policy_results"]
        mu_new = policy_results["action_mean"]

        with torch.no_grad():
            ref_results = self._forward_model(self._grpo_ref_model, mb_rollout_data)
        mu_ref = ref_results["policy_results"]["action_mean"].detach()

        na = getattr(self.policy_model, "num_actions", None)
        if na is not None:
            mu_new = mu_new[..., :na]
            mu_ref = mu_ref[..., :na]

        s = float(self.config.get("kl_ref_sigma", 0.1))
        kl = ((mu_new - mu_ref) ** 2).sum(dim=-1) / (2.0 * s * s)

        pad = mb_rollout_data.get("mb_padding_mask", None)
        if pad is not None and pad.shape == kl.shape:
            valid = (~pad).float()
            return (kl * valid).sum() / valid.sum().clamp(min=1)
        return kl.mean()

    def _compute_loss(self, forward_results, mb_rollout_data):
        ret = super()._compute_loss(forward_results, mb_rollout_data)
        if getattr(self, "_grpo_ref_model", None) is not None and self._kl_ref_coef > 0.0:
            kl_ref = self._kl_to_reference(forward_results, mb_rollout_data)
            ret["loss"] = ret["loss"] + self._kl_ref_coef * kl_ref
            ret["kl_ref"] = kl_ref.detach()
            self.grpo_kl_ref = float(kl_ref.detach().item())
        return ret

    # ------------------------------------------------------------------ #
    # Rollout hook: the base only fills advantages when a critic exists. #
    # ------------------------------------------------------------------ #
    def _rollout_step(self, model, obs_dict):
        """Collect a rollout, then compute group-relative advantages.

        The base `_rollout_step` gates its returns/advantages computation (and
        the `_compute_returns` call) behind `if self.value_model is not None:`
        because GAE needs a critic. Critic-free GRPO must fill the advantage
        storage itself, otherwise the PPO surrogate multiplies the ratios by
        the zero-initialized buffer and the whole run is a silent no-op.
        """
        obs_dict = super()._rollout_step(model, obs_dict)
        if self.value_model is None:
            returns, advantages = self._compute_returns(
                values=None,
                last_values=None,
                policy_state_dict={
                    "dones": self.storage.query_key("dones"),
                    "rewards": self.storage.query_key("rewards"),
                },
            )
            self.storage.batch_update_data("returns", returns)
            self.storage.batch_update_data("advantages", advantages)
        return obs_dict

    # ------------------------------------------------------------------ #
    # Episode segmentation with exact cross-window return carry.         #
    # ------------------------------------------------------------------ #
    def _grpo_episode_segments(self, rewards, dones):
        """Segment the window into episodes and accumulate their returns.

        rewards, dones: [T, N] (window steps x envs).

        Returns:
          segments: list of (env, start, end, R) for episodes that TERMINATED
                    in this window. `start` is the episode's first step inside
                    this window (0 if it began in an earlier window); `end` is
                    the terminal step (inclusive). R is the return of the WHOLE
                    episode, including reward carried from earlier windows.
        Persistent state carries each env's running (discounted) reward sum and
        discount factor across window boundaries.
        """
        T, N = rewards.shape
        device = rewards.device
        gamma = self.gamma

        # Lazy init / re-init if env count changes (e.g. resumed run).
        if getattr(self, "_grpo_run_return", None) is None or self._grpo_run_return.shape[0] != N:
            self._grpo_run_return = torch.zeros(N, device=device)
            self._grpo_run_discount = torch.ones(N, device=device)

        run = self._grpo_run_return.to(device)
        disc = self._grpo_run_discount.to(device)
        seg_start = torch.zeros(N, dtype=torch.long, device=device)  # episode start in window

        segments = []
        for t in range(T):
            run = run + disc * rewards[t]
            disc = disc * gamma
            done_t = dones[t].bool()
            if done_t.any():
                for n in torch.nonzero(done_t, as_tuple=False).flatten().tolist():
                    segments.append((n, int(seg_start[n].item()), t, run[n].detach()))
                run = torch.where(done_t, torch.zeros_like(run), run)
                disc = torch.where(done_t, torch.ones_like(disc), disc)
                seg_start = torch.where(
                    done_t, torch.full_like(seg_start, t + 1), seg_start
                )

        # Carry in-progress episodes into the next window.
        self._grpo_run_return = run.detach()
        self._grpo_run_discount = disc.detach()
        return segments

    def _grpo_gather_group(self, completed_returns):
        """Gather the group of completed-episode returns across processes.

        `accelerator.gather` needs equal shapes per rank, but ranks complete
        different numbers of episodes -- pad to the max count and mask.
        Single-process runs (the normal case here) skip the collective.
        """
        device = self.accelerator.device
        local = (
            torch.stack(completed_returns).to(device)
            if completed_returns
            else torch.empty(0, device=device)
        )
        if self.accelerator.num_processes == 1 or not self.sync_advantage_normalization:
            return local

        count = torch.tensor([local.numel()], device=device)
        counts = self.accelerator.gather(count)
        max_count = int(counts.max().item())
        if max_count == 0:
            return torch.empty(0, device=device)
        padded = torch.zeros(max_count, device=device)
        padded[: local.numel()] = local
        gathered = self.accelerator.gather(padded).reshape(-1, max_count)
        return torch.cat([gathered[i, : int(c.item())] for i, c in enumerate(counts)])

    # ------------------------------------------------------------------ #
    # Eq. (4): group-relative advantage over completed episodes.         #
    # ------------------------------------------------------------------ #
    def _compute_returns(self, values, last_values, policy_state_dict):
        """Group-relative, per-episode advantage (paper Eq. 4); critic-free.

        Each step of a completed episode i receives the constant advantage
        A_hat_i = (R_i - mean(R)) / std(R), computed over the group of episodes
        that completed in this batch. Steps of unfinished episodes get zero
        advantage. Returns (returns, advantages) shaped [T, N, 1] -- the
        storage/minibatch contract (`register_key(..., shape=(1,))`).
        """
        device = self.accelerator.device
        rewards = policy_state_dict["rewards"].to(device)  # [T, N, 1]
        dones = policy_state_dict["dones"].to(device)      # [T, N, 1]

        # When a critic exists, the base rollout loop has already added
        # `gamma * time_outs * V(s)` into the stored reward (truncation
        # bootstrap). GRPO is critic-free; if a critic is nevertheless present,
        # undo that untrained-critic noise to recover the raw env reward.
        if values is not None:
            time_outs = self.storage.query_key("time_outs").to(device)
            rewards = rewards - self.gamma * time_outs * values

        rewards = rewards.squeeze(-1).float()  # [T, N]
        dones = dones.squeeze(-1)              # [T, N]
        T, N = rewards.shape

        segments = self._grpo_episode_segments(rewards, dones)
        group = self._grpo_gather_group([seg[3] for seg in segments])

        returns = torch.zeros(T, N, device=device)
        advantages = torch.zeros(T, N, device=device)

        G = group.numel()
        if G >= self._GRPO_MIN_GROUP:
            mean = group.mean()
            std = group.std()
            if std > self._GRPO_STD_EPS:
                for n, s, e, R in segments:
                    returns[s : e + 1, n] = R
                    advantages[s : e + 1, n] = (R - mean) / (std + 1e-8)
            else:
                logger.warning(
                    f"GRPO: degenerate group (G={G}, std={std.item():.2e} ~ 0; all episode "
                    "returns identical) -- zero advantages this update. Increase exploration "
                    "noise (init_noise_std) so groups contain mixed outcomes."
                )
        else:
            logger.warning(
                f"GRPO: only {G} episode(s) completed in this batch (< {self._GRPO_MIN_GROUP}) "
                "-- zero advantages this update. Increase num_steps_per_env or num_envs."
            )

        # --- telemetry: make a silent no-op impossible to miss ---
        nonzero_frac = (advantages != 0).float().mean().item()
        logger.info(
            "GRPO group: G={} | R mean={:.3f} std={:.3f} | steps w/ signal: {:.1%}".format(
                G,
                group.mean().item() if G > 0 else float("nan"),
                group.std().item() if G > 1 else float("nan"),
                nonzero_frac,
            )
        )
        self.grpo_stats = {
            "grpo/group_size": float(G),
            "grpo/return_mean": group.mean().item() if G > 0 else 0.0,
            "grpo/return_std": group.std().item() if G > 1 else 0.0,
            "grpo/signal_step_frac": nonzero_frac,
        }

        return returns.unsqueeze(-1), advantages.unsqueeze(-1)
