"""
rl_trainer.py — Two-Stage Reranker Trainer:
  Stage 1: Oracle Imitation Pre-training (ListMLE loss against Greedy Submodular Oracle)
  Stage 2: Budget-Marginal REINFORCE with Reward-to-Go Policy Gradient & Self-Critical Baseline

Evaluates candidates using Contextual Sequential MMR (redundancy & token budget aware).
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim

from method.config import (
    CHECKPOINTS_DIR, DEVICE,
    RL_BASELINE_MOMENTUM, RL_GRAD_CLIP, RL_LR,
    RL_NUM_EPOCHS, RL_SAMPLES_K, RL_TEMP_END, RL_TEMP_START,
    RL_WEIGHT_DECAY, IMITATION_EPOCHS, IMITATION_LR
)
from method.reranker import RLReranker, plackett_luce_sample
from method.oracle import compute_greedy_oracle_sequence
from score import score_query, BUDGETS, EQUAL_MIN_COST as MIN_COST


class RLTrainer:
    """
    Two-Stage Trainer:
      1. ListMLE Imitation Pre-training on Greedy Oracle trajectories
      2. Budget-Marginal REINFORCE Fine-tuning
    """

    def __init__(
        self,
        model: RLReranker,
        lr: float = RL_LR,
        weight_decay: float = RL_WEIGHT_DECAY,
        baseline_momentum: float = RL_BASELINE_MOMENTUM,
        grad_clip: float = RL_GRAD_CLIP,
        checkpoint_dir: Path = CHECKPOINTS_DIR,
    ):
        self.model = model.to(DEVICE)
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.baseline_momentum = baseline_momentum
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _temperature(self, epoch: int, total_epochs: int) -> float:
        """Cosine annealing from RL_TEMP_START to RL_TEMP_END."""
        if total_epochs <= 1:
            return RL_TEMP_START
        frac = epoch / (total_epochs - 1)
        return RL_TEMP_END + (RL_TEMP_START - RL_TEMP_END) * (1 - frac)

    @staticmethod
    def compute_marginal_rewards(ranked_ids, answer_entry, tokens_map, k_trunc):
        """
        Compute per-position marginal nugget contribution across all budget windows.
        Returns: marginal_rewards (k_trunc,) float32
        """
        marginals = np.zeros(k_trunc, dtype=np.float32)
        nuggets = answer_entry["nuggets"]

        for B in BUDGETS:
            cumulative_cost = 0
            prev_nugget_best = {nid: 0.0 for nid in nuggets}

            # Compute oracle reach for this budget
            reach = sum(
                max((v for e, v in vals.items()
                     if max(tokens_map.get(e, MIN_COST), MIN_COST) <= B), default=0.0)
                for vals in nuggets.values()
            )
            if reach == 0:
                continue

            for j in range(k_trunc):
                eid = ranked_ids[j]
                cost = max(tokens_map.get(eid, MIN_COST), MIN_COST)
                if cumulative_cost + cost > B:
                    continue  # this item doesn't fit in this budget

                cumulative_cost += cost
                # Compute marginal nugget improvement from this item
                delta = 0.0
                for nid, vals in nuggets.items():
                    new_val = vals.get(eid, 0.0)
                    if new_val > prev_nugget_best[nid]:
                        delta += (new_val - prev_nugget_best[nid])
                        prev_nugget_best[nid] = new_val

                marginals[j] += delta / (reach * len(BUDGETS))

        return marginals

    def train_imitation_epoch(self, train_data: list, optimizer: optim.Optimizer) -> float:
        """
        One epoch of ListMLE imitation pre-training on Greedy Oracle trajectories.
        Supervises the base relevance head directly against the optimal item sequence.
        """
        self.model.train()
        total_loss = 0.0
        n_questions = 0

        optimizer.zero_grad()
        accum_steps = 8

        for step_idx, item in enumerate(train_data):
            oracle_indices = item.get("oracle_indices")
            if not oracle_indices:
                # Dynamically compute if not precomputed
                bids = item["block_ids"]
                entry = item["answer_entry"]
                tokens_map = item["tokens_map"]
                o_ids, _ = compute_greedy_oracle_sequence(entry, bids, tokens_map, max_items=25)
                oracle_indices = [bids.index(b) for b in o_ids if b in bids]
                item["oracle_indices"] = oracle_indices

            if not oracle_indices:
                continue

            q_emb = torch.tensor(item["query_emb"], dtype=torch.float32, device=DEVICE)
            b_embs = torch.tensor(item["block_embs"], dtype=torch.float32, device=DEVICE)
            s_feats = torch.tensor(item["struct_feats"], dtype=torch.float32, device=DEVICE)
            r_feats_np = item.get("retrieval_feats")
            r_feats = torch.tensor(r_feats_np, dtype=torch.float32, device=DEVICE) if r_feats_np is not None else None

            # Forward pass: relevance scores
            scores = self.model(q_emb, b_embs, s_feats, r_feats)  # (N,)

            # ListMLE loss over oracle sequence
            N = len(scores)
            mask = torch.ones(N, dtype=torch.bool, device=DEVICE)
            item_loss = 0.0
            
            # Limit supervision to top oracle items that contributed score
            for target_idx in oracle_indices[:15]:
                rem_scores = scores[mask]
                target_score = scores[target_idx]
                step_loss = -(target_score - torch.logsumexp(rem_scores, dim=0))
                item_loss = item_loss + step_loss
                mask = mask.clone()
                mask[target_idx] = False

            if isinstance(item_loss, torch.Tensor):
                (item_loss / accum_steps).backward()
                total_loss += item_loss.item()
                n_questions += 1

                if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(train_data):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

        return total_loss / max(1, n_questions)

    def pretrain_imitation(
        self,
        train_data: list,
        eval_data: Optional[list] = None,
        epochs: int = IMITATION_EPOCHS,
        lr: float = IMITATION_LR,
        save_prefix: str = "reranker",
    ) -> float:
        """
        Stage 1: Pre-train policy network on Greedy Oracle demonstrations.
        """
        print(f"\n--- Stage 1: Oracle Imitation Pre-training (ListMLE, {epochs} epochs) ---")
        imitation_opt = optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)

        best_score = -1.0
        if eval_data is not None:
            init_eval = self.evaluate(eval_data)
            best_score = init_eval["score"]
            print(f"  [Initial Baseline] eval score={best_score:.4f} above_floor={init_eval['above_floor']:+.4f}")

        for ep in range(epochs):
            t0 = time.time()
            loss = self.train_imitation_epoch(train_data, imitation_opt)
            elapsed = time.time() - t0

            if eval_data is not None and ((ep + 1) % 5 == 0 or (ep + 1) == epochs):
                ev = self.evaluate(eval_data)
                is_best = ev["score"] >= best_score
                if is_best:
                    best_score = ev["score"]
                    self.save(f"{save_prefix}_imitation_best")
                star = " [NEW BEST]" if is_best else ""
                print(f"  Imitation Epoch {ep+1:2d}/{epochs}: loss={loss:.4f} | EVAL score={ev['score']:.4f} "
                      f"above_floor={ev['above_floor']:+.4f} ({elapsed:.1f}s){star}")
            else:
                print(f"  Imitation Epoch {ep+1:2d}/{epochs}: loss={loss:.4f} ({elapsed:.1f}s)")

        # Load best imitation checkpoint
        best_im_path = self.checkpoint_dir / f"{save_prefix}_imitation_best.pt"
        if best_im_path.exists():
            self.model.load_state_dict(torch.load(best_im_path, map_location=DEVICE))
            print(f"  [✓] Loaded best imitation checkpoint (eval score: {best_score:.4f})")

        return best_score

    def train_epoch(
        self,
        train_data: list,
        epoch: int,
        total_epochs: int,
        samples_k: int = RL_SAMPLES_K,
    ) -> dict:
        """
        Stage 2: Run one epoch of budget-marginal REINFORCE fine-tuning.
        """
        self.model.train()
        tau = self._temperature(epoch, total_epochs)

        total_loss = 0.0
        total_reward = 0.0
        total_above_floor = 0.0
        n_questions = len(train_data)

        self.optimizer.zero_grad()
        accum_steps = 8

        for step_idx, item in enumerate(train_data):
            q_emb = torch.tensor(item["query_emb"], dtype=torch.float32, device=DEVICE)
            b_embs = torch.tensor(item["block_embs"], dtype=torch.float32, device=DEVICE)
            s_feats = torch.tensor(item["struct_feats"], dtype=torch.float32, device=DEVICE)
            block_ids = item["block_ids"]
            tokens_map = item["tokens_map"]
            answer_entry = item["answer_entry"]

            r_feats_np = item.get("retrieval_feats")
            r_feats = torch.tensor(r_feats_np, dtype=torch.float32, device=DEVICE) if r_feats_np is not None else None

            if len(block_ids) == 0:
                continue

            # Forward: relevance scores
            scores = self.model(q_emb, b_embs, s_feats, r_feats)  # (N,)

            # Sample K rankings via Plackett-Luce over full candidate pool (or top 100)
            k_trunc = min(80, len(block_ids))
            rewards_list = []
            G_list = []
            per_pos_lps = []

            for _ in range(samples_k):
                perm, lp, per_pos_lp = plackett_luce_sample(scores, temperature=tau, k=k_trunc)
                ranked_ids = [block_ids[i] for i in perm.detach().tolist()]

                # Compute per-position marginal rewards
                marginals = self.compute_marginal_rewards(
                    ranked_ids, answer_entry, tokens_map, k_trunc
                )
                # Reward-to-go: G[j] = sum_{t=j}^{k-1} marginals[t]
                G = np.cumsum(marginals[::-1])[::-1].copy()
                G_list.append(G)
                per_pos_lps.append(per_pos_lp)

                result = score_query(answer_entry, ranked_ids, tokens_map)
                rewards_list.append(result["score"])

            # Compute advantages using batch mean across the K rollouts
            G_matrix = np.array(G_list, dtype=np.float32)  # (K, k_trunc)
            baseline_per_pos = G_matrix.mean(axis=0, keepdims=True)  # (1, k_trunc)
            advantages_matrix = G_matrix - baseline_per_pos  # (K, k_trunc)

            per_pos_losses = []
            for k_idx in range(samples_k):
                adv_t = torch.tensor(advantages_matrix[k_idx], dtype=torch.float32, device=DEVICE)
                per_pos_loss = -(adv_t * per_pos_lps[k_idx]).sum()
                per_pos_losses.append(per_pos_loss)

            loss = torch.stack(per_pos_losses).mean()

            # Mini-batch gradient accumulation
            (loss / accum_steps).backward()
            if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == n_questions:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            mean_reward = np.mean(rewards_list)
            total_reward += mean_reward
            total_above_floor += (mean_reward - answer_entry.get("floor", 0.0))

        return {
            "mean_loss": total_loss / max(1, n_questions),
            "mean_reward": total_reward / max(1, n_questions),
            "mean_above_floor": total_above_floor / max(1, n_questions),
            "temperature": tau,
        }

    def evaluate(self, eval_data: list) -> Dict[str, float]:
        """
        Evaluate candidate rankings using Contextual Sequential MMR selection.
        """
        self.model.eval()
        total_reward = 0.0
        total_above_floor = 0.0
        total_full_support = 0.0
        n = len(eval_data)

        for item in eval_data:
            block_ids = item["block_ids"]
            if not block_ids:
                n -= 1
                continue

            r_feats_np = item.get("retrieval_feats")
            tokens_map = item["tokens_map"]
            answer_entry = item["answer_entry"]

            # Sequential Contextual MMR ranking
            ranked_ids = self.model.select_mmr(
                block_ids=block_ids,
                query_emb=item["query_emb"],
                block_embs=item["block_embs"],
                struct_feats=item["struct_feats"],
                retrieval_feats=r_feats_np,
                tokens_map=tokens_map,
            )

            result = score_query(answer_entry, ranked_ids, tokens_map)
            total_reward += result["score"]
            total_above_floor += result["above_floor"]
            total_full_support += result["full_support"]

        if n == 0:
            return {"score": 0.0, "above_floor": 0.0, "full_support": 0.0}
        return {
            "score": total_reward / n,
            "above_floor": total_above_floor / n,
            "full_support": total_full_support / n,
        }

    def train(
        self,
        train_data: list,
        eval_data: Optional[list] = None,
        num_epochs: int = RL_NUM_EPOCHS,
        imitation_epochs: int = IMITATION_EPOCHS,
        save_prefix: str = "reranker",
        samples_k: int = RL_SAMPLES_K,
    ) -> Dict[str, List]:
        """
        Full two-stage training loop:
          Stage 1: Oracle Imitation Pre-training
          Stage 2: Budget-Marginal RL Fine-Tuning
        """
        history = {
            "epoch": [], "loss": [], "mean_reward": [],
            "above_floor": [], "temperature": [],
            "eval_score": [], "eval_above_floor": [], "eval_full_support": [],
        }

        # ─── Stage 1: Oracle Imitation Pre-training ────────────────────────
        if imitation_epochs > 0:
            self.pretrain_imitation(
                train_data, eval_data, epochs=imitation_epochs,
                lr=IMITATION_LR, save_prefix=save_prefix
            )

        # ─── Stage 2: Budget-Marginal RL Fine-Tuning ───────────────────────
        print(f"\n--- Stage 2: Budget-Marginal RL Fine-Tuning ({num_epochs} epochs) ---")
        best_eval = -1.0
        if eval_data is not None:
            init_eval = self.evaluate(eval_data)
            best_eval = init_eval["score"]
            self.save(f"{save_prefix}_best")
            history["eval_score"].append(best_eval)
            history["eval_above_floor"].append(init_eval["above_floor"])
            history["eval_full_support"].append(init_eval["full_support"])
            print(f"  [Post-Imitation Checkpoint] eval score={best_eval:.4f} above_floor={init_eval['above_floor']:+.4f} (best saved)")

        # Re-initialize optimizer for RL with fine-tuning learning rate
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        for epoch in range(num_epochs):
            t0 = time.time()
            metrics = self.train_epoch(train_data, epoch, num_epochs, samples_k)
            elapsed = time.time() - t0

            history["epoch"].append(epoch + 1)
            history["loss"].append(metrics["mean_loss"])
            history["mean_reward"].append(metrics["mean_reward"])
            history["above_floor"].append(metrics["mean_above_floor"])
            history["temperature"].append(metrics["temperature"])

            log_parts = [
                f"RL Epoch {epoch+1:2d}/{num_epochs}",
                f"loss={metrics['mean_loss']:.4f}",
                f"reward={metrics['mean_reward']:.4f}",
                f"above_floor={metrics['mean_above_floor']:+.4f}",
                f"τ={metrics['temperature']:.3f}",
                f"({elapsed:.1f}s)",
            ]

            if eval_data is not None and ((epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs):
                eval_metrics = self.evaluate(eval_data)
                history["eval_score"].append(eval_metrics["score"])
                history["eval_above_floor"].append(eval_metrics["above_floor"])
                history["eval_full_support"].append(eval_metrics["full_support"])
                is_best = eval_metrics["score"] >= best_eval
                if is_best:
                    best_eval = eval_metrics["score"]
                    self.save(f"{save_prefix}_best")
                star = " [NEW BEST -> SAVED]" if is_best else ""
                log_parts.append(
                    f"| eval score={eval_metrics['score']:.4f} above_floor={eval_metrics['above_floor']:+.4f}{star}"
                )

            print(" ".join(log_parts))

        # Ensure best checkpoint is loaded at completion
        best_path = self.checkpoint_dir / f"{save_prefix}_best.pt"
        if best_path.exists():
            self.load(f"{save_prefix}_best")

        # Save history
        hist_path = self.checkpoint_dir / f"{save_prefix}_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        return history

    def save(self, name: str):
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(self.model.state_dict(), path)

    def load(self, name: str):
        path = self.checkpoint_dir / f"{name}.pt"
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"Loaded checkpoint: {path}")
