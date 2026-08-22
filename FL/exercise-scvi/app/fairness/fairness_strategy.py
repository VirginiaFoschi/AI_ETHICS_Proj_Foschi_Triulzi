"""
Fairness-aware aggregation strategies for FedSCVI, built on top of the
Flower *Message API* classes in custom_strategy.py (aggregate_train /
aggregate_evaluate operating on lists of reply Messages).

FedAvgUniform
    Aggregates client array updates with EQUAL weight per client in
    aggregate_train, ignoring num-examples. 

FedAvgQFFL
    q-Fair Federated Learning (Li et al., 2020). Client weight in
    aggregate_train ~ num_examples * (train_loss ** q). q = 0 reduces to
    plain num-examples weighting.

FedAvgAdaptiveQFFL
    Adaptive q-FFL, inspired by Pei (2025) F^3.

Both classes subclass FedAvgSaveModelPlotLosses.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from flwr.app import ArrayRecord, MetricRecord

from app.custom_strategy import FedAvgSaveModelPlotLosses


def fairness_metrics(client_losses: Dict[int, float]) -> dict:
    """Jain's fairness index (1 = perfectly equal loss across clients,
    1/N = maximally unequal), plus variance / worst / best client loss.
    """
    losses = np.array(list(client_losses.values()), dtype=float)
    n = len(losses)
    if n == 0:
        return {}
    sq_sum = float((losses ** 2).sum())
    jain = float((losses.sum() ** 2) / (n * sq_sum)) if sq_sum > 0 else 1.0
    return {
        "n_clients": n,
        "mean_loss": float(losses.mean()),
        "variance": float(losses.var()),
        "worst_client_loss": float(losses.max()),
        "best_client_loss": float(losses.min()),
        "jain_index": jain,
    }


def _weighted_average_arrays(
    ndarrays_list: List[List[np.ndarray]], weights: List[float]
) -> List[np.ndarray]:
    """Elementwise weighted average of a list of model parameter lists.

    ndarrays_list[i] is client i's full list of parameter arrays
    (as returned by ArrayRecord.to_numpy_ndarrays()); weights[i] is that
    client's aggregation weight. Mirrors what FedAvg does internally,
    but with caller-supplied weights instead of num-examples only.
    """
    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("Total aggregation weight is zero; cannot aggregate.")

    num_arrays = len(ndarrays_list[0])
    aggregated = []
    for i in range(num_arrays):
        weighted_sum = sum(
            client_arrays[i] * w for client_arrays, w in zip(ndarrays_list, weights)
        )
        aggregated.append(weighted_sum / total_weight)
    return aggregated

class _FairnessLoggingMixin:
    def _init_fairness_logs(self, fairness_log_dir: str, tag: str | None = None) -> None:
        os.makedirs(fairness_log_dir, exist_ok=True)
        self.fairness_log_dir = fairness_log_dir
        tag = tag or self.__class__.__name__.lower()
        self._fair_csv_path = Path(fairness_log_dir) / f"fairness_log.csv"
        self._fair_jsonl_path = Path(fairness_log_dir) / f"fairness_metrics.jsonl"

        with open(self._fair_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "round",
                    "client_id",
                    "num_examples",
                    "valid_loss"
                ]
            )
        open(self._fair_jsonl_path, "w").close()

    def _log_fairness(self, server_round: int, replies) -> None:
        client_losses: Dict[int, float] = {}

        with open(self._fair_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for reply_msg in replies:
                if not reply_msg.has_content():
                    continue
                msg_metrics = reply_msg.content.get("metrics")
                if msg_metrics is None:
                    continue

                try:
                    n_i = int(msg_metrics["num-examples"])
                except (KeyError, TypeError):
                    continue
                if n_i <= 0:
                    continue

                try:
                    valid_loss = float(msg_metrics["eval_loss"])
                except (KeyError, TypeError):
                    try:
                        valid_loss = float(msg_metrics["valid_loss"])
                    except (KeyError, TypeError):
                        continue

                try:
                    client_id = int(msg_metrics["client_id"])
                except (KeyError, TypeError):
                    client_id = None

                writer.writerow(
                    [server_round, client_id, n_i, valid_loss]
                )
                if client_id is not None:
                    client_losses[client_id] = valid_loss

        if client_losses:
            metrics = fairness_metrics(client_losses)
            
            with open(self._fair_jsonl_path, "a") as f:
                f.write(
                    json.dumps({"round": server_round, **metrics}) + "\n"
                )


# Strategy 1: equal-weight (uniform) aggregation
class FedAvgUniform(_FairnessLoggingMixin, FedAvgSaveModelPlotLosses):
    """Every client's array update counts the same in aggregate_train,
    regardless of num-examples. Fairness baseline #1."""

    def __init__(self, *, fairness_log_dir: str = ".", **kwargs):
        super().__init__(**kwargs)
        self._init_fairness_logs(fairness_log_dir)

    def aggregate_train(self, server_round: int, replies):
        valid_replies = [
            r for r in replies
            if r.has_content() and r.content.get("arrays") is not None
        ]
        if not valid_replies:
            return None, MetricRecord()

        ndarrays_list = [r.content["arrays"].to_numpy_ndarrays() for r in valid_replies]
        weights = [1.0 for _ in valid_replies]  # equal weight, ignore num-examples

        aggregated_ndarrays = _weighted_average_arrays(ndarrays_list, weights)
        aggregated_arrays = ArrayRecord(numpy_ndarrays=aggregated_ndarrays)

        if aggregated_arrays is not None and int(server_round) >= self._num_rounds:
            try:
                self._on_final_arrays(aggregated_arrays)
            except Exception as e:
                print(f"[WARN] Failed saving on final round: {e}")

        return aggregated_arrays, MetricRecord()

    def aggregate_evaluate(self, server_round: int, replies):
        aggregated_metrics = super().aggregate_evaluate(server_round, replies)
        self._log_fairness(server_round, replies)
        return aggregated_metrics


# Strategy 2: q-FFL

class FedAvgQFFL(_FairnessLoggingMixin, FedAvgSaveModelPlotLosses):
    """
    q-Fair Federated Learning (https://arxiv.org/abs/1905.10497).
    weight_k = num_examples_k * (train_loss_k ** q)

    train_loss_k is the client's *cached* training loss from the previous round's
    evaluate() reply. q=0 -> weight_k =
    num_examples_k, i.e. identical to plain FedAvg weighting.

    """

    def __init__(self, *, q: float = 1.0, eps: float = 1e-8,
                 fairness_log_dir: str = ".", fairness_log_tag: str | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.q = q
        self.eps = eps
        q_str = str(q).replace(".", "p")
        tag = fairness_log_tag or f"fedavgqffl_q{q_str}"
        self._init_fairness_logs(fairness_log_dir, tag=tag)
        self._cached_train_loss: Dict[int, float] = {}

    def aggregate_train(self, server_round: int, replies):
        valid_replies = [
            r for r in replies
            if r.has_content()
            and r.content.get("arrays") is not None
            and r.content.get("metrics") is not None
        ]
        if not valid_replies:
            return None, MetricRecord()

        ndarrays_list = []
        weights = []
        for r in valid_replies:
            metrics = r.content["metrics"]
            arrays = r.content["arrays"].to_numpy_ndarrays()

            try:
                n_k = int(metrics["num-examples"])
            except (KeyError, TypeError):
                n_k = 0
            if n_k <= 0:
                continue

            try:
                client_id = int(metrics["client_id"])
            except (KeyError, TypeError):
                client_id = None

            F_k = self._cached_train_loss.get(client_id) if client_id is not None else None
            w_k = n_k if F_k is None else n_k * ((F_k + self.eps) ** self.q)

            print(f"[QFFL DEBUG] round={server_round} client={client_id} "
                  f"n_k={n_k} F_k={F_k} w_k={w_k}")

            ndarrays_list.append(arrays)
            weights.append(w_k)

        if not ndarrays_list:
            return None, MetricRecord()

        aggregated_ndarrays = _weighted_average_arrays(ndarrays_list, weights)
        aggregated_arrays = ArrayRecord(numpy_ndarrays=aggregated_ndarrays)

        if aggregated_arrays is not None and int(server_round) >= self._num_rounds:
            try:
                self._on_final_arrays(aggregated_arrays)
            except Exception as e:
                print(f"[WARN] Failed saving on final round: {e}")

        return aggregated_arrays, MetricRecord()

    def aggregate_evaluate(self, server_round: int, replies):
        aggregated_metrics = super().aggregate_evaluate(server_round, replies)
        self._log_fairness(server_round, replies)

        # cache each client's train_loss for the next round
        for reply_msg in replies:
            if not reply_msg.has_content():
                continue
            m = reply_msg.content.get("metrics")
            if m is None:
                continue
            try:
                client_id = int(m["client_id"])
                train_loss = float(m["train_loss"])
                self._cached_train_loss[client_id] = train_loss
            except (KeyError, TypeError, ValueError):
                continue

        return aggregated_metrics


# Strategy 3: F^3-inspired adaptive q-FFL

class FedAvgAdaptiveQFFL(FedAvgQFFL):
    """
    Adaptive q-FFL, inspired by:
        Pei, J. "F^3: Fair Federated Learning Framework with adaptive
        regularization." Knowledge-Based Systems 316 (2025): 113392.
 
    F^3 injects a fairness regularization term (based on variance or MAD
    of client losses) directly into each CLIENT's local training loss,
    with a regularization weight lambda that the server adapts every
    round based on the round-over-round rate of change of the fairness
    metric: lambda increases when fairness is deteriorating, decreases
    when it's improving (F^3 Eq. 17-18).
    """
 
    def __init__(
        self,
        *,
        q: float = 1.0,
        alpha: float = 0.5,
        q_min: float = 0.0,
        q_max: float = 5.0,
        **kwargs,
    ):
        tag = f"fedavgadaptiveqffl"
 
        super().__init__(q=q, fairness_log_tag=tag, **kwargs)
        self.alpha = alpha
        self.q_min = q_min
        self.q_max = q_max
        self._jain_history: List[Tuple[int, float]] = []
 
        self._q_log_path = Path(self.fairness_log_dir) / f"q_trace_{tag}.csv"
        with open(self._q_log_path, "w", newline="") as f:
            csv.writer(f).writerow(["round", "q", "jain_index", "delta_F"])
 
    def aggregate_evaluate(self, server_round: int, replies):
        # Run the normal q-FFL evaluate handling first: global loss
        # logging/plotting, per-client fairness CSV/JSONL logging, and
        # caching each client's train_loss for aggregate_train's weighting
        # -- all inherited unchanged from FedAvgQFFL.
        aggregated_metrics = super().aggregate_evaluate(server_round, replies)
 
        # Recompute this round's Jain's index directly from client losses,
        # to decide how to adjust q for the NEXT round's aggregate_train.
        
        client_losses: Dict[int, float] = {}
        for reply_msg in replies:
            if not reply_msg.has_content():
                continue
            m = reply_msg.content.get("metrics")
            if m is None:
                continue
            try:
                cid = int(m["client_id"])
            except (KeyError, TypeError):
                continue
            try:
                loss = float(m["eval_loss"])
            except (KeyError, TypeError):
                try:
                    loss = float(m["valid_loss"])
                except (KeyError, TypeError):
                    continue
            client_losses[cid] = loss
 
        delta_F = None
        if client_losses:
            jain_now = fairness_metrics(client_losses)["jain_index"]
            self._jain_history.append((server_round, jain_now))
 
            if len(self._jain_history) >= 2:
                _, jain_prev = self._jain_history[-2]
                if jain_prev > 0:
                    delta_F = (jain_now - jain_prev) / jain_prev
                    # jain rising (fairer) -> relax q down;
                    # jain falling (less fair) -> push q up
                    new_q = self.q - self.alpha * delta_F
                    self.q = float(np.clip(new_q, self.q_min, self.q_max))
 
            with open(self._q_log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [server_round, self.q, jain_now, delta_F if delta_F is not None else ""]
                )
 
        return aggregated_metrics