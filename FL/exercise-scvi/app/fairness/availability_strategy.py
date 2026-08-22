"""
Participation regimes:
-"full": Every connected client is asked to train every round
-"random": Each client is independently available each round with a fixed
    probability (p_available), regardless of its size
-"size_correlated": Availability probability depends on the client's local training-set
    size: small clients (< ``size_threshold`` cells) are available with
    probability ``p_small``, larger clients with probability ``p_large``.
    This mimics smaller/under-resourced sites being less reliably online,
    and is meant to test whether it amplifies the size effect already
    observed with full participation.
"""

from __future__ import annotations

import csv
import os
import random
from typing import Dict, Iterable, Optional

from flwr.app import ConfigRecord, MessageType, RecordDict
from flwr.common import log
from logging import INFO

from app.fairness.fairness_strategy import FedAvgQFFL


class AvailabilityMixin:
    """Overrides ``configure_train`` to apply per-round client dropout
    """

    def _init_availability(
        self,
        *,
        participation_regime: str = "full",
        p_available: float = 0.7,
        size_threshold: int = 800,
        p_small: float = 0.5,
        p_large: float = 0.9,
        availability_seed: int = 42,
        bootstrap_rounds: int = 1,
        participation_log_path: Optional[str] = None,
    ) -> None:
        valid_regimes = {"full", "random", "size_correlated"}
        if participation_regime not in valid_regimes:
            raise ValueError(
                f"Unknown participation_regime={participation_regime!r}. "
                f"Expected one of {sorted(valid_regimes)}."
            )

        self.participation_regime = participation_regime
        self.p_available = float(p_available)
        self.size_threshold = int(size_threshold)
        self.p_small = float(p_small)
        self.p_large = float(p_large)
        self.bootstrap_rounds = int(bootstrap_rounds)

        self._avail_rng = random.Random(availability_seed)

        self._node_to_client: Dict[int, int] = {}
        self._client_sizes: Dict[int, int] = {}

        self._participation_log_path = participation_log_path
        if self._participation_log_path is not None:
            os.makedirs(os.path.dirname(self._participation_log_path) or ".", exist_ok=True)
            with open(self._participation_log_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["round", "node_id", "client_id", "client_size", "available"]
                )

    def _learn_from_replies(self, replies: Iterable) -> None:
        for reply_msg in replies:
            if not reply_msg.has_content():
                continue
            metrics = reply_msg.content.get("metrics")
            if metrics is None:
                continue
            try:
                client_id = int(metrics["client_id"])
            except (KeyError, TypeError):
                continue

            node_id = reply_msg.metadata.src_node_id
            self._node_to_client[node_id] = client_id

            # Only train replies carry the client's training-set size; evaluate
            # replies report the (smaller) validation-set size under the same
            # key, so we must not let those overwrite the cached train size.
            if reply_msg.metadata.message_type == MessageType.TRAIN:
                try:
                    n_k = int(metrics["num-examples"])
                    self._client_sizes[client_id] = n_k
                except (KeyError, TypeError, ValueError):
                    pass

    def _availability_prob(self, client_id: Optional[int]) -> float:
        if self.participation_regime == "full":
            return 1.0
        if self.participation_regime == "random":
            return self.p_available
        if self.participation_regime == "size_correlated":
            if client_id is None:
                return 1.0
            size = self._client_sizes.get(client_id)
            if size is None:
                # Size not learned yet for this client -> don't penalize it.
                return 1.0
            return self.p_small if size < self.size_threshold else self.p_large
        raise ValueError(self.participation_regime)  # unreachable, validated earlier

    def _select_participating_nodes(self, all_node_ids, server_round: int):
        """Return (selected_node_ids, {node_id: bool available}) for this round."""
        # Bootstrap: run at full participation until we've learned the
        # node_id<->client_id map (and, for size_correlated, client sizes).
        if self.participation_regime == "full" or server_round <= self.bootstrap_rounds:
            return list(all_node_ids), {n: True for n in all_node_ids}

        availability: Dict[int, bool] = {}
        selected = []
        for node_id in all_node_ids:
            client_id = self._node_to_client.get(node_id)
            prob = self._availability_prob(client_id)
            is_available = self._avail_rng.random() < prob
            availability[node_id] = is_available
            if is_available:
                selected.append(node_id)

        # Never let a bad draw stall the whole run: top up to min_train_nodes
        # (present on any FedAvg subclass) by pulling back in extra nodes.
        min_needed = max(1, getattr(self, "min_train_nodes", 1))
        if len(selected) < min_needed:
            missing = min_needed - len(selected)
            fallback_candidates = [n for n in all_node_ids if n not in selected]
            self._avail_rng.shuffle(fallback_candidates)
            for node_id in fallback_candidates[:missing]:
                availability[node_id] = True
                selected.append(node_id)

        return selected, availability

    def _log_participation(self, server_round: int, all_node_ids, availability) -> None:
        if self._participation_log_path is None:
            return
        with open(self._participation_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            for node_id in all_node_ids:
                client_id = self._node_to_client.get(node_id)
                size = self._client_sizes.get(client_id) if client_id is not None else None
                writer.writerow(
                    [server_round, node_id, client_id, size, availability.get(node_id, True)]
                )

    def configure_train(self, server_round, arrays, config, grid):
        """Same as FedAvg.configure_train, but node selection = availability
        draw instead of a uniform ``fraction_train`` sample."""
        if self.fraction_train == 0.0:
            return []

        all_node_ids = list(grid.get_node_ids())
        selected, availability = self._select_participating_nodes(all_node_ids, server_round)
        self._log_participation(server_round, all_node_ids, availability)

        n_avail = sum(availability.values())
        log(
            INFO,
            "configure_train [%s]: %d/%d clients available this round, %d selected",
            self.participation_regime,
            n_avail,
            len(all_node_ids),
            len(selected),
        )

        config["server-round"] = server_round
        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})
        return self._construct_messages(record, selected, MessageType.TRAIN)

    def aggregate_train(self, server_round, replies):
        self._learn_from_replies(replies)
        return super().aggregate_train(server_round, replies)

    def aggregate_evaluate(self, server_round, replies):
        self._learn_from_replies(replies)
        return super().aggregate_evaluate(server_round, replies)


class FedAvgQFFLAvailability(AvailabilityMixin, FedAvgQFFL):
    """q-FFL (q=0 == plain num-examples-weighted FedAvg) crossed with a
    client-availability regime
    """

    def __init__(
        self,
        *,
        participation_regime: str = "full",
        p_available: float = 0.7,
        size_threshold: int = 800,
        p_small: float = 0.5,
        p_large: float = 0.9,
        availability_seed: int = 42,
        bootstrap_rounds: int = 1,
        participation_log_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._init_availability(
            participation_regime=participation_regime,
            p_available=p_available,
            size_threshold=size_threshold,
            p_small=p_small,
            p_large=p_large,
            availability_seed=availability_seed,
            bootstrap_rounds=bootstrap_rounds,
            participation_log_path=participation_log_path,
        )
