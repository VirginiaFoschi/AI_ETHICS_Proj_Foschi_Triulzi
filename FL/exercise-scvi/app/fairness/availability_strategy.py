"""Client-availability simulation for FedSCVI.

Real cross-institution FL deployments rarely see every client show up to
every round (a lab's compute is busy, a connection drops, an IRB-gated
node is offline that week, ...). This module adds that on top of the
existing strategies (plain FedAvg-style weighting *or* q-FFL) without
touching them: it is a mixin that only overrides ``configure_train``.

Participation regimes
----------------------
"full"
    Every connected client is asked to train every round (status quo).

"random"
    Each client is independently available each round with a fixed
    probability ``p_available`` (e.g. 0.7), regardless of its size.
    Same *expected* participation for everyone -> isolates the effect of
    dropout itself from any size confound.

"size_correlated"
    Availability probability depends on the client's local training-set
    size: small clients (< ``size_threshold`` cells) are available with
    probability ``p_small``, larger clients with probability ``p_large``.
    This mimics smaller/under-resourced sites being less reliably online,
    and is meant to test whether it amplifies the size effect already
    observed with full participation.

Why a mixin, not ``fraction_train``
------------------------------------
Flower's built-in ``fraction_train`` just changes *how many* of the
currently-connected clients are uniformly sampled each round -- every
client still has the same participation probability, and there is no way
to make that probability depend on the client's identity/size. To get
independent per-client Bernoulli availability (and size-correlated
probabilities) we replace ``configure_train``'s node-selection logic
instead of leaning on ``fraction_train``. ``fraction_train`` is left at
its default (1.0) here and is not what drives dropout in this file.

Mapping node_id -> client_id
------------------------------
Flower's Grid only gives you opaque ``node_id``s; it does not expose
which data partition a node holds. We learn that mapping ourselves from
the ``client_id`` (and ``num-examples``) every client already reports in
its train/evaluate replies, and cache it. Because the mapping (and, for
the size-correlated regime, each client's size) isn't known until at
least one reply has come in, the first ``bootstrap_rounds`` round(s) are
always run at full participation; dropout kicks in from there.

Evaluation is intentionally left untouched (``configure_evaluate`` is not
overridden): every client is still evaluated every round regardless of
whether it trained, so per-round fairness metrics reflect the true
global model quality for *all* clients, not just the ones that happened
to be online.
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
    """Overrides ``configure_train`` to apply per-round client dropout.

    Must be mixed in *before* a FedAvg-derived class, e.g.::

        class MyStrategy(AvailabilityMixin, FedAvgQFFL):
            ...

    and ``_init_availability(...)`` must be called (typically from
    ``__init__``) before training starts.
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

        # Learned online from replies: real Flower node_id -> partition/client_id,
        # and client_id -> (train) num-examples, used as the client's "size".
        self._node_to_client: Dict[int, int] = {}
        self._client_sizes: Dict[int, int] = {}

        self._participation_log_path = participation_log_path
        if self._participation_log_path is not None:
            os.makedirs(os.path.dirname(self._participation_log_path) or ".", exist_ok=True)
            with open(self._participation_log_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["round", "node_id", "client_id", "client_size", "available"]
                )

    # ------------------------------------------------------------------
    # Learning the node_id <-> client_id map and client sizes from replies
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Availability model
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Strategy hooks
    # ------------------------------------------------------------------

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
    client-availability regime.

    Using ``FedAvgQFFL`` as the base for *both* arms of the grid (q=0 and
    q=5) -- rather than a separate plain-FedAvg class -- means both arms
    get identical fairness logging (``fairness_log.csv`` /
    ``fairness_metrics.jsonl``), so the two are directly comparable.
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
