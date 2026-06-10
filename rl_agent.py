import random
from collections import defaultdict

from network import Network
from routing import parent_energy_ratio, parent_queue_ratio, parent_score


class QLearningRPL:
    name = "Q-learning RL-RPL"

    def __init__(
        self,
        learning_rate: float = 0.25,
        discount_factor: float = 0.75,
        epsilon: float = 0.04,
        min_epsilon: float = 0.01,
        epsilon_decay: float = 0.99,
        seed: int = 42,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.q_table: dict[tuple[int, int, int, int, int], dict[int, float]] = defaultdict(dict)
        self.last_decision = None

    def select_parent(self, network: Network, node_id: int, current_time: float) -> int | None:
        candidates = [
            parent_id
            for parent_id in network.candidate_parents(node_id)
            if network.nodes[parent_id].is_root or network.can_deliver_to_root(parent_id)
        ]
        if candidates:
            closest_root_distance = min(
                network.nodes[parent_id].distance_to(network.root) for parent_id in candidates
            )
            progress_window = network.config.transmission_range * 0.05
            focused_candidates = [
                parent_id
                for parent_id in candidates
                if network.nodes[parent_id].distance_to(network.root) <= closest_root_distance + progress_window
            ]
            candidates = focused_candidates or candidates
        if not candidates:
            self.last_decision = {
                "protocol": self.name,
                "state": "",
                "rule": "drop: no candidate parent closer to ROOT",
                "candidates": "",
                "candidate_records": [],
                "selection_mode": "drop",
                "equations": "No valid parent satisfies alive, in range, and closer-to-ROOT constraints.",
            }
            return None

        candidate_records = self._candidate_records(network, node_id, candidates)

        if self.rng.random() < self.epsilon:
            selected = self.rng.choice(candidates)
            mode = "exploration"
            rule = f"epsilon-greedy exploration: epsilon={self.epsilon:.3f}, random valid parent"
        else:
            selected = max(
                candidates,
                key=lambda parent_id: (
                    candidate_records[parent_id]["selection_value"],
                    candidate_records[parent_id]["state_value"],
                ),
            )
            mode = "exploitation"
            rule = "Q-learning exploitation: maximize selection_value = Q(s,a) + 2.0 * state_value"

        selected_state = candidate_records[selected]["state"]
        self._remember_decision(selected_state, node_id, selected, candidate_records, mode, rule)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return selected

    def observe(
        self,
        network: Network,
        node_id: int,
        parent_id: int | None,
        old_state: tuple[int, int, int, int, int] | None,
        delivered: bool,
        delay: float,
        lost: bool,
    ) -> dict | None:
        if parent_id is None or old_state is None:
            return None

        reward_terms = self._reward_terms(network, parent_id, delivered, delay, lost)
        reward = reward_terms["reward"]
        next_candidates = network.candidate_parents(node_id)
        best_next_q = 0.0
        next_state_text = ""

        if next_candidates:
            for candidate_id in next_candidates:
                next_state = self._state_for_parent(network, node_id, candidate_id)
                self._ensure_action(next_state, candidate_id)
                candidate_q = self.q_table[next_state].get(candidate_id, 0.5)
                if candidate_q >= best_next_q:
                    best_next_q = candidate_q
                    next_state_text = str(next_state)

        current_q = self.q_table[old_state].setdefault(parent_id, 0.5)
        updated_q = current_q + self.learning_rate * (
            reward + self.discount_factor * best_next_q - current_q
        )
        self.q_table[old_state][parent_id] = updated_q
        return {
            "reward": reward,
            "old_q_value": current_q,
            "new_q_value": updated_q,
            "best_next_q": best_next_q,
            "next_state": next_state_text,
            "reward_equation": (
                f"reward = delivery({reward_terms['delivery_reward']:.3f}) "
                f"+ energy({reward_terms['energy_reward']:.3f}) "
                f"- delay({reward_terms['delay_penalty']:.3f}) "
                f"- queue({reward_terms['queue_penalty']:.3f}) "
                f"- loss({reward_terms['loss_penalty']:.3f}) "
                f"- dead_parent({reward_terms['dead_parent_penalty']:.3f}) "
                f"= {reward:.3f}"
            ),
            "q_update_equation": (
                f"Q = {current_q:.3f} + {self.learning_rate:.3f} * "
                f"[{reward:.3f} + {self.discount_factor:.3f} * {best_next_q:.3f} "
                f"- {current_q:.3f}] = {updated_q:.3f}"
            ),
        }

    def end_round(self) -> None:
        return None

    def state_for(self, network: Network, node_id: int) -> tuple[int, int, int, int, int]:
        return self._state(network, node_id)

    def _state(self, network: Network, node_id: int) -> tuple[int, int, int, int, int]:
        candidates = network.candidate_parents(node_id)
        if not candidates:
            return node_id, 0, 3, 0, 3

        best_parent = max(candidates, key=lambda parent_id: self._state_value(network, node_id, parent_id))
        return self._state_for_parent(network, node_id, best_parent)

    def _state_for_parent(self, network: Network, node_id: int, parent_id: int) -> tuple[int, int, int, int, int]:
        energy, distance, link_quality, queue = parent_score(network, node_id, parent_id)
        return (
            node_id,
            self._bucket(energy, [0.25, 0.5, 0.75]),
            self._bucket(distance, [0.2, 0.4, 0.7]),
            self._bucket(link_quality, [0.25, 0.5, 0.75]),
            self._bucket(queue, [0.25, 0.5, 0.75]),
        )

    def _state_value(self, network: Network, node_id: int, parent_id: int) -> float:
        energy, distance, link_quality, queue = parent_score(network, node_id, parent_id)
        return self._state_value_from_components(energy, distance, link_quality, queue)

    @staticmethod
    def _state_value_from_components(
        energy: float,
        distance: float,
        link_quality: float,
        queue: float,
    ) -> float:
        # Educational weighted score: reward energy/link quality, penalize distance/queue load.
        return 0.25 * energy + 0.45 * link_quality - 0.35 * distance - 0.15 * queue

    def _ensure_action(self, state: tuple[int, int, int, int, int], parent_id: int) -> None:
        if parent_id not in self.q_table[state]:
            self.q_table[state][parent_id] = 0.5

    def _candidate_records(
        self,
        network: Network,
        node_id: int,
        candidates: list[int],
    ) -> dict[int, dict]:
        records = {}
        root_scale = max(network.config.area_width, network.config.area_height)
        for parent_id in candidates:
            energy, distance, link_quality, queue = parent_score(network, node_id, parent_id)
            state = self._state_for_parent(network, node_id, parent_id)
            self._ensure_action(state, parent_id)
            q_value = self.q_table[state].get(parent_id, 0.5)
            state_value = self._state_value_from_components(energy, distance, link_quality, queue)
            selection_value = q_value + 2.0 * state_value
            parent = network.nodes[parent_id]
            records[parent_id] = {
                "candidate_parent": parent_id,
                "state": state,
                "energy_ratio": energy,
                "distance_ratio": distance,
                "queue_ratio": queue,
                "link_quality": link_quality,
                "q_value": q_value,
                "old_q_value": q_value,
                "state_value": state_value,
                "selection_value": selection_value,
                "parent_energy": parent.energy,
                "parent_initial_energy": parent.initial_energy,
                "parent_queue_load": parent.queue_load,
                "distance_to_root": parent.distance_to(network.root),
                "root_scale": root_scale,
                "is_root": parent.is_root,
            }
        return records

    def _candidate_summary(self, candidate_records: dict[int, dict]) -> str:
        details = []
        for parent_id, record in candidate_records.items():
            details.append(
                f"{parent_id}:Q={record['q_value']:.3f},state_value={record['state_value']:.3f},"
                f"selection={record['selection_value']:.3f},energy={record['energy_ratio']:.3f},"
                f"dist={record['distance_ratio']:.3f},link={record['link_quality']:.3f},"
                f"queue={record['queue_ratio']:.3f}"
            )
        return "; ".join(details)

    def _remember_decision(
        self,
        state: tuple[int, int, int, int, int],
        node_id: int,
        selected: int,
        candidate_records: dict[int, dict],
        selection_mode: str,
        rule: str,
    ) -> None:
        selected_record = candidate_records[selected]
        self.last_decision = {
            "protocol": self.name,
            "state": str(state),
            "state_full": str(state),
            "state_tuple": selected_record["state"],
            "rule": rule,
            "selection_mode": selection_mode,
            "candidates": self._candidate_summary(candidate_records),
            "candidate_records": list(candidate_records.values()),
            "candidate_parent": selected,
            "action": selected,
            "candidate_energy_ratio": selected_record["energy_ratio"],
            "candidate_distance_ratio": selected_record["distance_ratio"],
            "candidate_queue_ratio": selected_record["queue_ratio"],
            "candidate_link_quality": selected_record["link_quality"],
            "old_q_value": selected_record["q_value"],
            "state_value": selected_record["state_value"],
            "selection_value": selected_record["selection_value"],
            "equations": self._candidate_equations(node_id, selected_record),
        }

    def _candidate_equations(self, node_id: int, record: dict) -> str:
        if record["is_root"]:
            energy_expression = "ROOT energy = 1.000"
        else:
            energy_expression = (
                f"{record['parent_energy']:.5f} / {record['parent_initial_energy']:.5f} "
                f"= {record['energy_ratio']:.5f}"
            )
        raw_link = (1.0 - record["distance_ratio"]) - (record["parent_queue_load"] * 0.05)
        return (
            f"Node {node_id} evaluating Parent {record['candidate_parent']}: "
            f"energy_ratio = {energy_expression}; "
            f"distance_ratio = {record['distance_to_root']:.5f} / {record['root_scale']:.5f} "
            f"= {record['distance_ratio']:.5f}; "
            f"queue_ratio = min({record['parent_queue_load']:.5f} / 10, 1) "
            f"= {record['queue_ratio']:.5f}; "
            f"link_quality = clip((1 - {record['distance_ratio']:.5f}) "
            f"- ({record['parent_queue_load']:.5f} * 0.05), 0.05, 1.0) "
            f"= clip({raw_link:.5f}, 0.05, 1.0) = {record['link_quality']:.5f}; "
            f"state_value = 0.25({record['energy_ratio']:.5f}) "
            f"+ 0.45({record['link_quality']:.5f}) "
            f"- 0.35({record['distance_ratio']:.5f}) "
            f"- 0.15({record['queue_ratio']:.5f}) = {record['state_value']:.5f}; "
            f"selection_value = Q(s,a) + 2.0 * state_value = "
            f"{record['q_value']:.5f} + 2.0 * {record['state_value']:.5f} "
            f"= {record['selection_value']:.5f}"
        )

    def _reward_terms(
        self,
        network: Network,
        parent_id: int,
        delivered: bool,
        delay: float,
        lost: bool,
    ) -> dict[str, float]:
        parent = network.nodes[parent_id]
        delivery_reward = 2.0 if delivered else -2.0
        loss_penalty = 1.5 if lost else 0.0
        energy_reward = parent_energy_ratio(parent)
        normalized_delay = min(delay / 10.0, 1.0)
        delay_penalty = normalized_delay * 0.5
        queue_penalty = parent_queue_ratio(parent) * 1.0
        dead_parent_penalty = 2.0 if not parent.is_root and not parent.alive else 0.0
        reward = (
            delivery_reward
            + energy_reward
            - delay_penalty
            - queue_penalty
            - loss_penalty
            - dead_parent_penalty
        )
        return {
            "delivery_reward": delivery_reward,
            "energy_reward": energy_reward,
            "loss_penalty": loss_penalty,
            "delay_penalty": delay_penalty,
            "queue_penalty": queue_penalty,
            "dead_parent_penalty": dead_parent_penalty,
            "reward": reward,
        }

    @staticmethod
    def _bucket(value: float, thresholds: list[float]) -> int:
        for index, threshold in enumerate(thresholds):
            if value < threshold:
                return index
        return len(thresholds)
