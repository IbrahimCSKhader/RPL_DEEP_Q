import csv
from pathlib import Path


class DecisionLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "protocol",
                "round",
                "source_id",
                "current_node",
                "selected_parent",
                "state",
                "decision_rule",
                "candidate_details",
                "selection_mode",
                "candidate_parent",
                "candidate_energy_ratio",
                "candidate_distance_ratio",
                "candidate_queue_ratio",
                "candidate_link_quality",
                "action",
                "old_q_value",
                "state_value",
                "selection_value",
                "reward",
                "new_q_value",
                "best_next_q",
                "current_energy",
                "parent_energy",
                "parent_queue_load",
                "delivered",
                "lost",
                "delay",
                "hop_distance",
                "hop_distance_ratio",
                "tx_cost",
                "energy_consumed",
                "loss_probability",
                "delay_increment",
                "equations",
                "reward_equation",
                "q_update_equation",
            ],
        )
        self.writer.writeheader()

    def log_decision(
        self,
        protocol,
        round_number: int,
        source_id: int,
        network,
        node_id: int,
        parent_id: int | None,
        delivered: bool,
        lost: bool,
        delay: float,
        decision=None,
        learning_result=None,
        hop_details=None,
    ) -> None:
        decision = decision or getattr(protocol, "last_decision", None) or {}
        learning_result = learning_result or {}
        hop_details = hop_details or {}
        current = network.nodes[node_id]
        parent = network.nodes[parent_id] if parent_id is not None else None
        self.writer.writerow(
            {
                "protocol": protocol.name,
                "round": round_number,
                "source_id": source_id,
                "current_node": node_id,
                "selected_parent": parent_id if parent_id is not None else "",
                "state": decision.get("state", ""),
                "decision_rule": decision.get("rule", ""),
                "candidate_details": decision.get("candidates", ""),
                "selection_mode": decision.get("selection_mode", ""),
                "candidate_parent": decision.get("candidate_parent", ""),
                "candidate_energy_ratio": self._format_float(decision.get("candidate_energy_ratio", "")),
                "candidate_distance_ratio": self._format_float(decision.get("candidate_distance_ratio", "")),
                "candidate_queue_ratio": self._format_float(decision.get("candidate_queue_ratio", "")),
                "candidate_link_quality": self._format_float(decision.get("candidate_link_quality", "")),
                "action": decision.get("action", ""),
                "old_q_value": self._format_float(learning_result.get("old_q_value", decision.get("old_q_value", ""))),
                "state_value": self._format_float(decision.get("state_value", "")),
                "selection_value": self._format_float(decision.get("selection_value", "")),
                "reward": self._format_float(learning_result.get("reward", "")),
                "new_q_value": self._format_float(learning_result.get("new_q_value", "")),
                "best_next_q": self._format_float(learning_result.get("best_next_q", "")),
                "current_energy": f"{current.energy:.5f}",
                "parent_energy": "inf" if parent and parent.is_sink else f"{parent.energy:.5f}" if parent else "",
                "parent_queue_load": f"{parent.queue_load:.2f}" if parent else "",
                "delivered": int(delivered),
                "lost": int(lost),
                "delay": f"{delay:.5f}",
                "hop_distance": self._format_float(hop_details.get("hop_distance", "")),
                "hop_distance_ratio": self._format_float(hop_details.get("hop_distance_ratio", "")),
                "tx_cost": self._format_float(hop_details.get("tx_cost", "")),
                "energy_consumed": self._format_float(hop_details.get("energy_consumed", "")),
                "loss_probability": self._format_float(hop_details.get("loss_probability", "")),
                "delay_increment": self._format_float(hop_details.get("delay_increment", "")),
                "equations": decision.get("equations", ""),
                "reward_equation": learning_result.get("reward_equation", ""),
                "q_update_equation": learning_result.get("q_update_equation", ""),
            }
        )

    @staticmethod
    def _format_float(value) -> str:
        if value == "" or value is None:
            return ""
        return f"{float(value):.5f}"

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
