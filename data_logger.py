import csv
from pathlib import Path


class PacketLogger:
    fieldnames = [
        "packet_id",
        "protocol",
        "source_sensor_id",
        "temperature",
        "generation_time",
        "hop_number",
        "current_node",
        "selected_parent",
        "link_quality",
        "queue_load",
        "energy_before",
        "energy_after",
        "hop_delay",
        "packet_status",
    ]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()

    def log_packet_hop(self, packet: dict, hop: dict) -> None:
        self.writer.writerow(
            {
                "packet_id": packet["packet_id"],
                "protocol": packet["protocol_mode"],
                "source_sensor_id": packet["source_sensor_id"],
                "temperature": self._format_float(packet["temperature_value"]),
                "generation_time": self._format_float(packet["generation_time"]),
                "hop_number": hop.get("hop_number", ""),
                "current_node": hop.get("current_node", ""),
                "selected_parent": hop.get("selected_parent", ""),
                "link_quality": self._format_float(hop.get("link_quality", "")),
                "queue_load": self._format_float(hop.get("queue_load", "")),
                "energy_before": self._format_float(hop.get("energy_before", "")),
                "energy_after": self._format_float(hop.get("energy_after", "")),
                "hop_delay": self._format_float(hop.get("hop_delay", "")),
                "packet_status": hop.get("packet_status", packet.get("status", "")),
            }
        )

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def _format_float(value) -> str:
        if value == "" or value is None:
            return ""
        return f"{float(value):.5f}"


class DecisionLogger:
    fieldnames = [
        "time",
        "protocol",
        "node_id",
        "candidate_parent",
        "energy_ratio",
        "distance_ratio",
        "queue_ratio",
        "link_quality",
        "Q_value",
        "state_value",
        "selection_value",
        "selected_parent",
        "reward",
        "updated_Q_value",
        "decision_mode",
    ]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()

    def log_decision(
        self,
        protocol,
        current_time: float,
        node_id: int,
        parent_id: int | None,
        decision: dict | None = None,
        learning_result: dict | None = None,
    ) -> None:
        decision = decision or getattr(protocol, "last_decision", None) or {}
        learning_result = learning_result or {}
        candidate_records = decision.get("candidate_records") or []

        if not candidate_records:
            self._write_row(current_time, protocol.name, node_id, {}, parent_id, decision, learning_result)
            return

        for record in candidate_records:
            self._write_row(current_time, protocol.name, node_id, record, parent_id, decision, learning_result)

    def _write_row(
        self,
        current_time: float,
        protocol_name: str,
        node_id: int,
        record: dict,
        parent_id: int | None,
        decision: dict,
        learning_result: dict,
    ) -> None:
        candidate_parent = record.get("candidate_parent", "")
        selected = parent_id if parent_id is not None else ""
        is_selected = candidate_parent == parent_id
        self.writer.writerow(
            {
                "time": self._format_float(current_time),
                "protocol": protocol_name,
                "node_id": node_id,
                "candidate_parent": candidate_parent,
                "energy_ratio": self._format_float(record.get("energy_ratio", "")),
                "distance_ratio": self._format_float(record.get("distance_ratio", "")),
                "queue_ratio": self._format_float(record.get("queue_ratio", "")),
                "link_quality": self._format_float(record.get("link_quality", "")),
                "Q_value": self._format_float(record.get("q_value", record.get("old_q_value", ""))),
                "state_value": self._format_float(record.get("state_value", "")),
                "selection_value": self._format_float(record.get("selection_value", "")),
                "selected_parent": selected,
                "reward": self._format_float(learning_result.get("reward", "") if is_selected else ""),
                "updated_Q_value": self._format_float(learning_result.get("new_q_value", "") if is_selected else ""),
                "decision_mode": decision.get("selection_mode", ""),
            }
        )

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def _format_float(value) -> str:
        if value == "" or value is None:
            return ""
        return f"{float(value):.5f}"


def export_summary_metrics(results: list, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "protocol",
                "generated_packets",
                "delivered_packets",
                "lost_packets",
                "PDR",
                "total_energy_consumed",
                "average_remaining_energy",
                "alive_nodes",
                "dead_nodes",
                "active_sensors",
                "average_delay",
                "communication_overhead",
                "first_dead_node_time",
                "first_unable_to_send_time",
                "network_lifetime",
                "parent_changes",
            ],
        )
        writer.writeheader()
        for result in results:
            final = result.final_metric
            if final is None:
                continue
            writer.writerow(
                {
                    "protocol": result.name,
                    "generated_packets": final.generated_packets,
                    "delivered_packets": final.delivered_packets,
                    "lost_packets": final.lost_packets,
                    "PDR": f"{final.packet_delivery_ratio:.5f}",
                    "total_energy_consumed": f"{final.total_energy_consumed:.5f}",
                    "average_remaining_energy": f"{final.average_remaining_energy:.5f}",
                    "alive_nodes": final.alive_nodes,
                    "dead_nodes": final.dead_nodes,
                    "active_sensors": final.active_sensors,
                    "average_delay": f"{final.average_delay:.5f}",
                    "communication_overhead": final.communication_overhead,
                    "first_dead_node_time": _optional_float(result.first_dead_node_time),
                    "first_unable_to_send_time": _optional_float(result.first_unable_to_send_time),
                    "network_lifetime": f"{result.network_lifetime:.5f}",
                    "parent_changes": final.parent_changes,
                }
            )


def export_comparison_table(comparison_rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Metric", "Traditional RPL", "Q-learning RL-RPL", "Better Approach"],
        )
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow(row)


def _optional_float(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.5f}"
