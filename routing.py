from network import Network


class TraditionalRPL:
    name = "Traditional RPL"

    def __init__(self):
        self.last_decision = None

    def select_parent(self, network: Network, node_id: int, current_time: float) -> int | None:
        candidates = network.candidate_parents(node_id)
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
        root = network.root
        selected = min(candidates, key=lambda candidate_id: network.nodes[candidate_id].distance_to(root))
        candidate_text = []
        candidate_records = []
        for candidate_id in candidates:
            candidate = network.nodes[candidate_id]
            energy_ratio = parent_energy_ratio(candidate)
            distance_to_root = candidate.distance_to(root)
            distance_ratio = parent_distance_ratio(network, candidate_id)
            queue_ratio = parent_queue_ratio(candidate)
            link_quality = estimate_link_quality(network, node_id, candidate_id)
            state_value = 1.0 - distance_ratio
            selection_value = state_value
            candidate_text.append(
                f"{candidate_id}:dist_root={distance_to_root:.2f},dist_ratio={distance_ratio:.3f},"
                f"energy={energy_ratio:.3f},queue={queue_ratio:.3f},link={link_quality:.3f}"
            )
            candidate_records.append(
                {
                    "candidate_parent": candidate_id,
                    "energy_ratio": energy_ratio,
                    "distance_ratio": distance_ratio,
                    "queue_ratio": queue_ratio,
                    "link_quality": link_quality,
                    "q_value": "",
                    "state_value": state_value,
                    "selection_value": selection_value,
                }
            )
        selected_parent = network.nodes[selected]
        selected_distance = selected_parent.distance_to(root)
        self.last_decision = {
            "protocol": self.name,
            "state": "",
            "rule": "choose candidate with minimum distance to ROOT",
            "candidates": "; ".join(candidate_text),
            "candidate_records": candidate_records,
            "selection_mode": "traditional",
            "candidate_parent": selected,
            "action": selected,
            "candidate_energy_ratio": parent_energy_ratio(selected_parent),
            "candidate_distance_ratio": parent_distance_ratio(network, selected),
            "candidate_queue_ratio": parent_queue_ratio(selected_parent),
            "candidate_link_quality": estimate_link_quality(network, node_id, selected),
            "equations": (
                f"distance(candidate, ROOT) < distance(node, ROOT); "
                f"selected parent {selected} because distance({selected}, ROOT)="
                f"{selected_distance:.3f} is the minimum valid distance."
            ),
        }
        return selected

    def observe(self, *args, **kwargs) -> None:
        return None


def estimate_link_quality(network: Network, node_id: int, parent_id: int) -> float:
    parent = network.nodes[parent_id]
    distance_ratio = parent_distance_ratio(network, parent_id)
    link_quality = (1.0 - distance_ratio) - (parent.queue_load * 0.05)
    return max(0.05, min(link_quality, 1.0))


def parent_score(network: Network, node_id: int, parent_id: int) -> tuple[float, float, float, float]:
    parent = network.nodes[parent_id]
    energy_ratio = parent_energy_ratio(parent)
    distance_ratio = parent_distance_ratio(network, parent_id)
    link_quality = estimate_link_quality(network, node_id, parent_id)
    queue_ratio = parent_queue_ratio(parent)
    return energy_ratio, distance_ratio, link_quality, queue_ratio


def parent_energy_ratio(parent) -> float:
    if parent.is_sink:
        return 1.0
    if parent.initial_energy <= 0:
        return 0.0
    return max(0.0, min(parent.energy / parent.initial_energy, 1.0))


def parent_distance_ratio(network: Network, parent_id: int) -> float:
    parent = network.nodes[parent_id]
    return parent.distance_to(network.root) / max(network.config.area_width, network.config.area_height)


def parent_queue_ratio(parent) -> float:
    return min(parent.queue_load / 10.0, 1.0)
