from network import Network


class TraditionalRPL:
    name = "Traditional RPL"

    def __init__(self):
        self.last_decision = None

    def select_parent(self, network: Network, node_id: int, round_number: int) -> int | None:
        candidates = network.candidate_parents(node_id)
        if not candidates:
            self.last_decision = {
                "protocol": self.name,
                "state": "",
                "rule": "drop: no candidate parent closer to sink",
                "candidates": "",
                "selection_mode": "drop",
                "equations": "No valid parent satisfies alive, in range, and closer-to-root constraints.",
            }
            return None
        sink = network.sink
        selected = min(candidates, key=lambda candidate_id: network.nodes[candidate_id].distance_to(sink))
        candidate_text = []
        for candidate_id in candidates:
            candidate = network.nodes[candidate_id]
            energy_ratio = parent_energy_ratio(candidate)
            distance_to_sink = candidate.distance_to(sink)
            distance_ratio = parent_distance_ratio(network, candidate_id)
            queue_ratio = parent_queue_ratio(candidate)
            link_quality = estimate_link_quality(network, node_id, candidate_id)
            candidate_text.append(
                f"{candidate_id}:dist_root={distance_to_sink:.2f},dist_ratio={distance_ratio:.3f},"
                f"energy={energy_ratio:.3f},queue={queue_ratio:.3f},link={link_quality:.3f}"
            )
        selected_parent = network.nodes[selected]
        selected_distance = selected_parent.distance_to(sink)
        self.last_decision = {
            "protocol": self.name,
            "state": "",
            "rule": "choose candidate with minimum distance to sink",
            "candidates": "; ".join(candidate_text),
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
    return parent.energy / parent.initial_energy


def parent_distance_ratio(network: Network, parent_id: int) -> float:
    parent = network.nodes[parent_id]
    return parent.distance_to(network.sink) / max(network.config.area_width, network.config.area_height)


def parent_queue_ratio(parent) -> float:
    return min(parent.queue_load / 10.0, 1.0)
