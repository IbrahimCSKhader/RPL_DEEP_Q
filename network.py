import random
from copy import deepcopy
from dataclasses import dataclass

from node import Node


@dataclass(frozen=True)
class SimulationConfig:
    num_nodes: int = 30
    area_width: float = 100.0
    area_height: float = 100.0
    transmission_range: float = 35.0
    initial_energy: float = 2.0
    rounds: int = 250
    tx_energy: float = 0.006
    tx_distance_energy: float = 0.008
    rx_energy: float = 0.003
    processing_energy: float = 0.001
    base_hop_delay: float = 1.0
    congestion_delay_factor: float = 0.15
    packet_loss_base: float = 0.02
    packet_loss_queue_factor: float = 0.03
    queue_decay_factor: float = 0.85
    seed: int = 42


class Network:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.nodes: dict[int, Node] = {}
        self.sink_id = 0

    @classmethod
    def create_random(cls, config: SimulationConfig) -> "Network":
        rng = random.Random(config.seed)
        network = cls(config)
        sink = Node(
            node_id=0,
            x=config.area_width / 2,
            y=config.area_height / 2,
            initial_energy=float("inf"),
            is_sink=True,
        )
        network.nodes[sink.node_id] = sink

        for node_id in range(1, config.num_nodes + 1):
            network.nodes[node_id] = Node(
                node_id=node_id,
                x=rng.uniform(0, config.area_width),
                y=rng.uniform(0, config.area_height),
                initial_energy=config.initial_energy,
            )

        network.discover_neighbors()
        return network

    def clone(self) -> "Network":
        return deepcopy(self)

    @property
    def sink(self) -> Node:
        return self.nodes[self.sink_id]

    def discover_neighbors(self) -> None:
        for node in self.nodes.values():
            node.neighbors.clear()

        node_list = list(self.nodes.values())
        for i, node in enumerate(node_list):
            for other in node_list[i + 1 :]:
                if node.distance_to(other) <= self.config.transmission_range:
                    node.neighbors.append(other.node_id)
                    other.neighbors.append(node.node_id)

    def alive_sensor_nodes(self) -> list[Node]:
        return [node for node in self.nodes.values() if not node.is_sink and node.alive]

    def candidate_parents(self, node_id: int) -> list[int]:
        node = self.nodes[node_id]
        node_sink_distance = node.distance_to(self.sink)
        candidates = []

        for neighbor_id in node.neighbors:
            neighbor = self.nodes[neighbor_id]
            if (
                neighbor.alive
                and node.distance_to(neighbor) <= self.config.transmission_range
                and neighbor.distance_to(self.sink) < node_sink_distance
            ):
                candidates.append(neighbor_id)

        return candidates

    def decay_queues(self) -> None:
        for node in self.nodes.values():
            node.queue_load *= self.config.queue_decay_factor

    def total_sensor_energy(self) -> float:
        return sum(node.energy for node in self.nodes.values() if not node.is_sink)

    def average_remaining_energy(self) -> float:
        sensors = [node for node in self.nodes.values() if not node.is_sink]
        return sum(node.energy for node in sensors) / len(sensors)

    def reset_runtime_state(self) -> None:
        for node in self.nodes.values():
            node.reset_counters()
