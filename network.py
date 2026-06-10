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
    simulation_duration: float = 420.0
    time_step: float = 1.0
    sensing_interval: float = 5.0
    rounds: int = 0
    tx_energy: float = 0.006
    tx_distance_energy: float = 0.008
    rx_energy: float = 0.003
    processing_energy: float = 0.001
    base_hop_delay: float = 1.0
    propagation_delay: float = 0.02
    congestion_delay_factor: float = 0.15
    packet_loss_base: float = 0.02
    packet_loss_queue_factor: float = 0.03
    queue_decay_rate: float = 0.65
    seed: int = 42

    @property
    def duration_seconds(self) -> float:
        if self.rounds > 0:
            return self.rounds * self.time_step
        return self.simulation_duration


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

    @property
    def root(self) -> Node:
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

    def sensor_nodes(self) -> list[Node]:
        return [node for node in self.nodes.values() if not node.is_sink]

    def candidate_parents(self, node_id: int) -> list[int]:
        node = self.nodes[node_id]
        if not node.alive:
            return []

        node_root_distance = node.distance_to(self.root)
        candidates = []

        for neighbor_id in node.neighbors:
            neighbor = self.nodes[neighbor_id]
            if (
                neighbor.alive
                and node.distance_to(neighbor) <= self.config.transmission_range
                and neighbor.distance_to(self.root) < node_root_distance
            ):
                candidates.append(neighbor_id)

        return candidates

    def decay_queues(self) -> None:
        for node in self.nodes.values():
            node.queue_load = max(0.0, node.queue_load - self.config.queue_decay_rate)

    def can_deliver_to_root(self, node_id: int, visited: set[int] | None = None) -> bool:
        node = self.nodes[node_id]
        if node.is_sink:
            return True
        if not node.alive:
            return False

        visited = visited or set()
        if node_id in visited:
            return False
        visited.add(node_id)

        for parent_id in self.candidate_parents(node_id):
            parent = self.nodes[parent_id]
            if parent.is_sink or self.can_deliver_to_root(parent_id, visited):
                return True
        return False

    def active_sensor_nodes(self) -> list[Node]:
        return [node for node in self.alive_sensor_nodes() if self.can_deliver_to_root(node.node_id)]

    def update_unable_to_send_times(self, current_time: float) -> float | None:
        first_time = None
        for node in self.sensor_nodes():
            unable = (not node.alive) or (not self.can_deliver_to_root(node.node_id))
            if unable and node.unable_to_send_time is None:
                node.unable_to_send_time = current_time
            if node.unable_to_send_time is not None:
                first_time = (
                    node.unable_to_send_time
                    if first_time is None
                    else min(first_time, node.unable_to_send_time)
                )
        return first_time

    def total_sensor_energy(self) -> float:
        return sum(node.energy for node in self.nodes.values() if not node.is_sink)

    def initial_total_sensor_energy(self) -> float:
        return sum(node.initial_energy for node in self.nodes.values() if not node.is_sink)

    def average_remaining_energy(self) -> float:
        alive_sensors = self.alive_sensor_nodes()
        if not alive_sensors:
            return 0.0
        return sum(node.energy for node in alive_sensors) / len(alive_sensors)

    def dead_sensor_count(self) -> int:
        return len([node for node in self.sensor_nodes() if not node.alive])

    def total_parent_changes(self) -> int:
        return sum(node.parent_changes for node in self.sensor_nodes())

    def reset_runtime_state(self) -> None:
        for node in self.nodes.values():
            node.reset_counters()
