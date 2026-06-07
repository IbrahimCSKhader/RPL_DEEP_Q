from dataclasses import dataclass, field


@dataclass
class RoundMetrics:
    round_number: int
    total_energy_consumed: float
    average_remaining_energy: float
    alive_nodes: int
    generated_packets: int
    delivered_packets: int
    packet_delivery_ratio: float
    average_delay: float


@dataclass
class SimulationResults:
    name: str
    rounds: list[RoundMetrics] = field(default_factory=list)

    @property
    def network_lifetime(self) -> int:
        for metric in self.rounds:
            if metric.alive_nodes == 0:
                return metric.round_number
        return self.rounds[-1].round_number if self.rounds else 0

    @property
    def first_node_death_round(self) -> int | None:
        if not self.rounds:
            return None
        initial_alive = self.rounds[0].alive_nodes
        for metric in self.rounds:
            if metric.alive_nodes < initial_alive:
                return metric.round_number
        return None

    @property
    def final_pdr(self) -> float:
        generated = sum(metric.generated_packets for metric in self.rounds)
        delivered = sum(metric.delivered_packets for metric in self.rounds)
        return delivered / generated if generated else 0.0

    @property
    def final_average_energy(self) -> float:
        return self.rounds[-1].average_remaining_energy if self.rounds else 0.0
