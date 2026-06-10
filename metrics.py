from dataclasses import dataclass, field


@dataclass
class TimeStepMetrics:
    step: int
    time: float
    total_energy_consumed: float
    average_remaining_energy: float
    alive_nodes: int
    active_sensors: int
    dead_nodes: int
    generated_packets: int
    delivered_packets: int
    lost_packets: int
    packet_delivery_ratio: float
    average_delay: float
    communication_overhead: int
    parent_changes: int
    cumulative_delivered_packets: int
    cumulative_lost_packets: int
    current_packet_id: str = ""
    current_source_sensor_id: int | str = ""
    current_temperature: float | str = ""
    selected_route_path: str = ""
    last_packet_status: str = ""
    last_packet_delay: float = 0.0
    last_packet_energy: float = 0.0

    @property
    def round_number(self) -> int:
        return self.step


@dataclass
class SimulationResults:
    name: str
    rounds: list[TimeStepMetrics] = field(default_factory=list)
    first_dead_node_time: float | None = None
    first_unable_to_send_time: float | None = None
    fifty_percent_dead_time: float | None = None
    no_active_sensor_time: float | None = None

    @property
    def final_metric(self) -> TimeStepMetrics | None:
        return self.rounds[-1] if self.rounds else None

    @property
    def network_lifetime(self) -> float:
        if self.no_active_sensor_time is not None:
            return self.no_active_sensor_time
        if self.rounds:
            return self.rounds[-1].time
        return 0.0

    @property
    def first_node_death_round(self) -> float | None:
        return self.first_dead_node_time

    @property
    def final_pdr(self) -> float:
        final = self.final_metric
        return final.packet_delivery_ratio if final else 0.0

    @property
    def final_average_energy(self) -> float:
        final = self.final_metric
        return final.average_remaining_energy if final else 0.0

    @property
    def generated_packets(self) -> int:
        final = self.final_metric
        return final.generated_packets if final else 0

    @property
    def delivered_packets(self) -> int:
        final = self.final_metric
        return final.delivered_packets if final else 0

    @property
    def lost_packets(self) -> int:
        final = self.final_metric
        return final.lost_packets if final else 0
