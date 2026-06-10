from dataclasses import dataclass, field
from math import hypot


@dataclass
class Node:
    node_id: int
    x: float
    y: float
    initial_energy: float
    is_sink: bool = False
    energy: float = field(init=False)
    alive: bool = field(default=True, init=False)
    neighbors: list[int] = field(default_factory=list)
    parent: int | None = None
    packets_sent: int = 0
    packets_received: int = 0
    packets_forwarded: int = 0
    packets_delivered: int = 0
    packets_dropped: int = 0
    queue_load: float = 0.0
    parent_changes: int = 0
    unable_to_send_time: float | None = None

    def __post_init__(self) -> None:
        self.energy = self.initial_energy
        if self.is_sink:
            self.energy = float("inf")

    def distance_to(self, other: "Node") -> float:
        return hypot(self.x - other.x, self.y - other.y)

    @property
    def is_root(self) -> bool:
        return self.is_sink

    def consume_energy(self, amount: float) -> float:
        if self.is_sink or not self.alive:
            return 0.0
        before = self.energy
        self.energy = max(0.0, self.energy - amount)
        if self.energy <= 0.0:
            self.alive = False
            self.parent = None
        return before - self.energy

    def reset_counters(self) -> None:
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_forwarded = 0
        self.packets_delivered = 0
        self.packets_dropped = 0
        self.queue_load = 0.0
        self.parent_changes = 0
        self.unable_to_send_time = None
