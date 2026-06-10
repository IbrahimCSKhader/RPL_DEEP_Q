import random
from copy import deepcopy
from pathlib import Path

from data_logger import DecisionLogger, PacketLogger, export_comparison_table, export_summary_metrics
from metrics import SimulationResults, TimeStepMetrics
from network import Network, SimulationConfig
from plots import plot_comparison, plot_single_result
from rl_agent import QLearningRPL
from routing import TraditionalRPL, estimate_link_quality


DATASET_DIR = Path("dataset")
RESULTS_DIR = Path("results")


def run_simulation(
    network: Network,
    routing_protocol,
    seed: int,
    packet_logger: PacketLogger | None = None,
    decision_logger: DecisionLogger | None = None,
    snapshot_callback=None,
) -> SimulationResults:
    rng = random.Random(seed)
    config = network.config
    initial_energy = network.initial_total_sensor_energy()
    results = SimulationResults(name=routing_protocol.name)
    next_sensing_time = {node.node_id: 0.0 for node in network.sensor_nodes()}
    protocol_prefix = "RPL" if isinstance(routing_protocol, TraditionalRPL) else "RL"

    packet_counter = 0
    total_generated = 0
    total_delivered = 0
    total_lost = 0
    total_delay = 0.0
    communication_overhead = 0
    total_steps = int(config.duration_seconds / config.time_step)

    for step in range(total_steps + 1):
        current_time = round(step * config.time_step, 5)
        recent_packets = []

        for source in sorted(network.alive_sensor_nodes(), key=lambda node: node.node_id):
            if current_time + 1e-9 < next_sensing_time[source.node_id]:
                continue

            packet_counter += 1
            total_generated += 1
            source.packets_sent += 1
            next_sensing_time[source.node_id] += config.sensing_interval

            packet = generate_temperature_packet(
                packet_id=f"{protocol_prefix}-{packet_counter:05d}",
                source_sensor_id=source.node_id,
                protocol_mode=routing_protocol.name,
                generation_time=current_time,
                rng=rng,
            )
            packet_result = forward_packet(
                network,
                source.node_id,
                routing_protocol,
                packet,
                current_time,
                rng,
                packet_logger=packet_logger,
                decision_logger=decision_logger,
            )
            communication_overhead += packet_result["routing_decisions"]
            total_delay += packet_result["delay"] if packet_result["delivered"] else 0.0

            if packet_result["delivered"]:
                total_delivered += 1
                source.packets_delivered += 1
            else:
                total_lost += 1
                source.packets_dropped += 1

            recent_packets.append(packet_result["packet_summary"])

        first_unable = network.update_unable_to_send_times(current_time)
        if results.first_unable_to_send_time is None and first_unable is not None:
            results.first_unable_to_send_time = first_unable

        update_queue_loads(network)
        metric = calculate_metrics(
            step=step,
            current_time=current_time,
            network=network,
            initial_energy=initial_energy,
            total_generated=total_generated,
            total_delivered=total_delivered,
            total_lost=total_lost,
            total_delay=total_delay,
            communication_overhead=communication_overhead,
            recent_packets=recent_packets,
        )
        results.rounds.append(metric)

        if results.first_dead_node_time is None and metric.dead_nodes > 0:
            results.first_dead_node_time = current_time
        if results.fifty_percent_dead_time is None and metric.dead_nodes >= config.num_nodes / 2:
            results.fifty_percent_dead_time = current_time
        if results.no_active_sensor_time is None and metric.active_sensors == 0:
            results.no_active_sensor_time = current_time

        if snapshot_callback:
            snapshot_callback(current_time, network, metric, recent_packets)

    return results


def generate_temperature_packet(
    packet_id: str,
    source_sensor_id: int,
    protocol_mode: str,
    generation_time: float,
    rng: random.Random,
) -> dict:
    temperature = rng.uniform(20.0, 35.0)
    return {
        "packet_id": packet_id,
        "source_sensor_id": source_sensor_id,
        "temperature_value": temperature,
        "generation_time": generation_time,
        "current_hop": 0,
        "selected_parent": "",
        "protocol_mode": protocol_mode,
        "status": "generated",
    }


def select_parent_traditional_rpl(
    routing_protocol: TraditionalRPL,
    network: Network,
    node_id: int,
    current_time: float,
) -> int | None:
    return routing_protocol.select_parent(network, node_id, current_time)


def select_parent_rl_rpl(
    routing_protocol: QLearningRPL,
    network: Network,
    node_id: int,
    current_time: float,
) -> int | None:
    return routing_protocol.select_parent(network, node_id, current_time)


def forward_packet(
    network: Network,
    source_id: int,
    routing_protocol,
    packet: dict,
    current_time: float,
    rng: random.Random,
    packet_logger: PacketLogger | None = None,
    decision_logger: DecisionLogger | None = None,
) -> dict:
    current_id = source_id
    route_path = [source_id]
    visited = set()
    delay = 0.0
    energy_consumed = 0.0
    routing_decisions = 0
    route_states = []
    delivered = False
    lost = False
    status = "failed"
    max_hops = len(network.nodes)

    for hop_number in range(1, max_hops + 1):
        current = network.nodes[current_id]
        if current.is_root:
            delivered = True
            status = "delivered"
            break
        if not current.alive or current_id in visited:
            lost = True
            status = "failed_path_broken"
            break

        visited.add(current_id)
        old_state = routing_protocol.state_for(network, current_id) if hasattr(routing_protocol, "state_for") else None
        parent_id = (
            select_parent_traditional_rpl(routing_protocol, network, current_id, current_time)
            if isinstance(routing_protocol, TraditionalRPL)
            else select_parent_rl_rpl(routing_protocol, network, current_id, current_time)
        )
        decision_snapshot = deepcopy(getattr(routing_protocol, "last_decision", None) or {})
        old_state = decision_snapshot.get("state_tuple", old_state)
        routing_decisions += 1

        if parent_id is not None:
            if current.parent is not None and current.parent != parent_id:
                current.parent_changes += 1
            current.parent = parent_id

        route_state = {
            "node_id": current_id,
            "parent_id": parent_id,
            "old_state": old_state,
            "decision": decision_snapshot,
        }
        route_states.append(route_state)

        if parent_id is None:
            status = "failed_no_parent"
            _log_packet_hop(packet_logger, packet, hop_number, current_id, "", "", "", current.energy, current.energy, 0.0, status)
            lost = True
            break

        parent = network.nodes[parent_id]
        if not parent.alive:
            status = "failed_dead_parent"
            _log_packet_hop(packet_logger, packet, hop_number, current_id, parent_id, "", parent.queue_load, current.energy, current.energy, 0.0, status)
            lost = True
            break

        energy_cost = calculate_energy_cost(network, current_id, parent_id)
        required_tx_energy = energy_cost["tx_cost"] + network.config.processing_energy
        current_energy_before = current.energy

        if current.energy < required_tx_energy:
            consumed = current.consume_energy(current.energy)
            energy_consumed += consumed
            status = "failed_energy"
            _log_packet_hop(packet_logger, packet, hop_number, current_id, parent_id, "", parent.queue_load, current_energy_before, current.energy, 0.0, status)
            lost = True
            break

        tx_consumed = current.consume_energy(required_tx_energy)
        rx_required = network.config.rx_energy + network.config.processing_energy
        parent_had_energy = parent.is_root or parent.energy >= rx_required
        rx_consumed = parent.consume_energy(rx_required)
        energy_consumed += tx_consumed + rx_consumed
        current.packets_forwarded += 1
        parent.packets_received += 1
        parent.queue_load += 1.0

        link_quality = estimate_link_quality(network, current_id, parent_id)
        hop_delay = calculate_delay(network, parent_id, link_quality)
        loss_probability = calculate_packet_loss(network, parent_id, link_quality)
        delay += hop_delay
        route_path.append(parent_id)

        if not parent_had_energy:
            status = "failed_parent_energy"
            lost = True
        elif not parent.is_root and not parent.alive:
            status = "failed_dead_parent"
            lost = True
        elif rng.random() < loss_probability:
            status = "lost"
            lost = True
        elif parent.is_root:
            status = "delivered"
            delivered = True
        else:
            status = "forwarded"

        packet["current_hop"] = hop_number
        packet["selected_parent"] = parent_id
        packet["status"] = status
        _log_packet_hop(
            packet_logger,
            packet,
            hop_number,
            current_id,
            parent_id,
            link_quality,
            parent.queue_load,
            current_energy_before,
            current.energy,
            hop_delay,
            status,
        )

        route_state.update(
            {
                "link_quality": link_quality,
                "queue_load": parent.queue_load,
                "hop_delay": hop_delay,
                "loss_probability": loss_probability,
                "energy_consumed": tx_consumed + rx_consumed,
            }
        )

        if delivered or lost:
            break

        current_id = parent_id

    else:
        lost = True
        status = "failed_max_hops"

    if delivered:
        packet["status"] = "delivered"
    elif status == "forwarded":
        packet["status"] = "failed_path_broken"
    else:
        packet["status"] = status

    for route_state in route_states:
        learning_result = update_q_value(
            routing_protocol,
            network,
            route_state["node_id"],
            route_state["parent_id"],
            route_state["old_state"],
            delivered,
            delay,
            lost,
        )
        if decision_logger:
            decision_logger.log_decision(
                routing_protocol,
                current_time,
                route_state["node_id"],
                route_state["parent_id"],
                decision=route_state["decision"],
                learning_result=learning_result,
            )

    return {
        "delivered": delivered,
        "lost": not delivered,
        "delay": delay,
        "energy_consumed": energy_consumed,
        "routing_decisions": routing_decisions,
        "packet_summary": {
            "packet_id": packet["packet_id"],
            "source_sensor_id": source_id,
            "temperature": packet["temperature_value"],
            "generation_time": packet["generation_time"],
            "route_path": route_path,
            "selected_route_path": " -> ".join(str(node_id) for node_id in route_path),
            "status": packet["status"],
            "delay": delay,
            "energy_consumed": energy_consumed,
        },
    }


def calculate_energy_cost(network: Network, current_id: int, parent_id: int) -> dict:
    current = network.nodes[current_id]
    parent = network.nodes[parent_id]
    hop_distance = current.distance_to(parent)
    distance_ratio = min(hop_distance / network.config.transmission_range, 1.0)
    tx_cost = network.config.tx_energy + network.config.tx_distance_energy * (distance_ratio**2)
    return {
        "hop_distance": hop_distance,
        "distance_ratio": distance_ratio,
        "tx_cost": tx_cost,
    }


def calculate_delay(network: Network, parent_id: int, link_quality: float) -> float:
    parent = network.nodes[parent_id]
    return (
        network.config.base_hop_delay
        + (1.0 - link_quality)
        + parent.queue_load * network.config.congestion_delay_factor
        + network.config.propagation_delay
    )


def calculate_packet_loss(network: Network, parent_id: int, link_quality: float) -> float:
    parent = network.nodes[parent_id]
    loss_probability = (
        network.config.packet_loss_base
        + (1.0 - link_quality) * 0.08
        + parent.queue_load * network.config.packet_loss_queue_factor
    )
    return min(loss_probability, 0.75)


def update_q_value(
    routing_protocol,
    network: Network,
    node_id: int,
    parent_id: int | None,
    old_state,
    delivered: bool,
    delay: float,
    lost: bool,
) -> dict | None:
    if not hasattr(routing_protocol, "observe"):
        return None
    return routing_protocol.observe(network, node_id, parent_id, old_state, delivered, delay, lost)


def update_queue_loads(network: Network) -> None:
    network.decay_queues()


def calculate_metrics(
    step: int,
    current_time: float,
    network: Network,
    initial_energy: float,
    total_generated: int,
    total_delivered: int,
    total_lost: int,
    total_delay: float,
    communication_overhead: int,
    recent_packets: list[dict],
) -> TimeStepMetrics:
    alive_nodes = len(network.alive_sensor_nodes())
    active_sensors = len(network.active_sensor_nodes())
    dead_nodes = network.dead_sensor_count()
    pdr = total_delivered / total_generated if total_generated else 0.0
    average_delay = total_delay / total_delivered if total_delivered else 0.0
    latest_packet = recent_packets[-1] if recent_packets else {}
    return TimeStepMetrics(
        step=step,
        time=current_time,
        total_energy_consumed=initial_energy - network.total_sensor_energy(),
        average_remaining_energy=network.average_remaining_energy(),
        alive_nodes=alive_nodes,
        active_sensors=active_sensors,
        dead_nodes=dead_nodes,
        generated_packets=total_generated,
        delivered_packets=total_delivered,
        lost_packets=total_lost,
        packet_delivery_ratio=pdr,
        average_delay=average_delay,
        communication_overhead=communication_overhead,
        parent_changes=network.total_parent_changes(),
        cumulative_delivered_packets=total_delivered,
        cumulative_lost_packets=total_lost,
        current_packet_id=latest_packet.get("packet_id", ""),
        current_source_sensor_id=latest_packet.get("source_sensor_id", ""),
        current_temperature=latest_packet.get("temperature", ""),
        selected_route_path=latest_packet.get("selected_route_path", ""),
        last_packet_status=latest_packet.get("status", ""),
        last_packet_delay=latest_packet.get("delay", 0.0),
        last_packet_energy=latest_packet.get("energy_consumed", 0.0),
    )


def generate_comparison_table(traditional_result: SimulationResults, rl_result: SimulationResults) -> list[dict]:
    traditional = result_summary(traditional_result)
    rl = result_summary(rl_result)
    rows = []
    metric_specs = [
        ("Generated Packets", "generated_packets", "higher", "int"),
        ("Delivered Packets", "delivered_packets", "higher", "int"),
        ("Lost Packets", "lost_packets", "lower", "int"),
        ("PDR", "PDR", "higher", "percent"),
        ("Total Energy Consumed", "total_energy_consumed", "lower", "float"),
        ("Average Remaining Energy", "average_remaining_energy", "higher", "float"),
        ("Alive Nodes", "alive_nodes", "higher", "int"),
        ("Dead Nodes", "dead_nodes", "lower", "int"),
        ("Active Sensors", "active_sensors", "higher", "int"),
        ("Average Delay", "average_delay", "lower", "float"),
        ("Communication Overhead", "communication_overhead", "lower", "int"),
        ("First Dead Node Time", "first_dead_node_time", "higher", "time"),
        ("First Unable-to-Send Sensor Time", "first_unable_to_send_time", "higher", "time"),
        ("Network Lifetime", "network_lifetime", "higher", "time"),
        ("Parent Changes", "parent_changes", "lower", "int"),
    ]

    for label, key, direction, value_type in metric_specs:
        traditional_value = traditional[key]
        rl_value = rl[key]
        rows.append(
            {
                "Metric": label,
                "Traditional RPL": format_metric_value(traditional_value, value_type),
                "Q-learning RL-RPL": format_metric_value(rl_value, value_type),
                "Better Approach": better_approach(traditional_value, rl_value, direction),
            }
        )
    return rows


def build_comparison_payload(traditional_result: SimulationResults, rl_result: SimulationResults) -> dict:
    rows = generate_comparison_table(traditional_result, rl_result)
    rl_wins = sum(1 for row in rows if row["Better Approach"] == "Q-learning RL-RPL")
    rpl_wins = sum(1 for row in rows if row["Better Approach"] == "Traditional RPL")
    ties = sum(1 for row in rows if row["Better Approach"] == "Tie")
    if rl_wins == rpl_wins:
        overall = "Tie"
    else:
        overall = "Q-learning RL-RPL" if rl_wins > rpl_wins else "Traditional RPL"
    return {
        "traditional": result_summary(traditional_result),
        "rl_rpl": result_summary(rl_result),
        "rows": rows,
        "overall_winner": overall,
        "rl_rpl_wins": rl_wins,
        "traditional_wins": rpl_wins,
        "ties": ties,
    }


def result_summary(result: SimulationResults) -> dict:
    final = result.final_metric
    if final is None:
        return {}
    return {
        "name": result.name,
        "generated_packets": final.generated_packets,
        "delivered_packets": final.delivered_packets,
        "lost_packets": final.lost_packets,
        "PDR": final.packet_delivery_ratio,
        "total_energy_consumed": final.total_energy_consumed,
        "average_remaining_energy": final.average_remaining_energy,
        "alive_nodes": final.alive_nodes,
        "dead_nodes": final.dead_nodes,
        "active_sensors": final.active_sensors,
        "average_delay": final.average_delay,
        "communication_overhead": final.communication_overhead,
        "first_dead_node_time": result.first_dead_node_time,
        "first_unable_to_send_time": result.first_unable_to_send_time,
        "network_lifetime": result.network_lifetime,
        "parent_changes": final.parent_changes,
        "fifty_percent_dead_time": result.fifty_percent_dead_time,
        "no_active_sensor_time": result.no_active_sensor_time,
    }


def better_approach(traditional_value, rl_value, direction: str) -> str:
    traditional_score = _numeric_or_default(traditional_value, -1.0 if direction == "higher" else float("inf"))
    rl_score = _numeric_or_default(rl_value, -1.0 if direction == "higher" else float("inf"))
    if abs(traditional_score - rl_score) < 1e-9:
        return "Tie"
    if direction == "higher":
        return "Traditional RPL" if traditional_score > rl_score else "Q-learning RL-RPL"
    return "Traditional RPL" if traditional_score < rl_score else "Q-learning RL-RPL"


def format_metric_value(value, value_type: str) -> str:
    if value is None:
        return "N/A"
    if value_type == "percent":
        return f"{float(value) * 100:.2f}%"
    if value_type == "int":
        return str(int(value))
    if value_type == "time":
        return f"{float(value):.1f} s"
    return f"{float(value):.5f}"


def export_logs(results: list[SimulationResults], comparison_rows: list[dict]) -> None:
    DATASET_DIR.mkdir(exist_ok=True)
    export_summary_metrics(results, DATASET_DIR / "summary_metrics.csv")
    export_comparison_table(comparison_rows, DATASET_DIR / "comparison_table.csv")


def print_summary(results: list[SimulationResults]) -> None:
    comparison_rows = generate_comparison_table(results[0], results[1])
    comparison = build_comparison_payload(results[0], results[1])
    print("\nTraditional RPL vs Q-learning RL-RPL")
    print("-" * 105)
    print(f"{'Metric':<36} {'Traditional RPL':>20} {'Q-learning RL-RPL':>22} {'Better':>20}")
    print("-" * 105)
    for row in comparison_rows:
        print(
            f"{row['Metric']:<36} {row['Traditional RPL']:>20} "
            f"{row['Q-learning RL-RPL']:>22} {row['Better Approach']:>20}"
        )
    print("-" * 105)
    print(f"Overall better approach: {comparison['overall_winner']}")
    print("CSV files were saved in dataset; graphs were saved in results.")


def _log_packet_hop(
    packet_logger: PacketLogger | None,
    packet: dict,
    hop_number: int,
    current_node,
    selected_parent,
    link_quality,
    queue_load,
    energy_before,
    energy_after,
    hop_delay,
    packet_status: str,
) -> None:
    if not packet_logger:
        return
    packet_logger.log_packet_hop(
        packet,
        {
            "hop_number": hop_number,
            "current_node": current_node,
            "selected_parent": selected_parent,
            "link_quality": link_quality,
            "queue_load": queue_load,
            "energy_before": energy_before,
            "energy_after": energy_after,
            "hop_delay": hop_delay,
            "packet_status": packet_status,
        },
    )


def _numeric_or_default(value, default: float) -> float:
    if value is None:
        return default
    return float(value)


def main() -> None:
    DATASET_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    config = SimulationConfig()
    base_network = Network.create_random(config)

    traditional_network = base_network.clone()
    rl_network = base_network.clone()

    with PacketLogger(DATASET_DIR / "packet_log.csv") as packet_logger, DecisionLogger(
        DATASET_DIR / "routing_decisions.csv"
    ) as decision_logger:
        traditional_result = run_simulation(
            traditional_network,
            TraditionalRPL(),
            config.seed,
            packet_logger=packet_logger,
            decision_logger=decision_logger,
        )
        rl_result = run_simulation(
            rl_network,
            QLearningRPL(seed=config.seed),
            config.seed,
            packet_logger=packet_logger,
            decision_logger=decision_logger,
        )

    comparison_rows = generate_comparison_table(traditional_result, rl_result)
    export_logs([traditional_result, rl_result], comparison_rows)
    plot_single_result(traditional_result)
    plot_comparison([traditional_result, rl_result])
    print_summary([traditional_result, rl_result])


if __name__ == "__main__":
    main()
