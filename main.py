import random

from metrics import RoundMetrics, SimulationResults
from network import Network, SimulationConfig
from plots import plot_comparison, plot_single_result
from rl_agent import QLearningRPL
from routing import TraditionalRPL, estimate_link_quality


def run_simulation(network: Network, routing_protocol, seed: int, decision_logger=None, snapshot_callback=None) -> SimulationResults:
    rng = random.Random(seed)
    config = network.config
    initial_energy = network.total_sensor_energy()
    results = SimulationResults(name=routing_protocol.name)

    for round_number in range(1, config.rounds + 1):
        generated_packets = 0
        delivered_packets = 0
        delivered_delays = []

        for source in list(network.alive_sensor_nodes()):
            generated_packets += 1
            source.packets_sent += 1
            delivered, delay, route_states, lost = transmit_packet(
                network, source.node_id, routing_protocol, round_number, rng
            )
            if delivered:
                delivered_packets += 1
                delivered_delays.append(delay)
                source.packets_delivered += 1
            else:
                source.packets_dropped += 1

            for route_state in route_states:
                learning_result = routing_protocol.observe(
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
                        round_number,
                        source.node_id,
                        network,
                        route_state["node_id"],
                        route_state["parent_id"],
                        delivered,
                        lost,
                        delay,
                        decision=route_state["decision"],
                        learning_result=learning_result,
                        hop_details=route_state,
                    )

        if hasattr(routing_protocol, "end_round"):
            routing_protocol.end_round()
        network.decay_queues()

        alive_nodes = len(network.alive_sensor_nodes())
        pdr = delivered_packets / generated_packets if generated_packets else 0.0
        average_delay = sum(delivered_delays) / len(delivered_delays) if delivered_delays else 0.0
        results.rounds.append(
            RoundMetrics(
                round_number=round_number,
                total_energy_consumed=initial_energy - network.total_sensor_energy(),
                average_remaining_energy=network.average_remaining_energy(),
                alive_nodes=alive_nodes,
                generated_packets=generated_packets,
                delivered_packets=delivered_packets,
                packet_delivery_ratio=pdr,
                average_delay=average_delay,
            )
        )
        if snapshot_callback:
            snapshot_callback(round_number, network, results.rounds[-1])

        if alive_nodes == 0:
            break

    return results


def transmit_packet(
    network: Network,
    source_id: int,
    routing_protocol,
    round_number: int,
    rng: random.Random,
):
    config = network.config
    current_id = source_id
    visited = set()
    delay = 0.0
    route_states = []
    lost = False
    max_hops = len(network.nodes)

    for _ in range(max_hops):
        current = network.nodes[current_id]
        if current.is_sink:
            return True, delay, route_states, lost
        if not current.alive or current_id in visited:
            return False, delay, route_states, True

        visited.add(current_id)
        old_state = routing_protocol.state_for(network, current_id) if hasattr(routing_protocol, "state_for") else None
        parent_id = routing_protocol.select_parent(network, current_id, round_number)
        decision_snapshot = dict(getattr(routing_protocol, "last_decision", None) or {})
        current.parent = parent_id
        route_state = {
            "node_id": current_id,
            "parent_id": parent_id,
            "old_state": old_state,
            "decision": decision_snapshot,
            "hop_distance": "",
            "hop_distance_ratio": "",
            "tx_cost": "",
            "current_energy_consumed": "",
            "parent_energy_consumed": "",
            "energy_consumed": "",
            "link_quality": "",
            "loss_probability": "",
            "delay_increment": "",
        }
        route_states.append(route_state)

        if parent_id is None:
            return False, delay, route_states, True

        parent = network.nodes[parent_id]
        if not parent.is_sink and not parent.alive:
            return False, delay, route_states, True

        hop_distance = current.distance_to(parent)
        hop_distance_ratio = hop_distance / config.transmission_range
        tx_cost = config.tx_energy + config.tx_distance_energy * (hop_distance_ratio**2)
        current_energy_before = current.energy
        parent_energy_before = parent.energy
        current.consume_energy(tx_cost + config.processing_energy)
        parent.consume_energy(config.rx_energy + config.processing_energy)
        current_energy_consumed = current_energy_before - current.energy
        parent_energy_consumed = 0.0 if parent.is_sink else parent_energy_before - parent.energy
        current.packets_forwarded += 1
        parent.packets_received += 1
        parent.queue_load += 1.0

        link_quality = estimate_link_quality(network, current_id, parent_id)
        loss_probability = min(
            0.75,
            config.packet_loss_base
            + (1.0 - link_quality) * 0.08
            + parent.queue_load * config.packet_loss_queue_factor,
        )
        delay_increment = (
            config.base_hop_delay
            + (1.0 - link_quality)
            + parent.queue_load * config.congestion_delay_factor
        )
        delay += delay_increment
        route_state.update(
            {
                "hop_distance": hop_distance,
                "hop_distance_ratio": hop_distance_ratio,
                "tx_cost": tx_cost,
                "current_energy_consumed": current_energy_consumed,
                "parent_energy_consumed": parent_energy_consumed,
                "energy_consumed": current_energy_consumed + parent_energy_consumed,
                "link_quality": link_quality,
                "loss_probability": loss_probability,
                "delay_increment": delay_increment,
            }
        )

        if rng.random() < loss_probability:
            lost = True
            return False, delay, route_states, lost

        if parent.is_sink:
            return True, delay, route_states, lost

        current_id = parent_id

    return False, delay, route_states, True


def print_summary(results: list[SimulationResults]) -> None:
    print("\nSimulation Summary")
    print("-" * 92)
    print(
        f"{'Protocol':<20} {'Lifetime':>10} {'First death':>12} "
        f"{'Final alive':>12} {'PDR':>10} {'Avg energy':>12}"
    )
    print("-" * 92)
    for result in results:
        first_death = result.first_node_death_round
        final_alive = result.rounds[-1].alive_nodes if result.rounds else 0
        print(
            f"{result.name:<20} {result.network_lifetime:>10} "
            f"{str(first_death or 'None'):>12} {final_alive:>12} "
            f"{result.final_pdr:>10.3f} {result.final_average_energy:>12.4f}"
        )
    print("-" * 92)
    print("Graphs were saved in the results folder.")


def main() -> None:
    config = SimulationConfig()
    base_network = Network.create_random(config)

    traditional_network = base_network.clone()
    rl_network = base_network.clone()

    traditional_result = run_simulation(traditional_network, TraditionalRPL(), config.seed)
    rl_result = run_simulation(rl_network, QLearningRPL(seed=config.seed), config.seed)

    plot_single_result(traditional_result)
    plot_comparison([traditional_result, rl_result])
    print_summary([traditional_result, rl_result])


if __name__ == "__main__":
    main()
