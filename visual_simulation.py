import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from data_logger import DecisionLogger
from main import run_simulation
from network import Network, SimulationConfig
from rl_agent import QLearningRPL
from routing import TraditionalRPL


OUTPUT_DIR = Path("results")
DATASET_DIR = Path("dataset")
METRIC_WEIGHTS = {
    "energy_consumption": 1,
    "energy_per_delivered_packet": 3,
    "average_remaining_energy": 1,
    "alive_nodes": 1,
    "delivered_packets": 2,
    "packet_delivery_ratio": 3,
    "average_delay": 2,
    "network_lifetime": 1,
    "first_node_death_round": 1,
}


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    DATASET_DIR.mkdir(exist_ok=True)

    config = SimulationConfig(num_nodes=30, rounds=80, seed=42)
    base_network = Network.create_random(config)
    traditional_network = base_network.clone()
    deep_q_network = base_network.clone()
    traditional_snapshots = []
    deep_q_snapshots = []

    traditional_results = run_simulation(
        traditional_network,
        TraditionalRPL(),
        config.seed,
        snapshot_callback=capture_snapshots(traditional_snapshots),
    )
    with DecisionLogger(DATASET_DIR / "rl_rpl_decisions.csv") as decision_logger:
        deep_q_results = run_simulation(
            deep_q_network,
            QLearningRPL(seed=config.seed),
            config.seed,
            decision_logger=decision_logger,
            snapshot_callback=capture_snapshots(deep_q_snapshots),
        )

    comparison = build_comparison_summary(traditional_results, deep_q_results)
    save_round_metrics(deep_q_results, DATASET_DIR / "rl_rpl_round_metrics.csv")
    save_round_metrics(traditional_results, DATASET_DIR / "rpl_round_metrics.csv")
    save_comparison_metrics(traditional_results, deep_q_results)
    save_snapshots(
        {
            "traditional": {
                "name": "Traditional RPL",
                "snapshots": traditional_snapshots,
                "metrics": metrics_payload(traditional_results),
            },
            "deep_q": {
                "name": "Deep Q RL-RPL",
                "snapshots": deep_q_snapshots,
                "metrics": metrics_payload(deep_q_results),
            },
        },
        config,
        comparison,
    )
    save_final_topology(
        traditional_snapshots[-1],
        config,
        OUTPUT_DIR / "rpl_final_topology.png",
        "Traditional RPL",
    )
    save_final_topology(
        deep_q_snapshots[-1],
        config,
        OUTPUT_DIR / "rl_rpl_final_topology.png",
        "Deep Q RL-RPL",
    )
    save_animation(deep_q_snapshots, config, OUTPUT_DIR / "rl_rpl_visual_simulation.gif", "Deep Q RL-RPL")
    print_visual_summary(traditional_results, deep_q_results, comparison, traditional_snapshots, deep_q_snapshots)


def capture_snapshots(target: list[dict]):
    def capture_snapshot(round_number, current_network, metric):
        target.append(snapshot_payload(round_number, current_network, metric))

    return capture_snapshot


def snapshot_payload(round_number, current_network, metric) -> dict:
    return {
        "round": round_number,
        "nodes": {
            node_id: {
                "x": node.x,
                "y": node.y,
                "energy": None if node.is_sink else node.energy,
                "alive": node.alive,
                "is_sink": node.is_sink,
                "parent": node.parent,
                "queue": node.queue_load,
            }
            for node_id, node in current_network.nodes.items()
        },
        "alive_nodes": metric.alive_nodes,
        "pdr": metric.packet_delivery_ratio,
        "average_delay": metric.average_delay,
        "average_energy": metric.average_remaining_energy,
        "total_energy_consumed": metric.total_energy_consumed,
        "generated_packets": metric.generated_packets,
        "delivered_packets": metric.delivered_packets,
    }


def save_round_metrics(results, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "round",
                "total_energy_consumed",
                "average_remaining_energy",
                "alive_nodes",
                "generated_packets",
                "delivered_packets",
                "packet_delivery_ratio",
                "average_delay",
                "network_lifetime",
            ],
        )
        writer.writeheader()
        for metric in results.rounds:
            writer.writerow(
                {
                    "round": metric.round_number,
                    "total_energy_consumed": f"{metric.total_energy_consumed:.5f}",
                    "average_remaining_energy": f"{metric.average_remaining_energy:.5f}",
                    "alive_nodes": metric.alive_nodes,
                    "generated_packets": metric.generated_packets,
                    "delivered_packets": metric.delivered_packets,
                    "packet_delivery_ratio": f"{metric.packet_delivery_ratio:.5f}",
                    "average_delay": f"{metric.average_delay:.5f}",
                    "network_lifetime": results.network_lifetime,
                }
            )


def save_comparison_metrics(traditional_results, deep_q_results) -> None:
    path = DATASET_DIR / "comparison_round_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "protocol",
                "round",
                "total_energy_consumed",
                "average_remaining_energy",
                "alive_nodes",
                "generated_packets",
                "delivered_packets",
                "packet_delivery_ratio",
                "average_delay",
                "network_lifetime",
            ],
        )
        writer.writeheader()
        for protocol_name, results in [
            ("Traditional RPL", traditional_results),
            ("Deep Q RL-RPL", deep_q_results),
        ]:
            for metric in results.rounds:
                writer.writerow(
                    {
                        "protocol": protocol_name,
                        "round": metric.round_number,
                        "total_energy_consumed": f"{metric.total_energy_consumed:.5f}",
                        "average_remaining_energy": f"{metric.average_remaining_energy:.5f}",
                        "alive_nodes": metric.alive_nodes,
                        "generated_packets": metric.generated_packets,
                        "delivered_packets": metric.delivered_packets,
                        "packet_delivery_ratio": f"{metric.packet_delivery_ratio:.5f}",
                        "average_delay": f"{metric.average_delay:.5f}",
                        "network_lifetime": results.network_lifetime,
                    }
                )


def metrics_payload(results) -> list[dict]:
    return [
        {
            "round": metric.round_number,
            "total_energy_consumed": metric.total_energy_consumed,
            "average_remaining_energy": metric.average_remaining_energy,
            "alive_nodes": metric.alive_nodes,
            "generated_packets": metric.generated_packets,
            "delivered_packets": metric.delivered_packets,
            "packet_delivery_ratio": metric.packet_delivery_ratio,
            "average_delay": metric.average_delay,
        }
        for metric in results.rounds
    ]


def build_comparison_summary(traditional_results, deep_q_results) -> dict:
    traditional = result_summary(traditional_results)
    deep_q = result_summary(deep_q_results)
    winners = {
        "energy_consumption": winner_lower(
            "Traditional RPL",
            traditional["total_energy_consumed"],
            "Deep Q RL-RPL",
            deep_q["total_energy_consumed"],
        ),
        "energy_per_delivered_packet": winner_lower(
            "Traditional RPL",
            traditional["energy_per_delivered_packet"],
            "Deep Q RL-RPL",
            deep_q["energy_per_delivered_packet"],
        ),
        "average_remaining_energy": winner_higher(
            "Traditional RPL",
            traditional["average_remaining_energy"],
            "Deep Q RL-RPL",
            deep_q["average_remaining_energy"],
        ),
        "alive_nodes": winner_higher(
            "Traditional RPL",
            traditional["alive_nodes"],
            "Deep Q RL-RPL",
            deep_q["alive_nodes"],
        ),
        "delivered_packets": winner_higher(
            "Traditional RPL",
            traditional["delivered_packets"],
            "Deep Q RL-RPL",
            deep_q["delivered_packets"],
        ),
        "packet_delivery_ratio": winner_higher(
            "Traditional RPL",
            traditional["packet_delivery_ratio"],
            "Deep Q RL-RPL",
            deep_q["packet_delivery_ratio"],
        ),
        "average_delay": winner_lower(
            "Traditional RPL",
            traditional["average_delay"],
            "Deep Q RL-RPL",
            deep_q["average_delay"],
        ),
        "network_lifetime": winner_higher(
            "Traditional RPL",
            traditional["network_lifetime"],
            "Deep Q RL-RPL",
            deep_q["network_lifetime"],
        ),
        "first_node_death_round": winner_higher(
            "Traditional RPL",
            traditional["first_node_death_round"] or 0,
            "Deep Q RL-RPL",
            deep_q["first_node_death_round"] or 0,
        ),
    }
    energy_saved = traditional["total_energy_consumed"] - deep_q["total_energy_consumed"]
    energy_saved_percent = (
        energy_saved / traditional["total_energy_consumed"] * 100
        if traditional["total_energy_consumed"]
        else 0.0
    )
    return {
        "traditional": traditional,
        "deep_q": deep_q,
        "energy_saved": energy_saved,
        "energy_saved_percent": energy_saved_percent,
        "average_energy_gain": deep_q["average_remaining_energy"] - traditional["average_remaining_energy"],
        "energy_per_delivered_saved": (
            traditional["energy_per_delivered_packet"] - deep_q["energy_per_delivered_packet"]
        ),
        "energy_per_delivered_saved_percent": (
            (traditional["energy_per_delivered_packet"] - deep_q["energy_per_delivered_packet"])
            / traditional["energy_per_delivered_packet"]
            * 100
            if traditional["energy_per_delivered_packet"]
            else 0.0
        ),
        "delivered_packet_gain": deep_q["delivered_packets"] - traditional["delivered_packets"],
        "metric_weights": METRIC_WEIGHTS,
        "winner_by_metric": winners,
        "weighted_winner": weighted_winner_summary(winners),
    }


def result_summary(results) -> dict:
    final = results.rounds[-1]
    delivered_rounds = [metric.average_delay for metric in results.rounds if metric.delivered_packets > 0]
    generated_packets = sum(metric.generated_packets for metric in results.rounds)
    delivered_packets = sum(metric.delivered_packets for metric in results.rounds)
    return {
        "name": results.name,
        "total_energy_consumed": final.total_energy_consumed,
        "average_remaining_energy": final.average_remaining_energy,
        "alive_nodes": final.alive_nodes,
        "generated_packets": generated_packets,
        "delivered_packets": delivered_packets,
        "packet_delivery_ratio": results.final_pdr,
        "energy_per_delivered_packet": (
            final.total_energy_consumed / delivered_packets if delivered_packets else 0.0
        ),
        "average_delay": sum(delivered_rounds) / len(delivered_rounds) if delivered_rounds else 0.0,
        "network_lifetime": results.network_lifetime,
        "first_node_death_round": results.first_node_death_round,
    }


def winner_higher(left_name: str, left_value: float, right_name: str, right_value: float) -> str:
    if left_value == right_value:
        return "Tie"
    return left_name if left_value > right_value else right_name


def winner_lower(left_name: str, left_value: float, right_name: str, right_value: float) -> str:
    if left_value == right_value:
        return "Tie"
    return left_name if left_value < right_value else right_name


def weighted_winner_summary(winners: dict[str, str]) -> dict:
    traditional_score = 0
    deep_q_score = 0
    tie_score = 0
    traditional_wins = 0
    deep_q_wins = 0
    tie_wins = 0
    for metric, winner in winners.items():
        weight = METRIC_WEIGHTS.get(metric, 1)
        if winner == "Deep Q RL-RPL":
            deep_q_score += weight
            deep_q_wins += 1
        elif winner == "Traditional RPL":
            traditional_score += weight
            traditional_wins += 1
        else:
            tie_score += weight
            tie_wins += 1

    if deep_q_score == traditional_score:
        overall = "Deep Q RL-RPL" if winners.get("packet_delivery_ratio") == "Deep Q RL-RPL" else "Traditional RPL"
    else:
        overall = "Deep Q RL-RPL" if deep_q_score > traditional_score else "Traditional RPL"

    return {
        "overall": overall,
        "traditional_score": traditional_score,
        "deep_q_score": deep_q_score,
        "tie_score": tie_score,
        "traditional_wins": traditional_wins,
        "deep_q_wins": deep_q_wins,
        "tie_wins": tie_wins,
        "method": "Weighted score: PDR, energy per delivered packet, delivered packets, and delay have higher priority.",
    }


def save_snapshots(protocols: dict, config: SimulationConfig, comparison: dict) -> None:
    path = DATASET_DIR / "network_snapshots.json"
    payload = {
        "config": {
            "num_nodes": config.num_nodes,
            "area_width": config.area_width,
            "area_height": config.area_height,
            "initial_energy": config.initial_energy,
            "transmission_range": config.transmission_range,
            "queue_decay_factor": config.queue_decay_factor,
            "rounds": len(protocols["deep_q"]["snapshots"]),
        },
        "snapshots": protocols["deep_q"]["snapshots"],
        "protocols": protocols,
        "comparison": comparison,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)


def save_animation(snapshots: list[dict], config: SimulationConfig, path: Path, protocol_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    def update(frame_index):
        ax.clear()
        draw_snapshot(ax, snapshots[frame_index], config, protocol_name)

    animation = FuncAnimation(fig, update, frames=len(snapshots), interval=250, repeat=True)
    try:
        animation.save(path, writer=PillowWriter(fps=4))
    except Exception as exc:
        print(f"Could not save GIF animation: {exc}")
        for index, snapshot in enumerate(snapshots[::10], start=1):
            frame_path = OUTPUT_DIR / f"rl_rpl_frame_{index:02d}.png"
            save_final_topology(snapshot, config, frame_path, protocol_name)
    finally:
        plt.close(fig)


def save_final_topology(snapshot: dict, config: SimulationConfig, path: Path, protocol_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    draw_snapshot(ax, snapshot, config, protocol_name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_snapshot(ax, snapshot: dict, config: SimulationConfig, protocol_name: str) -> None:
    nodes = snapshot["nodes"]
    ax.set_xlim(-5, config.area_width + 5)
    ax.set_ylim(-5, config.area_height + 5)
    ax.set_title(
        f"{protocol_name} Tree | Round {snapshot['round']} | "
        f"Alive {snapshot['alive_nodes']}/30 | PDR {snapshot['pdr']:.2f}"
    )
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.grid(True, alpha=0.2)

    for node_id, node in nodes.items():
        parent_id = node["parent"]
        if node_id == 0 or parent_id is None or parent_id not in nodes:
            continue
        parent = nodes[parent_id]
        if not node["alive"]:
            continue
        ax.plot(
            [node["x"], parent["x"]],
            [node["y"], parent["y"]],
            color="#5b8e7d",
            linewidth=1.0,
            alpha=0.45,
            zorder=1,
        )

    sensor_x = []
    sensor_y = []
    sensor_energy = []
    dead_x = []
    dead_y = []
    for node_id, node in nodes.items():
        if node["is_sink"]:
            ax.scatter(node["x"], node["y"], s=260, c="#d62828", marker="*", edgecolors="black", zorder=4)
            ax.text(node["x"] + 1.5, node["y"] + 1.5, "Sink", fontsize=9, weight="bold")
        elif node["alive"]:
            sensor_x.append(node["x"])
            sensor_y.append(node["y"])
            sensor_energy.append(node["energy"])
        else:
            dead_x.append(node["x"])
            dead_y.append(node["y"])

    scatter = ax.scatter(
        sensor_x,
        sensor_y,
        c=sensor_energy,
        cmap="viridis",
        vmin=0,
        vmax=config.initial_energy,
        s=95,
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )
    if dead_x:
        ax.scatter(dead_x, dead_y, s=90, c="#343a40", marker="x", linewidths=2.0, zorder=3)

    for node_id, node in nodes.items():
        if node["is_sink"]:
            continue
        ax.text(node["x"] + 1.0, node["y"] + 1.0, str(node_id), fontsize=7)

    colorbar = getattr(ax.figure, "_energy_colorbar", None)
    if colorbar is None:
        ax.figure._energy_colorbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        ax.figure._energy_colorbar.set_label("Remaining energy")

    ax.text(
        0.01,
        0.01,
        "Line = selected parent route | color = remaining energy | X = dead node",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )


def print_visual_summary(traditional_results, deep_q_results, comparison: dict, traditional_snapshots, deep_q_snapshots) -> None:
    traditional_final = traditional_results.rounds[-1]
    deep_q_final = deep_q_results.rounds[-1]
    print("\nTwo-tree RPL vs Deep Q simulation created")
    print("-" * 70)
    print(f"Routers/sensor nodes: 30")
    print(f"Rounds visualized: {len(deep_q_snapshots)}")
    print(
        f"Traditional RPL final: alive={traditional_final.alive_nodes}, "
        f"PDR={traditional_results.final_pdr:.3f}, "
        f"energy consumed={traditional_final.total_energy_consumed:.3f}"
    )
    print(
        f"Deep Q RL-RPL final: alive={deep_q_final.alive_nodes}, "
        f"PDR={deep_q_results.final_pdr:.3f}, "
        f"energy consumed={deep_q_final.total_energy_consumed:.3f}"
    )
    print(
        f"Energy saved by Deep Q: {comparison['energy_saved']:.3f} "
        f"({comparison['energy_saved_percent']:.2f}%)"
    )
    print(
        f"Energy per delivered packet saved by Deep Q: "
        f"{comparison['energy_per_delivered_saved']:.5f} "
        f"({comparison['energy_per_delivered_saved_percent']:.2f}%)"
    )
    print(
        f"Overall weighted winner: {comparison['weighted_winner']['overall']} "
        f"(Deep Q score={comparison['weighted_winner']['deep_q_score']}, "
        f"RPL score={comparison['weighted_winner']['traditional_score']})"
    )
    print(f"Decision dataset: {DATASET_DIR / 'rl_rpl_decisions.csv'}")
    print(f"Round metrics dataset: {DATASET_DIR / 'rl_rpl_round_metrics.csv'}")
    print(f"Traditional metrics dataset: {DATASET_DIR / 'rpl_round_metrics.csv'}")
    print(f"Comparison metrics dataset: {DATASET_DIR / 'comparison_round_metrics.csv'}")
    print(f"Network snapshots: {DATASET_DIR / 'network_snapshots.json'}")
    print(f"GIF animation: {OUTPUT_DIR / 'rl_rpl_visual_simulation.gif'}")
    print(f"Traditional final topology image: {OUTPUT_DIR / 'rpl_final_topology.png'}")
    print(f"Deep Q final topology image: {OUTPUT_DIR / 'rl_rpl_final_topology.png'}")


if __name__ == "__main__":
    main()
