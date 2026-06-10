import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from data_logger import DecisionLogger, PacketLogger, export_comparison_table, export_summary_metrics
from main import build_comparison_payload, generate_comparison_table, run_simulation
from network import Network, SimulationConfig
from plots import plot_comparison, plot_single_result
from rl_agent import QLearningRPL
from routing import TraditionalRPL


OUTPUT_DIR = Path("results")
DATASET_DIR = Path("dataset")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    DATASET_DIR.mkdir(exist_ok=True)

    config = SimulationConfig(num_nodes=30, simulation_duration=420.0, time_step=1.0, sensing_interval=5.0, seed=42)
    base_network = Network.create_random(config)
    traditional_network = base_network.clone()
    rl_network = base_network.clone()
    traditional_snapshots = []
    rl_snapshots = []

    with PacketLogger(DATASET_DIR / "packet_log.csv") as packet_logger, DecisionLogger(
        DATASET_DIR / "routing_decisions.csv"
    ) as decision_logger:
        traditional_results = run_simulation(
            traditional_network,
            TraditionalRPL(),
            config.seed,
            packet_logger=packet_logger,
            decision_logger=decision_logger,
            snapshot_callback=capture_snapshots(traditional_snapshots),
        )
        rl_results = run_simulation(
            rl_network,
            QLearningRPL(seed=config.seed),
            config.seed,
            packet_logger=packet_logger,
            decision_logger=decision_logger,
            snapshot_callback=capture_snapshots(rl_snapshots),
        )

    comparison = build_comparison_payload(traditional_results, rl_results)
    comparison_rows = generate_comparison_table(traditional_results, rl_results)
    export_summary_metrics([traditional_results, rl_results], DATASET_DIR / "summary_metrics.csv")
    export_comparison_table(comparison_rows, DATASET_DIR / "comparison_table.csv")
    save_time_metrics(traditional_results, DATASET_DIR / "rpl_time_metrics.csv")
    save_time_metrics(rl_results, DATASET_DIR / "rl_rpl_time_metrics.csv")
    save_combined_time_metrics([traditional_results, rl_results], DATASET_DIR / "comparison_time_metrics.csv")
    save_snapshots(
        {
            "traditional": {
                "name": "Traditional RPL",
                "snapshots": traditional_snapshots,
                "metrics": metrics_payload(traditional_results),
            },
            "rl_rpl": {
                "name": "Q-learning RL-RPL",
                "snapshots": rl_snapshots,
                "metrics": metrics_payload(rl_results),
            },
        },
        config,
        comparison,
    )

    plot_single_result(traditional_results)
    plot_single_result(rl_results)
    plot_comparison([traditional_results, rl_results])
    save_final_topology(
        traditional_snapshots[-1],
        config,
        OUTPUT_DIR / "rpl_final_topology.png",
        "Traditional RPL",
    )
    save_final_topology(
        rl_snapshots[-1],
        config,
        OUTPUT_DIR / "rl_rpl_final_topology.png",
        "Q-learning RL-RPL",
    )
    save_animation(rl_snapshots, config, OUTPUT_DIR / "rl_rpl_visual_simulation.gif", "Q-learning RL-RPL")
    print_visual_summary(traditional_results, rl_results, comparison)


def capture_snapshots(target: list[dict]):
    def capture_snapshot(current_time, current_network, metric, recent_packets):
        target.append(snapshot_payload(current_time, current_network, metric, recent_packets))

    return capture_snapshot


def snapshot_payload(current_time, current_network, metric, recent_packets: list[dict]) -> dict:
    return {
        "time": current_time,
        "nodes": {
            str(node_id): {
                "x": node.x,
                "y": node.y,
                "energy": None if node.is_root else node.energy,
                "alive": node.alive,
                "is_root": node.is_root,
                "parent": node.parent,
                "queue": node.queue_load,
                "unable_to_send_time": node.unable_to_send_time,
            }
            for node_id, node in current_network.nodes.items()
        },
        "alive_nodes": metric.alive_nodes,
        "active_sensors": metric.active_sensors,
        "dead_nodes": metric.dead_nodes,
        "pdr": metric.packet_delivery_ratio,
        "average_delay": metric.average_delay,
        "average_energy": metric.average_remaining_energy,
        "total_energy_consumed": metric.total_energy_consumed,
        "generated_packets": metric.generated_packets,
        "delivered_packets": metric.delivered_packets,
        "lost_packets": metric.lost_packets,
        "current_packet": recent_packets[-1] if recent_packets else None,
        "recent_packets": recent_packets[-8:],
    }


def save_time_metrics(results, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=time_metric_fields())
        writer.writeheader()
        for row in metrics_payload(results):
            writer.writerow(row)


def save_combined_time_metrics(results_list, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["protocol", *time_metric_fields()])
        writer.writeheader()
        for results in results_list:
            for row in metrics_payload(results):
                writer.writerow({"protocol": results.name, **row})


def time_metric_fields() -> list[str]:
    return [
        "time",
        "total_energy_consumed",
        "average_remaining_energy",
        "alive_nodes",
        "active_sensors",
        "dead_nodes",
        "generated_packets",
        "delivered_packets",
        "lost_packets",
        "packet_delivery_ratio",
        "average_delay",
        "communication_overhead",
        "parent_changes",
        "current_packet_id",
        "current_source_sensor_id",
        "current_temperature",
        "selected_route_path",
        "last_packet_status",
    ]


def metrics_payload(results) -> list[dict]:
    return [
        {
            "time": metric.time,
            "total_energy_consumed": metric.total_energy_consumed,
            "average_remaining_energy": metric.average_remaining_energy,
            "alive_nodes": metric.alive_nodes,
            "active_sensors": metric.active_sensors,
            "dead_nodes": metric.dead_nodes,
            "generated_packets": metric.generated_packets,
            "delivered_packets": metric.delivered_packets,
            "lost_packets": metric.lost_packets,
            "packet_delivery_ratio": metric.packet_delivery_ratio,
            "average_delay": metric.average_delay,
            "communication_overhead": metric.communication_overhead,
            "parent_changes": metric.parent_changes,
            "current_packet_id": metric.current_packet_id,
            "current_source_sensor_id": metric.current_source_sensor_id,
            "current_temperature": metric.current_temperature,
            "selected_route_path": metric.selected_route_path,
            "last_packet_status": metric.last_packet_status,
        }
        for metric in results.rounds
    ]


def save_snapshots(protocols: dict, config: SimulationConfig, comparison: dict) -> None:
    path = DATASET_DIR / "network_snapshots.json"
    payload = {
        "config": {
            "num_nodes": config.num_nodes,
            "area_width": config.area_width,
            "area_height": config.area_height,
            "initial_energy": config.initial_energy,
            "transmission_range": config.transmission_range,
            "queue_decay_rate": config.queue_decay_rate,
            "simulation_duration": config.duration_seconds,
            "time_step": config.time_step,
            "sensing_interval": config.sensing_interval,
        },
        "protocols": protocols,
        "comparison": comparison,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)


def save_animation(snapshots: list[dict], config: SimulationConfig, path: Path, protocol_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    animation_snapshots = snapshots[::3] or snapshots

    def update(frame_index):
        ax.clear()
        draw_snapshot(ax, animation_snapshots[frame_index], config, protocol_name)

    animation = FuncAnimation(fig, update, frames=len(animation_snapshots), interval=180, repeat=True)
    try:
        animation.save(path, writer=PillowWriter(fps=6))
    except Exception as exc:
        print(f"Could not save GIF animation: {exc}")
        for index, snapshot in enumerate(snapshots[::20], start=1):
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
        f"{protocol_name} Tree | Time {snapshot['time']:.0f}s | "
        f"Alive {snapshot['alive_nodes']}/{config.num_nodes} | Active {snapshot['active_sensors']} | "
        f"PDR {snapshot['pdr']:.2f}"
    )
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.grid(True, alpha=0.2)

    for node_id, node in nodes.items():
        parent_id = node["parent"]
        if node_id == "0" or parent_id is None or str(parent_id) not in nodes or not node["alive"]:
            continue
        parent = nodes[str(parent_id)]
        ax.plot(
            [node["x"], parent["x"]],
            [node["y"], parent["y"]],
            color="#5b8e7d",
            linewidth=1.0,
            alpha=0.45,
            zorder=1,
        )

    draw_packet_route(ax, snapshot)

    sensor_x = []
    sensor_y = []
    sensor_energy = []
    dead_x = []
    dead_y = []
    for node_id, node in nodes.items():
        if node["is_root"]:
            ax.scatter(node["x"], node["y"], s=280, c="#d62828", marker="*", edgecolors="black", zorder=4)
            ax.text(node["x"] + 1.5, node["y"] + 1.5, "ROOT", fontsize=9, weight="bold")
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
        if node["is_root"]:
            continue
        ax.text(node["x"] + 1.0, node["y"] + 1.0, node_id, fontsize=7)

    colorbar = getattr(ax.figure, "_energy_colorbar", None)
    if colorbar is None:
        ax.figure._energy_colorbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        ax.figure._energy_colorbar.set_label("Remaining energy")

    packet = snapshot.get("current_packet")
    packet_text = "No packet generated at this time"
    if packet:
        packet_text = (
            f"Packet {packet['packet_id']} | sensor {packet['source_sensor_id']} | "
            f"{packet['temperature']:.1f} C | {packet['status']} | path {packet['selected_route_path']}"
        )
    ax.text(
        0.01,
        0.01,
        packet_text,
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )


def draw_packet_route(ax, snapshot: dict) -> None:
    packet = snapshot.get("current_packet")
    if not packet or len(packet.get("route_path", [])) < 2:
        return
    nodes = snapshot["nodes"]
    route = [str(node_id) for node_id in packet["route_path"] if str(node_id) in nodes]
    for left, right in zip(route, route[1:]):
        node = nodes[left]
        parent = nodes[right]
        ax.plot(
            [node["x"], parent["x"]],
            [node["y"], parent["y"]],
            color="#f59f00",
            linewidth=2.6,
            alpha=0.85,
            zorder=2,
        )
    last_node = nodes[route[-1]]
    ax.scatter(last_node["x"], last_node["y"], s=70, c="#f59f00", edgecolors="black", zorder=5)


def print_visual_summary(traditional_results, rl_results, comparison: dict) -> None:
    traditional_final = traditional_results.final_metric
    rl_final = rl_results.final_metric
    print("\nEvent-based Traditional RPL vs Q-learning RL-RPL simulation created")
    print("-" * 78)
    print("Sensors: 30 temperature nodes plus one ROOT")
    print(
        f"Traditional RPL: generated={traditional_final.generated_packets}, "
        f"delivered={traditional_final.delivered_packets}, PDR={traditional_final.packet_delivery_ratio:.3f}, "
        f"energy consumed={traditional_final.total_energy_consumed:.3f} J"
    )
    print(
        f"Q-learning RL-RPL: generated={rl_final.generated_packets}, "
        f"delivered={rl_final.delivered_packets}, PDR={rl_final.packet_delivery_ratio:.3f}, "
        f"energy consumed={rl_final.total_energy_consumed:.3f} J"
    )
    print(f"Overall comparison winner: {comparison['overall_winner']}")
    print(f"Packet log: {DATASET_DIR / 'packet_log.csv'}")
    print(f"Routing decisions: {DATASET_DIR / 'routing_decisions.csv'}")
    print(f"Summary metrics: {DATASET_DIR / 'summary_metrics.csv'}")
    print(f"Comparison table: {DATASET_DIR / 'comparison_table.csv'}")
    print(f"Network snapshots: {DATASET_DIR / 'network_snapshots.json'}")
    print(f"Graphs and GIF: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
