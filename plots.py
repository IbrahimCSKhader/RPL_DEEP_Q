from pathlib import Path

import matplotlib.pyplot as plt

from metrics import SimulationResults


def plot_single_result(result: SimulationResults, output_dir: str = "results") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    times = [metric.time for metric in result.rounds]
    prefix = "traditional" if result.name == "Traditional RPL" else "rl_rpl"

    _save_plot(
        times,
        [metric.total_energy_consumed for metric in result.rounds],
        f"{result.name} Energy Consumed Over Time",
        "Simulation time (s)",
        "Total energy consumed (J)",
        output_path / f"{prefix}_energy_consumed_over_time.png",
        result.name,
    )
    _save_plot(
        times,
        [metric.packet_delivery_ratio for metric in result.rounds],
        f"{result.name} PDR Over Time",
        "Simulation time (s)",
        "Packet delivery ratio",
        output_path / f"{prefix}_pdr_over_time.png",
        result.name,
    )
    _save_plot(
        times,
        [metric.average_delay for metric in result.rounds],
        f"{result.name} Average Delay Over Time",
        "Simulation time (s)",
        "Average delay (s)",
        output_path / f"{prefix}_average_delay_over_time.png",
        result.name,
    )


def plot_comparison(results: list[SimulationResults], output_dir: str = "results") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    series = {
        "comparison_energy_consumed_over_time.png": (
            "Energy Consumed Over Time",
            "Total energy consumed (J)",
            lambda metric: metric.total_energy_consumed,
        ),
        "comparison_average_remaining_energy_over_time.png": (
            "Average Remaining Energy Over Time",
            "Average remaining energy (J)",
            lambda metric: metric.average_remaining_energy,
        ),
        "comparison_alive_nodes_over_time.png": (
            "Alive Sensors Over Time",
            "Alive sensor nodes",
            lambda metric: metric.alive_nodes,
        ),
        "comparison_active_sensors_over_time.png": (
            "Active Sensors Over Time",
            "Sensors with a deliverable path to ROOT",
            lambda metric: metric.active_sensors,
        ),
        "comparison_pdr_over_time.png": (
            "Packet Delivery Ratio Over Time",
            "Packet delivery ratio",
            lambda metric: metric.packet_delivery_ratio,
        ),
        "comparison_average_delay_over_time.png": (
            "Average Delay Over Time",
            "Average delay (s)",
            lambda metric: metric.average_delay,
        ),
        "comparison_cumulative_delivered_packets.png": (
            "Cumulative Delivered Packets",
            "Delivered packets",
            lambda metric: metric.cumulative_delivered_packets,
        ),
        "comparison_cumulative_lost_packets.png": (
            "Cumulative Lost Packets",
            "Lost/failed packets",
            lambda metric: metric.cumulative_lost_packets,
        ),
    }

    for file_name, (title, ylabel, accessor) in series.items():
        plt.figure(figsize=(8, 5))
        for result in results:
            times = [metric.time for metric in result.rounds]
            values = [accessor(metric) for metric in result.rounds]
            plt.plot(times, values, label=result.name)
        plt.title(title)
        plt.xlabel("Simulation time (s)")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / file_name, dpi=150)
        plt.close()

    _save_lifetime_comparison(results, output_path / "comparison_network_lifetime.png")


def _save_lifetime_comparison(results: list[SimulationResults], path: Path) -> None:
    labels = [result.name for result in results]
    lifetimes = [result.network_lifetime for result in results]
    colors = ["#536878", "#147d75"]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, lifetimes, color=colors[: len(labels)])
    plt.title("Network Lifetime Comparison")
    plt.ylabel("Time until no sensor can deliver to ROOT (s)")
    plt.grid(True, axis="y", alpha=0.3)
    for bar, lifetime in zip(bars, lifetimes):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{lifetime:.1f}s",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_plot(x, y, title, xlabel, ylabel, path: Path, label: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
