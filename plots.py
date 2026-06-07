from pathlib import Path

import matplotlib.pyplot as plt

from metrics import SimulationResults


def plot_single_result(result: SimulationResults, output_dir: str = "results") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    rounds = [metric.round_number for metric in result.rounds]
    _save_plot(
        rounds,
        [metric.total_energy_consumed for metric in result.rounds],
        "Energy Consumption vs Rounds",
        "Round",
        "Total energy consumed (J)",
        output_path / "traditional_energy_consumption.png",
        result.name,
    )
    _save_plot(
        rounds,
        [metric.alive_nodes for metric in result.rounds],
        "Alive Nodes vs Rounds",
        "Round",
        "Alive sensor nodes",
        output_path / "traditional_alive_nodes.png",
        result.name,
    )
    _save_plot(
        rounds,
        [metric.packet_delivery_ratio for metric in result.rounds],
        "Packet Delivery Ratio vs Rounds",
        "Round",
        "Packet delivery ratio",
        output_path / "traditional_packet_delivery_ratio.png",
        result.name,
    )
    _save_plot(
        rounds,
        [metric.average_delay for metric in result.rounds],
        "Average Delay vs Rounds",
        "Round",
        "Average delay",
        output_path / "traditional_average_delay.png",
        result.name,
    )


def plot_comparison(results: list[SimulationResults], output_dir: str = "results") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    series = {
        "comparison_energy_consumption.png": (
            "Energy Consumption Comparison",
            "Total energy consumed (J)",
            lambda metric: metric.total_energy_consumed,
        ),
        "comparison_alive_nodes.png": (
            "Alive Nodes Comparison",
            "Alive sensor nodes",
            lambda metric: metric.alive_nodes,
        ),
        "comparison_average_remaining_energy.png": (
            "Average Remaining Energy Comparison",
            "Average remaining energy (J)",
            lambda metric: metric.average_remaining_energy,
        ),
        "comparison_packet_delivery_ratio.png": (
            "Packet Delivery Ratio Comparison",
            "Packet delivery ratio",
            lambda metric: metric.packet_delivery_ratio,
        ),
        "comparison_average_delay.png": (
            "Average Delay Comparison",
            "Average delay",
            lambda metric: metric.average_delay,
        ),
    }

    for file_name, (title, ylabel, accessor) in series.items():
        plt.figure(figsize=(8, 5))
        for result in results:
            rounds = [metric.round_number for metric in result.rounds]
            values = [accessor(metric) for metric in result.rounds]
            plt.plot(rounds, values, label=result.name)
        plt.title(title)
        plt.xlabel("Round")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / file_name, dpi=150)
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
