from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASS_LABELS = ["Fully\npeeled", "Partially\npeeled", "Spoiled"]
SUMMARY_LABELS = [
    "Fully peeled garlic",
    "Partially peeled garlic",
    "Spoiled garlic",
]

DATASETS = {
    "Dataset-1": np.array([690, 438, 1006]),
    "Dataset-2": np.array([1500, 438, 1006]),
}

# Okabe-Ito palette: readable in print and safer for color-vision deficiencies.
COLORS = {
    "Dataset-1": "#0072B2",
    "Dataset-2": "#D55E00",
}

HATCHES = {
    "Dataset-1": "",
    "Dataset-2": "///",
}


def calculate_percentages(counts_by_dataset):
    return {
        name: counts / counts.sum() * 100
        for name, counts in counts_by_dataset.items()
    }


def configure_publication_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "axes.titleweight": "normal",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_panel_label(ax, label):
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def add_bar_labels(ax, bars, formatter, offset):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            formatter(height),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="black",
        )


def style_axes(ax, ylim, yticks, ylabel):
    ax.set_ylim(0, ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(CLASS_LABELS)))
    ax.set_xticklabels(CLASS_LABELS)
    ax.yaxis.grid(True, color="#d9d9d9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", pad=2)
    ax.margins(x=0.08)


def plot_grouped_bars(ax, values_by_dataset, ylabel, ylim, yticks, formatter):
    x = np.arange(len(CLASS_LABELS))
    width = 0.34
    offsets = [-width / 2, width / 2]
    label_offset = ylim * 0.018

    bars_for_legend = []
    for (dataset_name, values), offset in zip(values_by_dataset.items(), offsets):
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=dataset_name,
            color=COLORS[dataset_name],
            edgecolor="black",
            linewidth=0.6,
            hatch=HATCHES[dataset_name],
        )
        bars_for_legend.append(bars[0])
        add_bar_labels(ax, bars, formatter, label_offset)

    style_axes(ax, ylim, yticks, ylabel)
    return bars_for_legend


def save_figure(fig, output_stem):
    output_paths = [
        output_stem.with_suffix(".png"),
        output_stem.with_suffix(".pdf"),
        output_stem.with_suffix(".svg"),
    ]

    for output_path in output_paths:
        save_kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if output_path.suffix == ".png":
            save_kwargs["dpi"] = 600
        fig.savefig(output_path, **save_kwargs)

    return output_paths


def print_summary(percentages_by_dataset, output_paths):
    print("\n" + "=" * 58)
    print("DATASET DISTRIBUTION SUMMARY")
    print("=" * 58)

    for dataset_name, counts in DATASETS.items():
        total = counts.sum()
        print(f"\n{dataset_name}:")
        print(f"  Total samples: {total}")
        for class_name, count, percentage in zip(
            SUMMARY_LABELS,
            counts,
            percentages_by_dataset[dataset_name],
        ):
            print(f"  - {class_name}: {count} samples ({percentage:.2f}%)")

    print("\nSaved figures:")
    for output_path in output_paths:
        print(f"  - {output_path}")
    print("=" * 58)


def main():
    percentages_by_dataset = calculate_percentages(DATASETS)
    configure_publication_style()

    # 7.2 inches is a common double-column figure width.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=False)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.82, bottom=0.18, wspace=0.32)

    handles = plot_grouped_bars(
        axes[0],
        DATASETS,
        "Number of samples",
        ylim=1650,
        yticks=np.arange(0, 1750, 250),
        formatter=lambda value: f"{int(round(value)):,}",
    )
    axes[0].set_title("Sample count")
    add_panel_label(axes[0], "(a)")

    plot_grouped_bars(
        axes[1],
        percentages_by_dataset,
        "Class proportion (%)",
        ylim=56,
        yticks=np.arange(0, 61, 10),
        formatter=lambda value: f"{value:.1f}",
    )
    axes[1].set_title("Class proportion")
    add_panel_label(axes[1], "(b)")

    fig.legend(
        handles,
        list(DATASETS.keys()),
        loc="upper center",
        bbox_to_anchor=(0.54, 0.995),
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.8,
    )

    output_stem = Path(__file__).with_name("dataset_distribution")
    output_paths = save_figure(fig, output_stem)
    print_summary(percentages_by_dataset, output_paths)

    if plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
