import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

marker_styles = {
    'gross': 'o',
    'bacon': 's',
    'hh': '^',
    'surface': 'D',
    'other': 'X'
}

code_rename_map = {
    'bacon': 'Bacon-Shor',
    'hh': 'Heavy-hex',
    'gross': 'Gross'
}

error_type_map = {
    'Constant': 'constant',
    'SI1000': 'modsi1000'
}


code_palette = sns.color_palette("pastel", n_colors=6)
code_hatches = ["", "o", "//", "++", "xx", "**"]

def generate_size_plot(df_path):
    df = pd.read_csv(df_path)
    error_types = ['Constant', 'SI1000']
    error_probs = [0.004]
    df_filtered = df[~df['code'].str.contains('heavyhex', case=False, na=False)]
    df_filtered = df_filtered[~df_filtered['backend'].str.contains('heavyhex', case=False, na=False)]
    backends = df_filtered['backend'].unique()

    os.makedirs("../data", exist_ok=True)

    for backend in backends:
        n_rows = 1
        n_cols = len(error_probs) * len(error_types)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5), sharey=True)
        if n_cols == 1:
            axes = [axes]
        all_handles_labels = []

        for idx, (i, p) in enumerate(enumerate(error_probs)):
            for j, et in enumerate(error_types):
                ax = axes[idx * len(error_types) + j]
                original_et = error_type_map.get(et, et.lower())
                subset = df_filtered[
                    (df_filtered['backend'] == backend) &
                    (df_filtered['error_type'] == original_et) &
                    (df_filtered['error_probability'] == p)
                ]

                for code, group in subset.groupby('code'):
                    code_key = code.lower()
                    code_display = code_rename_map.get(code_key, code.capitalize())
                    marker = marker_styles.get(code_key, marker_styles['other'])
                    group_sorted = group.sort_values('backend_size')

                    line = ax.plot(
                        group_sorted['backend_size'],
                        group_sorted['logical_error_rate'],
                        label=code_display,
                        marker=marker
                    )

                    if code_key == 'gross':
                        line_color = line[0].get_color()
                        gross_div12 = group_sorted['logical_error_rate'] / 12
                        ax.plot(
                            group_sorted['backend_size'],
                            gross_div12,
                            linestyle='--',
                            color=line_color,
                            label='Gross / 12'
                        )

                ax.set_xlabel('Backend Size')
                xticks = sorted(subset['backend_size'].unique())
                ax.set_xticks(xticks)
                ax.set_title(f'{et}', fontsize=11)

                if idx == 0 and j == 0:
                    ax.set_ylabel('Logical Error Rate')
                    ax.text(-0.1, 1.05, 'Lower is better ↓', transform=ax.transAxes,
                            fontsize=10, fontweight='bold', va='top', ha='left')

                ax.grid(True)
                ax.set_ylim(0, 1.05)
                handles, labels = ax.get_legend_handles_labels()
                all_handles_labels.extend(zip(handles, labels))

        unique_labels = {}
        for h, l in all_handles_labels:
            if l not in unique_labels:
                unique_labels[l] = h

        fig.legend(
            handles=list(unique_labels.values()),
            labels=list(unique_labels.keys()),
            title='Code',
            fontsize='small',
            title_fontsize='small',
            loc='upper right',
            bbox_to_anchor=(0.95, 0.88)
        )

        plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        plt.savefig(f"data/size_{backend}.pdf")
        plt.close(fig)

def generate_connectivity_plot(df_path):
    df = pd.read_csv(df_path)
    df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))

    backend_order = ['custom_grid', 'custom_cube', 'custom_full']
    df["backend"] = pd.Categorical(df["backend"], categories=backend_order, ordered=True)
    df["std"] = np.sqrt(df["logical_error_rate"] * (1 - df["logical_error_rate"]) / df["num_samples"])

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    codes = sorted(df["code"].unique())
    backends = backend_order
    x = np.arange(len(backends))
    bar_width = 0.15

    for i, code in enumerate(codes):
        subset = df[df["code"] == code]
        means = []
        stds = []

        for backend in backends:
            row = subset[subset["backend"] == backend]
            if not row.empty:
                means.append(row["logical_error_rate"].values[0])
                stds.append(row["std"].values[0])
            else:
                means.append(0)
                stds.append(0)

        plt.bar(
            x + i * bar_width,
            means,
            yerr=stds,
            width=bar_width,
            color=code_palette[i % len(code_palette)],
            hatch=code_hatches[i % len(code_hatches)],
            edgecolor="black",
            label=code
        )

    plt.xticks(x + bar_width * (len(codes) - 1) / 2, [b.replace("custom_", "").capitalize() for b in backends])
    plt.xlabel("Backend Connectivity")
    plt.ylabel("Logical Error Rate (Log Scale)")
    plt.title("Logical Error Rate by Backend")
    plt.legend(title="Code")
    plt.yscale("log")
    plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
    plt.text(-0.05, 1.05, 'Lower is better ↓', transform=plt.gca().transAxes,
             fontsize=10, fontweight='bold', va='top', ha='left')
    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/connectivity.pdf", format="pdf")
    plt.close()


if __name__ == '__main__':
    size = "experiment_results/Size_full/results.csv"
    connectivity = "experiment_results/Connectivity_small/results.csv"
    generate_size_plot(size)
    generate_connectivity_plot(connectivity)