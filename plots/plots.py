from utils import *

def generate_size_plot(df_path):
    df = pd.read_csv(df_path)
    error_types = ["Constant", "SI1000"]
    error_probs = [0.002, 0.004, 0.008]
    backend = "custom_full"

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    all_codes = sorted([c.lower() for c in df['code'].unique()])
    code_color = {c: default_colors[i % len(default_colors)] for i, c in enumerate(all_codes)}

    letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    # one figure per error type
    for et in error_types:
        original_et = error_type_map.get(et, et.lower())

        n_rows, n_cols = 1, len(error_probs)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(WIDE_FIGSIZE * 2, HEIGHT_FIGSIZE),
            sharey=True,
            gridspec_kw={'wspace': 0.1}  # minimal spacing
        )
        if n_cols == 1:
            axes = [axes]

        for col, p in enumerate(error_probs):
            ax = axes[col]

            subset = df[
                (df['backend'] == backend) &
                (df['error_type'] == original_et) &
                (df['error_probability'] == p)
            ]

            for code, group in subset.groupby('code'):
                code_key = code.lower()
                code_display = code_rename_map.get(code_key, code.capitalize())
                marker = marker_styles.get(code_key, marker_styles['other'])
                group_sorted = group.sort_values('backend_size')

                xs = group_sorted['backend_size'].to_numpy()
                ys = group_sorted['logical_error_rate'].to_numpy()

                # base line + plain markers
                line = ax.plot(
                    xs, ys,
                    label=code_display,
                    marker=marker,
                    color=code_color[code_key],
                    markeredgecolor="none",
                )[0]

                # overlay highlighted markers with black outline
                highlight_x = HIGHLIGHT.get((code_key, et), [])
                if highlight_x:
                    sel = np.isin(xs, highlight_x)
                    ax.plot(
                        xs[sel], ys[sel],
                        linestyle="None",
                        marker=marker,
                        markersize=line.get_markersize() * 1.4,
                        markerfacecolor=code_color[code_key],
                        markeredgecolor="black",
                        markeredgewidth=1.5,
                        color=code_color[code_key],
                        zorder=line.get_zorder() + 2,
                        label="_nolegend_",
                    )

            # axes formatting
            ax.set_xlabel("Backend Size", fontsize=FONTSIZE)
            xticks = sorted(subset['backend_size'].unique())
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, fontsize=FONTSIZE - 2)

            if et == "Constant":
                title_et = "Const."
                ax.set_title(f"{chr(100 + col)}) {title_et} ({p})", loc="left", fontsize=12, fontweight="bold")
            else:
                title_et = et
                ax.set_title(f"{chr(97 + col)}) {title_et} ({p})", loc="left", fontsize=12, fontweight="bold")

            if col == 0:
                ax.set_ylabel("Logical Error Rate", fontsize=FONTSIZE)

            # separate "Lower is better ↓" text
            ax.text(
                1.0, 1.16, "Lower is better ↓",
                transform=ax.transAxes,
                fontsize=12, fontweight="bold",
                color="blue", va="top", ha="right"
            )

            ax.grid(True)
            ax.set_ylim(0, 0.65)
            ax.margins(x=0.0)  # remove padding on data ends

            # legend below each subplot
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = {}
            for h, l in zip(handles, labels):
                if l not in unique_labels and l != "_nolegend_":
                    unique_labels[l] = h

        # adjust figure margins to remove left/right padding
        plt.subplots_adjust(
            left=0.05,    # small left margin
            right=0.95,   # small right margin
            bottom=0.3,   # room for legend
        )

        for ax in axes:
            ax.margins(x=0)  # no extra padding for the plotted data


        if et == "Constant":
            fig.legend(
                handles=list(unique_labels.values()),
                labels=list(unique_labels.keys()),
                loc="lower center",
                bbox_to_anchor=(0.5, -0.03),
                ncol=len(unique_labels),
                frameon=False
            )

        os.makedirs("data", exist_ok=True)
        plt.savefig(f"data/size_{backend}_{et}.pdf", format="pdf")
        plt.close(fig)


def generate_connectivity_plot(df_path):
    df = pd.read_csv(df_path)
    df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))
    backend_order = ['custom_grid', 'custom_cube', 'custom_full']
    df["backend"] = pd.Categorical(df["backend"], categories=backend_order, ordered=True)
    df["std"] = np.sqrt(df["logical_error_rate"] * (1 - df["logical_error_rate"]) / df["num_samples"])

    codes = sorted(df["code"].unique())
    backends = backend_order
    n_backends = len(backends)
    n_codes = len(codes)

    # create figure with same size as a single plot in generate_size_plot
    fig, ax = plt.subplots(figsize=(WIDE_FIGSIZE, HEIGHT_FIGSIZE))

    # spacing between backend groups
    x = np.arange(n_backends) * (BAR_WIDTH * n_codes + group_spacing)

    for i, code in enumerate(codes):
        subset = df[df["code"] == code]
        means, stds = [], []

        for backend in backends:
            row = subset[subset["backend"] == backend]
            if not row.empty:
                means.append(row["logical_error_rate"].values[0])
                stds.append(row["std"].values[0])
            else:
                means.append(0)
                stds.append(0)

        ax.bar(
            x + i * BAR_WIDTH,
            means,
            yerr=stds,
            width=BAR_WIDTH,
            color=code_palette[i % len(code_palette)],
            hatch=code_hatches[i % len(code_hatches)],
            edgecolor="black",
            label=code
        )

    # axes formatting
    ax.set_xticks(x + BAR_WIDTH * (n_codes - 1) / 2)
    ax.set_xticklabels([b.replace("custom_", "").capitalize() for b in backends], fontsize=FONTSIZE - 2)
    ax.set_ylabel("Log. Err. Rate (Log)", fontsize=FONTSIZE)
    ax.set_yscale("log")
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    # title in top-left corner
    ax.set_title("a) Artificial Topology", loc="left", fontsize=12, fontweight="bold")

    # separate "Lower is better ↓" text
    ax.text(1.0, 1.14, "Lower is better ↓", transform=ax.transAxes,
            fontsize=12, fontweight="bold", color="blue",
            va="top", ha="right")

    # collect unique handles/labels for legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels[l] = h

    # leave space at bottom for legend
    plt.subplots_adjust(bottom=0.3)
    fig.legend(
        handles=list(unique_labels.values()),
        labels=list(unique_labels.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=len(unique_labels) // 2,
        frameon=False
    )

    os.makedirs("data", exist_ok=True)
    plt.savefig("data/connectivity.pdf", format="pdf")
    plt.close(fig)


def generate_topology_plot(df_path):
    df = pd.read_csv(df_path)
    df = df[df["backend"] != "real_infleqtion"]
    df["backend"] = df["backend"].replace(backend_rename_map)
    df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))
    df["std"] = np.sqrt(df["logical_error_rate"] * (1 - df["logical_error_rate"]) / df["num_samples"])

    codes = sorted(df["code"].unique())
    backends = df["backend"].unique()
    backends = backends[::-1]
    # reorder backends same as original
    n_backends = len(backends)
    n_codes = len(codes)

    # create figure with same size as a single plot in generate_size_plot
    fig, ax = plt.subplots(figsize=(WIDE_FIGSIZE, HEIGHT_FIGSIZE))

    # spacing between backend groups
    x = np.arange(n_backends) * (BAR_WIDTH * n_codes + group_spacing)

    for i, code in enumerate(codes):
        subset = df[df["code"] == code]
        means, stds = [], []

        for backend in backends:
            row = subset[subset["backend"] == backend]
            if not row.empty:
                means.append(row["logical_error_rate"].values[0])
                stds.append(row["std"].values[0])
            else:
                means.append(0)
                stds.append(0)

        ax.bar(
            x + i * BAR_WIDTH,
            means,
            yerr=stds,
            width=BAR_WIDTH,
            color=code_palette[i % len(code_palette)],
            hatch=code_hatches[i % len(code_hatches)],
            edgecolor="black",
            label=code
        )

    # axes formatting
    ax.set_xticks(x + BAR_WIDTH * (n_codes - 1) / 2)
    ax.set_xticklabels(backends, fontsize=FONTSIZE - 2)
    ax.set_ylabel("Log. Err. Rate (Log)", fontsize=FONTSIZE)
    ax.set_yscale("log")
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    # title in top-left corner
    ax.set_title("b) Real Topology", loc="left", fontsize=12, fontweight="bold")

    # separate "Lower is better ↓" text
    ax.text(1.0, 1.14, "Lower is better ↓", transform=ax.transAxes,
            fontsize=12, fontweight="bold", color="blue",
            va="top", ha="right")

    # collect unique handles/labels for legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels[l] = h

    # leave space at bottom for legend
    plt.subplots_adjust(bottom=0.3)
    fig.legend(
        handles=list(unique_labels.values()),
        labels=list(unique_labels.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=len(unique_labels) // 2,
        frameon=False
    )

    os.makedirs("data", exist_ok=True)
    plt.savefig("data/topology.pdf", format="pdf")
    plt.close(fig)

def generate_connectivity_topology_plot(connectivity_csv, topology_csv):
    """
    Generate a single PDF with two subplots:
    - Left: artificial connectivity
    - Right: real topology
    Both plots have log-scaled Y-axis, error bars, grid lines, and a shared legend on the right.
    """
    # --- Load and preprocess data ---
    def preprocess(df_path, backend_order=None, reverse_backends=False):
        df = pd.read_csv(df_path)
        df = df[df["backend"] != "real_infleqtion"]
        df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))
        if backend_order is not None:
            df["backend"] = pd.Categorical(df["backend"], categories=backend_order, ordered=True)
        else:
            df["backend"] = df["backend"].replace(backend_rename_map)
        if reverse_backends:
            backends = df["backend"].unique()[::-1]
        else:
            backends = df["backend"].unique()
        df["std"] = np.sqrt(df["logical_error_rate"] * (1 - df["logical_error_rate"]) / df["num_samples"])
        codes = sorted(df["code"].unique())
        return df, codes, backends

    df_conn, codes_conn, backends_conn = preprocess(connectivity_csv,
                                                    backend_order=['custom_grid', 'custom_cube', 'custom_full'])
    df_topo, codes_topo, backends_topo = preprocess(topology_csv, reverse_backends=True)

    # --- Figure setup ---
    fig, axes = plt.subplots(1, 2, figsize=(2 * WIDE_FIGSIZE, HEIGHT_FIGSIZE), sharey=True)
    bar_width = 0.2
    group_spacing = 0.25
    palette = sns.color_palette("pastel", n_colors=max(len(codes_conn), len(codes_topo)))
    hatches = code_hatches

    # --- Plot helper ---
    def plot_subplot(ax, df, codes, backends, title):
        x = np.arange(len(backends)) * (bar_width * len(codes) + group_spacing)
        for i, code in enumerate(codes):
            subset = df[df["code"] == code]
            means, stds = [], []
            for backend in backends:
                row = subset[subset["backend"] == backend]
                if not row.empty:
                    means.append(row["logical_error_rate"].values[0])
                    stds.append(row["std"].values[0])
                else:
                    means.append(0)
                    stds.append(0)
            ax.bar(
                x + i * bar_width,
                means,
                yerr=stds,
                width=bar_width,
                color=palette[i % len(palette)],
                hatch=hatches[i % len(hatches)],
                edgecolor="black",
                label=code
            )

        ax.set_xticks(x + bar_width * (len(codes) - 1) / 2)
        ax.set_xticklabels([str(b).replace("custom_", "").capitalize() for b in backends], fontsize=FONTSIZE - 2)
        ax.set_yscale("log")
        ax.grid(axis="y", linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.text(
            1.0, 1.14, "Lower is better ↓", transform=ax.transAxes,
            fontsize=12, fontweight="bold", color="blue", va="top", ha="right"
        )

        # Ensure axes spines are black
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1)

    # --- Plot left/right ---
    plot_subplot(axes[0], df_conn, codes_conn, backends_conn, "a) Artificial Topology")
    plot_subplot(axes[1], df_topo, codes_topo, backends_topo, "b) Real Topology")
    axes[0].set_ylabel("Log. Err. Rate (Log)", fontsize=FONTSIZE)

    # --- Shared vertical legend on the right ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(0.8, 0.5),
        fontsize=FONTSIZE,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/connectivity.pdf", format="pdf")
    plt.close(fig)



def generate_technology_plot(path):
    technologies = ["Willow", "Apollo", "Infleqtion"]
    raw_dfs = []

    # Load all data
    for tech in technologies:
        tech_path = os.path.join(path, tech, "results.csv")
        df = pd.read_csv(tech_path)
        raw_dfs.append(df)

    df = pd.concat(raw_dfs, ignore_index=True)

    # Split into two subsets BEFORE mapping backend names
    subsets = {
        "ns": df[df["backend"].str.contains("ns", case=False)].copy(),
        "non_ns": df[~df["backend"].str.contains("ns", case=False)].copy()
    }

    # --- Figure with 2 subplots ---
    fig, axes = plt.subplots(1, 2, figsize=(2 * WIDE_FIGSIZE, HEIGHT_FIGSIZE), sharey=True)

    for i, (key, subset_df) in enumerate(subsets.items()):
        ax = axes[i]
        if subset_df.empty:
            print(f"⚠️ No data for {key}, skipping.")
            continue

        # Map backend names, normalize codes
        subset_df["backend"] = subset_df["backend"].replace(backend_rename_map)
        subset_df["code"] = subset_df["code"].apply(
            lambda x: code_rename_map.get(x.lower(), x.capitalize())
        )

        backends = subset_df["backend"].unique()
        codes = sorted(subset_df["code"].unique())
        n_backends = len(backends)
        n_codes = len(codes)

        # spacing between backend groups
        x = np.arange(n_backends) * (BAR_WIDTH * n_codes + group_spacing)

        for j, code in enumerate(codes):
            code_subset = subset_df[subset_df["code"] == code]
            means = []

            for backend in backends:
                row = code_subset[code_subset["backend"] == backend]
                if not row.empty:
                    means.append(row["logical_error_rate"].values[0])
                else:
                    means.append(0)

            ax.bar(
                x + j * BAR_WIDTH,
                means,
                width=BAR_WIDTH,
                color=code_palette[j % len(code_palette)],
                hatch=code_hatches[j % len(code_hatches)],
                edgecolor="black",
                label=code
            )

        # Axes formatting
        ax.set_xticks(x + BAR_WIDTH * (n_codes - 1) / 2)
        ax.set_xticklabels(backends, fontsize=FONTSIZE - 2)
        if i == 0:
            ax.set_ylabel("Log. Err. Rate (Log)", fontsize=FONTSIZE)

        ax.set_yscale("log")
        ax.grid(axis="y")
        ax.set_axisbelow(True)

        # Title
        plot_title = "a) W/o Shuttling" if key == "ns" else "b) W/ Shuttling"
        ax.set_title(plot_title, loc="left", fontsize=12, fontweight="bold")

        # "Lower is better ↓"
        ax.text(
            1.0, 1.14, "Lower is better ↓", transform=ax.transAxes,
            fontsize=12, fontweight="bold", color="blue",
            va="top", ha="right"
        )

    # ✅ Shared vertical legend on the right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
        fontsize=FONTSIZE,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space for legend
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/technologies.pdf", format="pdf")
    plt.close(fig)


def generate_dqc_plot(csv_path):
    """
    Generate a single PDF with two DQC subplots (error_probability=1 and 10),
    matching the style of other combined plots.
    """
    df = pd.read_csv(csv_path)
    df = df[df["routing_method"] != "basic"]
    df = df[df["code"] != "gross"]

    # Standard renaming
    df["backend"] = df["backend"].replace(backend_rename_map)
    df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))

    fig, axes = plt.subplots(1, 2, figsize=(2 * WIDE_FIGSIZE, HEIGHT_FIGSIZE), sharey=True)

    device = csv_path.split("/")[1].split("_")[1]

    print(df)

    for i, err_prob in enumerate([1, 10]):
        ax = axes[i]
        sub_df = df[df["error_probability"] == err_prob]

        # subset-specific codes and backends
        codes = sorted(sub_df["code"].unique())
        backends = sorted(sub_df["backend"].unique())
        n_codes = len(codes)
        n_backends = len(backends)

        # X positions
        x = np.arange(n_codes)
        total_bar_width = BAR_WIDTH * n_backends
        offsets = np.linspace(-total_bar_width/2 + BAR_WIDTH/2,
                              total_bar_width/2 - BAR_WIDTH/2,
                              n_backends)

        for j, backend in enumerate(backends):
            means, stds = [], []
            for code in codes:
                row = sub_df[(sub_df["code"] == code) & (sub_df["backend"] == backend)]
                if not row.empty:
                    mean_val = row["logical_error_rate"].mean()
                    means.append(mean_val)
                    stds.append(np.sqrt(mean_val * (1 - mean_val) / row["num_samples"].sum()))
                else:
                    means.append(0)
                    stds.append(0)

            ax.bar(
                x + offsets[j],
                means,
                yerr=stds,
                width=BAR_WIDTH,
                color=code_palette[j % len(code_palette)],
                hatch=code_hatches[j % len(code_hatches)],
                edgecolor="black",
                label=backend
            )

        # Axes formatting
        ax.set_xticks(x)
        ax.set_xticklabels(codes, fontsize=FONTSIZE - 2)
        if i == 0:
            ax.set_ylabel("Log. Err. Rate (Log)", fontsize=FONTSIZE)
        ax.set_yscale("log")
        ax.grid(axis="y", linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines['top'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')

        # Title
        plot_label = "a)" if err_prob == 1 else "b)"
        ax.set_title(f"{plot_label} IBM {device} ({100 if err_prob == 1 else 10}%)",
                     loc="left", fontsize=12, fontweight="bold")

        # "Lower is better ↓"
        ax.text(
            1.0, 1.14, "Lower is better ↓", transform=ax.transAxes,
            fontsize=12, fontweight="bold", color="blue",
            va="top", ha="right"
        )

    # Shared vertical legend on the right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(0.83, 0.5),
        fontsize=FONTSIZE,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    os.makedirs("data", exist_ok=True)
    plt.savefig(f"data/dqc_{device}.pdf", format="pdf")
    plt.close(fig)




import numpy as np
import os
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter

def generate_swap_overhead_plot(df_path, backend_label, total_columns=3):
    df = pd.read_csv(df_path)

    # format codes
    def format_code(code):
        code = code.lower()
        return {
            "hh": "Heavy-Hex",
            "surface": "Surface",
            "color": "Color"
        }.get(code, code.capitalize())

    df["code"] = df["code"].apply(format_code)

    routing_methods = df["routing_method"].dropna().unique()
    layout_methods = df["layout_method"].dropna().unique()
    codes = sorted(df["code"].unique())

    n_routing = len(routing_methods)
    n_cols = total_columns
    n_rows = int(np.ceil(n_routing / n_cols))

    # figure setup
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(WIDE_FIGSIZE * 2, HEIGHT_FIGSIZE),
        sharey=True,
        gridspec_kw={'wspace': 0.1}
    )
    axes = np.array(axes).reshape(-1)  # flatten

    bar_width = 0.2
    hatches = ['/', '\\', '//', 'o', '-']
    layout_styles = {
        layout: (code_palette[i % len(code_palette)], hatches[i % len(hatches)])
        for i, layout in enumerate(layout_methods)
    }

    letters = [f"({chr(97+i)})" for i in range(len(routing_methods))]

    # helper formatter: produce "4.5e-5" style, no offset
    def sci_formatter(y, _pos):
        if y == 0:
            return "0"
        s = f"{y:.2e}"          # e.g. "4.50e-05"
        mant, exp = s.split('e')
        mant = mant.rstrip('0').rstrip('.')   # "4.5"
        exp_int = int(exp)                   # -5 (as int)
        return f"{mant}e{exp_int}"           # "4.5e-5"

    func_formatter = FuncFormatter(sci_formatter)

    # plotting
    for i, routing in enumerate(routing_methods):
        ax = axes[i]
        subset = df[df["routing_method"] == routing]

        x = np.arange(len(codes))
        for j, layout in enumerate(layout_methods):
            values = []
            errors = []
            for code in codes:
                entry = subset[(subset["layout_method"] == layout) & (subset["code"] == code)]
                if not entry.empty:
                    mean = entry["swap_overhead_mean"].values[0]
                    var = entry["swap_overhead_var"].values[0]
                else:
                    mean, var = 0, 0
                values.append(mean)
                errors.append(np.sqrt(var))

            ax.bar(
                x + j * bar_width,
                values,
                width=bar_width,
                label=layout.capitalize(),
                color=layout_styles[layout][0],
                edgecolor="black",
                hatch=layout_styles[layout][1],
                yerr=errors,
                capsize=3,
            )

        # title with (a), (b), ...
        ax.set_title(f"{chr(100 + i)}) {routing.capitalize()}", loc="left", fontsize=12, fontweight="bold")
        ax.set_xticks(x + (bar_width * (len(layout_methods) - 1)) / 2)
        ax.set_xticklabels(codes, fontsize=FONTSIZE - 2, rotation=30, ha="right")

        if i % n_cols == 0:
            ax.set_ylabel("SWAP Overhead", fontsize=FONTSIZE)

        ax.grid(axis="y")
        ax.set_axisbelow(True)

        # apply our formatter (inline scientific notation) and hide offset text
        ax.yaxis.set_major_formatter(func_formatter)
        # ensure no separate offset text is shown
        try:
            ax.yaxis.get_offset_text().set_visible(False)
        except Exception:
            pass

    # remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # add "Lower is better ↓" on all subplots that have data
    for ax in axes:
        if ax is not None and ax.has_data():
            ax.text(
                1.0, 1.14, "Lower is better ↓",
                transform=ax.transAxes,
                fontsize=12, fontweight="bold", color="blue",
                va="top", ha="right"
            )

    # adjust margins
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.3)

    # global legend (as before)
    handles = [
        plt.Rectangle((0, 0), 1, 1,
                      facecolor=layout_styles[layout][0],
                      edgecolor='black',
                      hatch=layout_styles[layout][1])
        for layout in layout_methods
    ]
    labels = [layout.capitalize() for layout in layout_methods]

    axes[0].legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.18, 0.31),  # just above the first subplot
        ncol=1,
        fontsize=FONTSIZE,
        frameon=True
    )

    os.makedirs("data", exist_ok=True)
    plt.savefig(f"data/{backend_label}_swap_overhead.pdf", format="pdf")
    plt.close(fig)



import matplotlib.ticker as mtick

def generate_swap_overhead_norm_plot(df_path, backend_label, total_columns=3):

    def format_code(code):
        code = code.lower()
        return {
            "hh": "Heavy-Hex",
            "surface": "Surface",
            "color": "Color"
        }.get(code, code.capitalize())

    # Load main data
    df = pd.read_csv(df_path)
    df["code"] = df["code"].apply(format_code)

    # Load original gate counts
    df_og_gates = pd.read_csv("experiment_results/Translation/results.csv")
    df_og_gates["code"] = df_og_gates["code"].apply(format_code)

    # Map: code -> original 2q gates
    og_gate_dict = dict(zip(df_og_gates["code"], df_og_gates["original_2q_gates"]))

    routing_methods = df["routing_method"].dropna().unique()
    layout_methods = df["layout_method"].dropna().unique()
    codes = sorted(df["code"].unique())

    n_routing = len(routing_methods)
    n_cols = total_columns
    n_rows = int(np.ceil(n_routing / n_cols))

    bar_width = 0.2
    hatches = ['/', '\\', '//', 'o', '-']
    layout_styles = {
        layout: (code_palette[i % len(code_palette)], hatches[i % len(hatches)])
        for i, layout in enumerate(layout_methods)
    }

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2*WIDE_FIGSIZE, HEIGHT_FIGSIZE),
        sharey=True,
        constrained_layout=True
    )
    axes = axes.flatten()

    for i, routing in enumerate(routing_methods):
        ax = axes[i]
        subset = df[df["routing_method"] == routing]

        x = np.arange(len(codes))
        for j, layout in enumerate(layout_methods):
            values = []
            errors = []
            for code in codes:
                entry = subset[(subset["layout_method"] == layout) & (subset["code"] == code)]
                mean = entry["tq_gate_overhead_mean"].values[0] if not entry.empty else 0
                var = entry["tq_gate_overhead_var"].values[0] if not entry.empty else 0

                # normalize by original gate counts
                og_val = og_gate_dict.get(code, np.nan)
                if not np.isnan(og_val) and og_val > 0:
                    mean = (mean / og_val) * 100  # percent
                    var = (var / (og_val ** 2)) * (100 ** 2)  # scale variance for percent
                values.append(mean)
                errors.append(np.sqrt(var))

            ax.bar(
                x + j * bar_width,
                values,
                width=bar_width,
                label=layout.capitalize(),
                color=layout_styles[layout][0],
                edgecolor="black",
                hatch=layout_styles[layout][1],
                yerr=errors,
                capsize=3,
            )

        # subplot title with letter
        ax.set_title(f"{chr(100 + i)}) {routing.capitalize()}",
                     fontsize=12, fontweight='bold', loc='left')

        ax.set_xticks(x + (bar_width * (len(layout_methods) - 1)) / 2)
        ax.set_xticklabels(codes, fontsize=FONTSIZE - 2, rotation=30, ha="right")
        axes[0].tick_params(axis='y', labelsize=FONTSIZE - 2)

        if i % n_cols == 0:
            ax.set_ylabel("% TQ Gates", fontsize=FONTSIZE)

        ax.grid(axis="y")
        ax.set_axisbelow(True)

        # ✅ "Lower is better ↓" on every subplot
        ax.text(
            1.0, 1.14, "Lower is better ↓",
            transform=ax.transAxes,
            fontsize=12, fontweight="bold", color="blue",
            va="top", ha="right"
        )

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1,
                      facecolor=layout_styles[layout][0],
                      edgecolor='black',
                      hatch=layout_styles[layout][1])
        for layout in layout_methods
    ]
    labels = [layout.capitalize() for layout in layout_methods]
    # Legend horizontally at the top of the first subplot
    axes[0].legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.17, 0.33),  # just above the first subplot
        ncol=1,             # horizontal layout
        fontsize=FONTSIZE,
        frameon=True
    )


    plt.savefig(f"data/{backend_label}_swap_overhead_norm.pdf",
                format="pdf")
    plt.close()



def generate_plot_variance(df_path):
    df = pd.read_csv(df_path)

    # Ensure backend is categorical with proper order
    backend_order = ["variance_low", "variance_mid", "variance_high"]
    df["backend"] = pd.Categorical(df["backend"], categories=backend_order, ordered=True)
    df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))

    # Calculate standard deviation using Bernoulli std
    df["std"] = np.sqrt(df["logical_error_rate"] * (1 - df["logical_error_rate"]) / df["num_samples"])

    codes = sorted(df["code"].unique())
    n_codes = len(codes)
    n_backends = len(backend_order)

    # Create figure with same height as size/topology plots
    fig, ax = plt.subplots(figsize=(WIDE_FIGSIZE, HEIGHT_FIGSIZE))

    # spacing between backend groups
    group_spacing = 0.4
    x = np.arange(n_codes) * (BAR_WIDTH * n_backends + group_spacing)

    palette = sns.color_palette("pastel", n_colors=n_backends)
    hatches = code_hatches[:n_backends]

    # Plot bars
    for i, backend in enumerate(backend_order):
        subset = df[df["backend"] == backend]
        means, stds = [], []

        for code in codes:
            row = subset[subset["code"] == code]
            if not row.empty:
                means.append(row["logical_error_rate"].values[0])
                stds.append(row["std"].values[0])
            else:
                means.append(0)
                stds.append(0)

        ax.bar(
            x + i * BAR_WIDTH,
            means,
            yerr=stds,
            width=BAR_WIDTH,
            color=palette[i],
            hatch=hatches[i],
            edgecolor="black",
            label=backend.replace("variance_", "").capitalize()
        )

    # axes formatting
    ax.set_xticks(x + BAR_WIDTH * (n_backends - 1) / 2)
    ax.set_xticklabels(codes, fontsize=FONTSIZE - 2)
    ax.set_ylabel("Logical Error Rate", fontsize=FONTSIZE)
    ax.set_ylim(0, 1.0)  # ensure Y-axis goes from 0 to 1
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])  # fixed steps
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    # title in top-left corner
    ax.set_title(f"{chr(98)}) IBM Heron Noise {df_path.split('/')[1].split('_')[2]}%", loc="left", fontsize=12, fontweight="bold")

    # separate "Lower is better ↓" text
    ax.text(1.0, 1.14, "Lower is better ↓", transform=ax.transAxes,
            fontsize=12, fontweight="bold", color="blue",
            va="top", ha="right")

    # collect unique handles/labels for legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels[l] = h

    # leave space at bottom for legend
    plt.subplots_adjust(bottom=0.3)
    fig.legend(
        handles=list(unique_labels.values()),
        labels=list(unique_labels.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=len(unique_labels),
        frameon=False
    )

    os.makedirs("data", exist_ok=True)
    plt.savefig(f"data/{df_path.split('/')[1]}.pdf", format="pdf")
    plt.close(fig)


def generate_normalized_gate_overhead(df_path):
    """
    Generate one PDF with two subplots of normalized gate overhead:
    - IBM Heron
    - H2
    Shared vertical legend on the right.
    """

    df = pd.read_csv(df_path)
    method_label_map = {
        "tket": "tket",
        "tket_optimized": "tket_optimized",
        "qiskit": "qiskit",
        "qiskit_optimized": "qiskit_optimized",
    }
    df["translating_method"] = df["translating_method"].map(method_label_map)

    gate_sets = ["ibm_heron", "h2"]
    df = df[df["gate_set"].isin(gate_sets) & df["translating_method"].isin(method_label_map)]
    df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))

    # --- Colors ---
    qiskit_base, tket_base = code_palette[0], code_palette[1]
    qiskit_opt = tuple(min(1, c + 0.25) for c in qiskit_base)
    tket_opt = tuple(min(1, c + 0.25) for c in tket_base)

    color_map = {
        "tket": tket_base,
        "tket_optimized": tket_opt,
        "qiskit": qiskit_base,
        "qiskit_optimized": qiskit_opt,
    }
    hatches = ["\\", "//", "/", "\\\\"]

    layout_styles = {
        method: (color_map[method], hatches[i % len(hatches)])
        for i, method in enumerate(method_label_map)
    }
    method_labels = {
        "tket": "TKET",
        "tket_optimized": "TKET Opt.",
        "qiskit": "Qiskit",
        "qiskit_optimized": "Qiskit Opt."
    }

    methods = ["qiskit", "qiskit_optimized", "tket", "tket_optimized"]

    # --- Figure with 2 subplots ---
    fig, axes = plt.subplots(1, 2, figsize=(2 * WIDE_FIGSIZE, HEIGHT_FIGSIZE), sharey=True)

    for i, gate_set in enumerate(gate_sets):
        ax = axes[i]
        subset = df[df["gate_set"] == gate_set]
        if subset.empty:
            print(f"⚠️ No data for {gate_set}, skipping.")
            continue

        codes = sorted(subset["code"].unique())
        x = np.arange(len(codes))

        pivot = subset.pivot(index="code", columns="translating_method", values="gate_overhead_mean")
        total_gates = subset.pivot(index="code", columns="translating_method", values="original_total_gates")

        # Normalize
        for method in methods:
            if method in pivot.columns and method in total_gates.columns:
                pivot[method] = pivot[method] / total_gates[method]

        # --- Bars ---
        for j, method in enumerate(methods):
            if method not in pivot.columns:
                continue
            values = pivot.loc[codes, method].values
            ax.bar(
                x + j * BAR_WIDTH,
                values,
                width=BAR_WIDTH,
                label=method_labels[method],
                color=layout_styles[method][0],
                hatch=layout_styles[method][1],
                edgecolor="black",
            )

        # Title
        device_label = "IBM Heron" if gate_set == "ibm_heron" else "H2"
        ax.set_title(f"{chr(97 + i)}) {device_label}",
                     fontsize=12, fontweight="bold", loc="left")

        # Axes formatting
        ax.set_xticks(x + (BAR_WIDTH * (len(methods) - 1)) / 2)
        ax.set_xticklabels(codes, fontsize=FONTSIZE - 2, rotation=30, ha="right")
        ax.tick_params(axis="y", labelsize=FONTSIZE - 2)
        if i == 0:
            ax.set_ylabel("Norm. Overhead", fontsize=FONTSIZE)

        ax.grid(axis="y")
        ax.set_ylim(0, 12)
        ax.set_axisbelow(True)

        # "Lower is better"
        ax.text(
            1.0, 1.14, "Lower is better ↓",
            transform=ax.transAxes,
            fontsize=12, fontweight="bold", color="blue",
            va="top", ha="right"
        )

    plt.subplots_adjust(left=0.08, bottom=0.3, right=0.86, wspace=0.15)

    # ✅ Shared vertical legend on the right
    handles = [
        plt.Rectangle((0, 0), 1, 1,
                      facecolor=layout_styles[m][0],
                      edgecolor="black",
                      hatch=layout_styles[m][1])
        for m in methods
    ]
    labels = [method_labels[m] for m in methods]
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(0.86, 0.55),
        fontsize=FONTSIZE,
        frameon=False,
    )

    os.makedirs("data", exist_ok=True)
    plt.savefig("data/gate_overhead_normalized.pdf", format="pdf")
    plt.close(fig)






def generate_gate_overhead(df_path):
    """
    Generate one PDF with two subplots (IBM Heron and H2).
    Shared vertical legend on the right.
    """

    df = pd.read_csv(df_path)
    method_label_map = {
        "tket": "tket",
        "tket_optimized": "tket_optimized",
        "qiskit": "qiskit",
        "qiskit_optimized": "qiskit_optimized",
    }
    df["translating_method"] = df["translating_method"].map(method_label_map)

    gate_sets = ["ibm_heron", "h2"]
    df = df[df["gate_set"].isin(gate_sets) & df["translating_method"].isin(method_label_map)]
    df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))

    # --- Colors ---
    qiskit_base, tket_base = code_palette[0], code_palette[1]
    qiskit_opt = tuple(min(1, c + 0.25) for c in qiskit_base)  # lighter variant
    tket_opt = tuple(min(1, c + 0.25) for c in tket_base)

    color_map = {
        "tket": tket_base,
        "tket_optimized": tket_opt,
        "qiskit": qiskit_base,
        "qiskit_optimized": qiskit_opt,
    }
    hatches = ["\\", "//", "/", "\\\\"]

    layout_styles = {
        method: (color_map[method], hatches[i % len(hatches)])
        for i, method in enumerate(method_label_map)
    }
    method_labels = {
        "tket": "TKET",
        "tket_optimized": "TKET Opt.",
        "qiskit": "Qiskit",
        "qiskit_optimized": "Qiskit Opt."
    }

    methods = ["qiskit", "tket", "qiskit_optimized", "tket_optimized"]

    # --- Figure with 2 subplots ---
    fig, axes = plt.subplots(1, 2, figsize=(2 * WIDE_FIGSIZE, HEIGHT_FIGSIZE), sharey=True)

    for i, gate_set in enumerate(gate_sets):
        ax = axes[i]
        subset = df[df["gate_set"] == gate_set]
        if subset.empty:
            print(f"⚠️ No data for {gate_set}, skipping.")
            continue

        codes = sorted(subset["code"].unique())
        x = np.arange(len(codes))
        pivot = subset.pivot(index="code", columns="translating_method", values="gate_overhead_mean")

        for j, method in enumerate(methods):
            if method not in pivot.columns:
                continue
            values = pivot.loc[codes, method].values
            ax.bar(
                x + j * BAR_WIDTH,
                values,
                width=BAR_WIDTH,
                label=method_labels[method],
                color=layout_styles[method][0],
                hatch=layout_styles[method][1],
                edgecolor="black",
            )

        # Title
        device_label = "IBM Heron" if gate_set == "ibm_heron" else "H2"
        ax.set_title(f"{chr(97 + i)}) {device_label}",
                     fontsize=12, fontweight="bold", loc="left")

        # Axes formatting
        ax.set_xticks(x + (BAR_WIDTH * (len(methods) - 1)) / 2)
        ax.set_xticklabels(codes, fontsize=FONTSIZE - 2, rotation=30, ha="right")
        ax.tick_params(axis="y", labelsize=FONTSIZE - 2)
        if i == 0:
            ax.set_ylabel("Gate Overhead", fontsize=FONTSIZE)

        ax.grid(axis="y")
        ax.set_axisbelow(True)

        # "Lower is better"
        ax.text(
            1.0, 1.14, "Lower is better ↓",
            transform=ax.transAxes,
            fontsize=12, fontweight="bold", color="blue",
            va="top", ha="right"
        )

    plt.subplots_adjust(left=0.08, bottom=0.3, right=0.86, wspace=0.15)

    # ✅ Shared vertical legend on the right
    handles = [
        plt.Rectangle((0, 0), 1, 1,
                      facecolor=layout_styles[m][0],
                      edgecolor="black",
                      hatch=layout_styles[m][1])
        for m in methods
    ]
    labels = [method_labels[m] for m in methods]
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(0.87, 0.55),
        fontsize=FONTSIZE,
        frameon=False,
    )

    os.makedirs("data", exist_ok=True)
    plt.savefig("data/gate_overhead.pdf", format="pdf")
    plt.close(fig)

def generate_plot_variance_two(low_noise_csv, high_noise_csv):
    # --- Preprocess function ---
    def preprocess(path):
        df = pd.read_csv(path)
        backend_order = ["variance_low", "variance_mid", "variance_high"]
        df["backend"] = pd.Categorical(df["backend"], categories=backend_order, ordered=True)
        df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))
        df["std"] = np.sqrt(df["logical_error_rate"] * (1 - df["logical_error_rate"]) / df["num_samples"])
        return df, backend_order

    df_low, backend_order = preprocess(low_noise_csv)
    df_high, _ = preprocess(high_noise_csv)

    # --- Figure settings ---
    fig, axes = plt.subplots(1, 2, figsize=(2 * WIDE_FIGSIZE, HEIGHT_FIGSIZE), sharey=True)
    bar_width = BAR_WIDTH
    group_spacing = 0.25
    codes = sorted(set(df_low["code"].unique()).union(df_high["code"].unique()))

    palette = code_palette
    hatches = ['/', '\\', '//']

    # --- Plot helper ---
    def plot_variance(ax, df, title, show_labels=False):
        n_backends = len(backend_order)
        x = np.arange(len(codes)) * (bar_width * n_backends + group_spacing)
        for i, backend in enumerate(backend_order):
            subset = df[df["backend"] == backend]
            means, stds = [], []
            for code in codes:
                row = subset[subset["code"] == code]
                if not row.empty:
                    means.append(row["logical_error_rate"].values[0])
                    stds.append(row["std"].values[0])
                else:
                    means.append(0)
                    stds.append(0)
            ax.bar(
                x + i * bar_width,
                means,
                yerr=stds,
                width=bar_width,
                color=palette[i],
                hatch=hatches[i % len(hatches)],
                edgecolor="black",
                label=backend.replace("variance_", "").capitalize() if show_labels else None
            )

        ax.set_xticks(x + bar_width * (n_backends - 1) / 2)
        ax.set_xticklabels(codes, fontsize=FONTSIZE - 2)
        ax.set_yscale("log")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.text(
            1.0, 1.14, 'Lower is better ↓',
            transform=ax.transAxes,
            fontsize=12, fontweight='bold', color="blue",
            va='top', ha='right'
        )
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")

    # --- Plot subplots ---
    plot_variance(axes[0], df_high, "a) IBM Heron Noise 100%", show_labels=True)
    axes[0].set_ylabel("Log. Err. Rate (Log)", fontsize=FONTSIZE)
    plot_variance(axes[1], df_low, "b) IBM Heron Noise 10%")
    

    # --- Shared vertical legend on the right ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(0.83, 0.5),
        fontsize=FONTSIZE,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/plot_variance.pdf", format="pdf")
    plt.close(fig)




if __name__ == '__main__':
    size = "experiment_results/Size_full/results.csv"
    size_grid = "experiment_results/Size_grid/results.csv"
    connectivity = "experiment_results/Connectivity_small/results.csv"
    topology = "experiment_results/Topology/results.csv"
    path = "experiment_results"
    df_grid = "experiment_results/Routing_grid/results.csv"
    df_hh = "experiment_results/Routing_hh/results.csv"
    variance_high = "experiment_results/Variance_noise_100/results.csv"
    variance_low = "experiment_results/Variance_noise_10/results.csv"
    gate_overhead = "experiment_results/Translation/results.csv"
    dqc_flamingo = "experiment_results/DQC_Flamingo/results.csv"
    dqc_nighthawk = "experiment_results/DQC_Nighthawk/results.csv"
    generate_size_plot(size)
    #generate_connectivity_plot(connectivity)
    #generate_topology_plot(topology)
    #generate_connectivity_topology_plot(connectivity, topology)
    #generate_plot_variance(variance_high)
    #generate_plot_variance(variance_low)
    #generate_technology_plot(path)
    #generate_dqc_plot(dqc_flamingo)
    #generate_dqc_plot(dqc_nighthawk)
    #generate_swap_overhead_plot(df_grid, "Grid")
    #generate_swap_overhead_norm_plot(df_grid, "Grid")
    generate_swap_overhead_norm_plot(df_hh, "Heavy-Hex")
    generate_swap_overhead_plot(df_hh, "Heavy-Hex")
    #generate_plot_variance_two(low_noise_csv=variance_low, high_noise_csv=variance_high)
    #generate_gate_overhead(gate_overhead)
    #generate_normalized_gate_overhead(gate_overhead)
