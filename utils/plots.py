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
    'steane': 'X',
    'color': '*',
    'other': ''
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

backend_rename_map = {
    "real_willow": "Willow",
    "real_infleqtion": "Infleqtion",
    "real_nsinfleqtion": "Infleqtion w/o\nshuttling",
    "real_apollo": "Apollo",
    "real_flamingo": "Flamingo",
    #"real_nighthawk": "Nighthawk"
}

code_palette = sns.color_palette("pastel", n_colors=6)
code_hatches = ["/", "\\", "//", "++", "xx", "**"]

def generate_size_plot(df_path):
    df = pd.read_csv(df_path)
    error_types = ['Constant']
    error_probs = [0.002, 0.004, 0.008]
    df_filtered = df[~df['code'].str.contains('heavyhex', case=False, na=False)]
    df_filtered = df_filtered[~df_filtered['backend'].str.contains('heavyhex', case=False, na=False)]
    backends = df_filtered['backend'].unique()

    os.makedirs("../data", exist_ok=True)

    for backend in backends:
        n_rows = 1
        n_cols = len(error_probs) * len(error_types)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2), sharey=True, gridspec_kw={'wspace': 0.03})
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


                    HIGHLIGHT = {
                        ('surface',  'Constant'): [400, 500],
                        ('hh', 'Constant'):   [450],          
                        ('color','Constant'): [400,500],
                        ('bacon','Constant'): [400,450],
                        ('surface',  'SI1000'): [400, 500],
                        ('hh', 'SI1000'):   [450],          
                        ('color','SI1000'): [400,500],
                        ('bacon','SI1000'): [400,450],
                    }
                    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

                    # stable order of codes across the whole figure
                    all_codes = [c.lower() for c in df_filtered['code'].unique()]
                    all_codes.sort()  # or keep your preferred order

                    # map each code -> one of the default colors
                    code_color = {c: default_colors[i % len(default_colors)]
                                for i, c in enumerate(all_codes)}

                    line = ax.plot(
                        group_sorted['backend_size'],
                        group_sorted['logical_error_rate'],
                        label=code_display,
                        marker=marker,
                        color=code_color[code_key],   # <- lock to default palette color
                    )[0]

                    # highlight only specific x positions (does NOT change the cycle)
                    highlight_x = HIGHLIGHT.get((code_key, et), [])
                    if highlight_x:
                        xs = group_sorted['backend_size'].to_numpy()
                        ys = group_sorted['logical_error_rate'].to_numpy()
                        sel = np.isin(xs, highlight_x)

                        m  = line.get_marker() or 'o'
                        ms = line.get_markersize()
                        base = code_color[code_key]
                        mfc = line.get_markerfacecolor()
                        if (mfc is None) or (mfc == 'auto'):
                            mfc = base

                        ax.plot(
                            xs[sel], ys[sel],
                            linestyle='None',
                            marker=m,
                            markersize=ms * 1.4,           # slightly larger for the ring
                            markerfacecolor=mfc,           # same fill as the line
                            markeredgecolor='black',
                            markeredgewidth=max(1.5, ms*0.2),
                            color=base,                    # explicit color -> no cycle advance
                            zorder=line.get_zorder() + 2,
                            label='_nolegend_',            # don’t duplicate legend entries
                        )



                ax.set_xlabel('Backend Size', fontsize=12)
                xticks = sorted(subset['backend_size'].unique())
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticks, fontsize=12)
                ax.set_title(f'({chr(100 + j)}) {et} ({p})', loc='left', fontsize=14, fontweight='bold')
                ax.text(1.0, 1.14, 'Lower is better ↓', transform=ax.transAxes,
                            fontsize=12, fontweight='bold', color="blue", va='top', ha='right')
                if idx == 0 and j == 0:
                    ax.set_ylabel('Logical Error Rate', fontsize=12)
                elif idx == 0 and j == 1:
                    ax.text(1.0, 1.14, 'Lower is better ↓', transform=ax.transAxes,
                            fontsize=12, fontweight='bold', color="blue", va='top', ha='right')

                ax.grid(True)
                ax.set_ylim(0, 0.65)
                handles, labels = ax.get_legend_handles_labels()
                all_handles_labels.extend(zip(handles, labels))

        unique_labels = {}
        for h, l in all_handles_labels:
            if l not in unique_labels:
                unique_labels[l] = h

        fig.legend(
            handles=list(unique_labels.values()),
            labels=list(unique_labels.keys()),
            #title='Code',
            fontsize=12,
            title_fontsize=12,
            loc='upper center',
            ncols=len(unique_labels),
            bbox_to_anchor=(0.5, -0.1)
        )

        #plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        plt.savefig(f"data/size_{backend}_{error_types}.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig)

def generate_connectivity_plot(df_path):
    df = pd.read_csv(df_path)
    df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))
    backend_order = ['custom_grid', 'custom_cube', 'custom_full']
    df["backend"] = pd.Categorical(df["backend"], categories=backend_order, ordered=True)
    df["std"] = np.sqrt(df["logical_error_rate"] * (1 - df["logical_error_rate"]) / df["num_samples"])

    sns.set(style="whitegrid")
    plt.figure(figsize=(13, 4))

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

    plt.xticks(x + bar_width * (len(codes) - 1) / 2, [b.replace("custom_", "").capitalize() for b in backends],fontsize=12)
    #plt.xlabel("Backend Connectivity")
    plt.ylabel("Logical Error Rate (Log)", fontsize=12)
    plt.title("Logical Error Rate by Backend Connectivity", loc='left', fontweight='bold', fontsize=14)
    plt.yscale("log")
    plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
    plt.legend(
        #title="QEC Code",
        loc='lower center',
        bbox_to_anchor=(0.5, -0.35),
        ncol=len(codes),
        fontsize=14,
        title_fontsize=12,
        frameon=False
    )
    plt.text(1.00, 1.15, 'Lower is better ↓', transform=plt.gca().transAxes,
             fontsize=14, fontweight='bold', color="blue", va='top', ha='right')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/connectivity.pdf", format="pdf", bbox_inches="tight")
    plt.close()

def generate_topology_plot(df_path):
    df = pd.read_csv(df_path)
    df["backend"] = df["backend"].replace(backend_rename_map)
    df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))
    df["std"] = np.sqrt(df["logical_error_rate"] * (1 - df["logical_error_rate"]) / df["num_samples"])

    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 5))

    backends = df["backend"].unique()
    backends = [backends[3], backends[1], backends[0], backends[2]]  # Reorder for better visualization
    codes = sorted(df["code"].unique())
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

    plt.xticks(x + bar_width * (len(codes) - 1) * 3/5, backends, rotation=0, ha="right", fontsize=14)
    #plt.xlabel("Backend")
    plt.ylabel("Logical Error Rate (Log)", fontsize=12)
    plt.title("Logical Error Rate by Backend Topology", loc='left', fontweight='bold', fontsize=14)
    plt.yscale("log")
    plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
    plt.legend(
        #title="QEC Code",
        loc='lower center',
        bbox_to_anchor=(0.5, -0.51),
        ncol=len(codes),
        fontsize=14,
        title_fontsize=12,
        frameon=False
    )
    plt.text(1.00, 1.10, 'Lower is better ↓', transform=plt.gca().transAxes,
             fontsize=14, fontweight='bold', color="blue", va='top', ha='right')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/topology.pdf", format="pdf", bbox_inches="tight")
    plt.close()

def generate_technology_plot(path):
    #technologies = ["Willow", "Apollo", "Infleqtion", "DQC", "Nighthawk"]
    technologies = ["Willow", "Apollo", "Infleqtion", "DQC"]
    dfs = []

    for tech in technologies:
        tech_path = os.path.join(path, tech, "results.csv")
        df = pd.read_csv(tech_path)
        df["backend"] = df["backend"].replace(backend_rename_map)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Normalize code labels
    df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))

    # Ensure consistent order
    backends = df["backend"].unique()
    codes = sorted(df["code"].unique())

    df["backend"] = pd.Categorical(df["backend"], categories=backends, ordered=True)
    df["code"] = pd.Categorical(df["code"], categories=codes, ordered=True)

    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 3))

    x = np.arange(len(backends))
    bar_width = 0.8 / len(codes)

    for i, code in enumerate(codes):
        means = []
        for backend in backends:
            subset = df[(df["backend"] == backend) & (df["code"] == code)]
            if not subset.empty:
                means.append(subset["logical_error_rate"].values[0])
            else:
                means.append(0)

        plt.bar(
            x + i * bar_width,
            means,
            width=bar_width,
            color=code_palette[i % len(code_palette)],
            hatch=code_hatches[i % len(code_hatches)],
            edgecolor="black",
            label=code
        )

    plt.xticks(x + bar_width * (len(codes) - 1) * 3/4, backends, rotation=0, ha="right", fontsize=14)
    #plt.xlabel("Backend")
    plt.ylabel("Logical Error Rate (Log)", fontsize=12)
    plt.title("Logical Error Rate by Backend and QEC Code", loc='left', fontweight='bold', fontsize=14)
    plt.yscale("log")
    plt.legend(
        #title="QEC Code",
        loc='lower center',
        bbox_to_anchor=(0.5, -0.51),
        ncol=len(codes),
        fontsize=14,
        title_fontsize=12,
        frameon=False
    )

    plt.text(1.00, 1.10, 'Lower is better ↓', transform=plt.gca().transAxes,
             fontsize=14, fontweight='bold', color="blue", va='top', ha='right')

    plt.subplots_adjust(bottom=0.25)  # Reserve space for rotated labels + legend
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/technologies.pdf", format="pdf", bbox_inches="tight")
    plt.close()

def generate_dqc_plot(path):
    datasets_normal = [
        ("DQC", "DQC Full Size"),
        ("DQC_1_QPU", "DQC 1 QPU Size")
    ]

    datasets_lower = [
        ("DQC_LOWER", "DQC Full Size (Lower)"),
        ("DQC_1_QPU_LOWER", "DQC 1 QPU Size (Lower)")
    ]

    all_dataset_groups = [
        (datasets_normal, "Normal"),
        (datasets_lower, "Lower")
    ]

    dfs = []

    # Load all datasets (normal + lower)
    for dataset_group, _ in all_dataset_groups:
        for folder, label in dataset_group:
            tech_path = os.path.join(path, folder, "results.csv")
            df = pd.read_csv(tech_path)
            df["backend"] = df["backend"].replace(backend_rename_map)
            df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))
            df["dataset"] = label
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # 🚨 Exclude "Gross" completely
    df = df[df["code"].str.lower() != "gross"]

    # Codes
    codes = sorted(df["code"].unique())
    df["code"] = pd.Categorical(df["code"], categories=codes, ordered=True)

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)  # shorter height

    bar_width = 0.1  # thinner bars

    # One subplot per group (Normal, Lower)
    for ax, (dataset_group, title_suffix) in zip(axes, all_dataset_groups):
        datasets_labels = [label for _, label in dataset_group]

        # Filter only the subset for this group
        subdf = df[df["dataset"].isin(datasets_labels)].copy()
        subdf["dataset"] = pd.Categorical(subdf["dataset"], categories=datasets_labels, ordered=True)

        x = np.arange(len(codes)) * 0.5

        for j, dataset_label in enumerate(datasets_labels):
            means = []
            for code in codes:
                subset = subdf[(subdf["code"] == code) & (subdf["dataset"] == dataset_label)]
                means.append(subset["logical_error_rate"].mean() if not subset.empty else 0)

            ax.bar(
                x + (j - 0.5) * bar_width,
                means,
                width=bar_width,
                color=code_palette[j % len(code_palette)],
                hatch='/' if j == 0 else '\\',
                edgecolor="black",
                label=dataset_label
            )

        ax.set_xticks(x)
        ax.set_xticklabels(codes, rotation=0, ha="center", fontsize=12)
        if title_suffix == "Lower":
            ax.set_title(f"b) Noise 10%",
                     loc='left', fontweight='bold', fontsize=14)
        elif title_suffix == "Normal":
            ax.set_title(f"a) Noise 100%",
                     loc='left', fontweight='bold', fontsize=14)
        # No log scale anymore

    axes[0].set_ylabel("Logical Error Rate", fontsize=12)

    custom_labels = ["Full Size", "1 QPU Size"]

    axes[1].legend(
        labels=custom_labels,
        loc='center left',         # stick to the left side of the bbox
        bbox_to_anchor=(0.58, 0.86),  # (x=1.02 → just outside, y=0.5 → vertically centered)
        fontsize=12
    )

    fig.text(-0.05, 1.08, 'Lower is better ↓', transform=axes[1].transAxes,
             fontsize=12, fontweight='bold', color="blue", va='top', ha='right')

    fig.text(1.0, 1.08, 'Lower is better ↓', transform=axes[1].transAxes,
             fontsize=12, fontweight='bold', color="blue", va='top', ha='right')

    #plt.subplots_adjust(bottom=0.25, wspace=0.15)
    plt.subplots_adjust(bottom=0.25, wspace=0.05)
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/dqc_flamingo.pdf", format="pdf", bbox_inches="tight")
    plt.close()


def generate_swap_overhead_plot(df_path, backend_label, total_columns=3):

    def format_code(code):
        code = code.lower()
        return {
            "hh": "Heavy-Hex",
            "surface": "Surface",
            "color": "Color"
        }.get(code, code.capitalize())

    df = pd.read_csv(df_path)
    df["code"] = df["code"].apply(format_code)
    
    routing_methods = df["routing_method"].dropna().unique()
    layout_methods = df["layout_method"].dropna().unique()
    codes = sorted(df["code"].unique())

    n_routing = len(routing_methods)
    n_cols = total_columns
    n_rows = int(np.ceil(n_routing / n_cols))

    plot_width_per_col = 6
    plot_height_per_row = 5

    bar_width = 0.2
    palette = sns.color_palette("pastel", n_colors=len(layout_methods))
    #hatches = ['/', 'o', '*', '\\', '-']
    hatches = ['/', '\\', '//', 'o', '-']
    layout_styles = {
        layout: (palette[i % len(palette)], hatches[i % len(hatches)])
        for i, layout in enumerate(layout_methods)
    }

    #fig, axes = plt.subplots(n_rows, n_cols, figsize=(plot_width_per_col * n_cols, plot_height_per_row * n_rows), sharey=True, constrained_layout=True)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18,4), sharey=True, constrained_layout=True)
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
                mean = entry["swap_overhead_mean"].values[0] if not entry.empty else 0
                var = entry["swap_overhead_var"].values[0] if not entry.empty else 0
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
        #add (a), (b), (c) in front of title
        ax.set_title(f"({chr(97 + i)}) {routing.capitalize()}", fontsize=18, fontweight='bold', loc='left')
        ax.set_xticks(x + (bar_width * (len(layout_methods) - 1)) / 2)
        ax.set_xticklabels(codes, rotation=75, ha="center", fontsize=16)
        axes[0].tick_params(axis='y', labelsize=16)
        #ax.set_xlabel("QEC Code")
        if i % n_cols == 0:
            ax.set_ylabel("SWAP Overhead", fontsize=16)
            #ax.set_yticks(ax.get_yticks())
            #ax.set_yticklabels([f"{int(tick)}" for tick in ax.get_yticks()], fontsize=14)

        # ✅ Add horizontal grid lines
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    ylim = axes[0].get_ylim()
    xlim = axes[0].get_xlim()

    axes[0].text(
        xlim[1],
        ylim[1] * 1.10,
        "Lower is better ↓",
        fontsize=16, fontweight='bold', color="blue", va='top', ha='right'
    )
    axes[1].text(
        xlim[1],
        ylim[1] * 1.10,
        "Lower is better ↓",
        fontsize=16, fontweight='bold', color="blue", va='top', ha='right'
    )
    axes[2].text(
        xlim[1],
        ylim[1] * 1.10,
        "Lower is better ↓",
        fontsize=16, fontweight='bold', color="blue", va='top', ha='right'
    )

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    handles = [
        plt.Rectangle((0, 0), 1, 1,
                    facecolor=layout_styles[layout][0],
                    edgecolor='black',
                    hatch=layout_styles[layout][1])
        for layout in layout_methods
    ]
    labels = [layout.capitalize() for layout in layout_methods]

    # Put legend inside the first subplot (axes[0])
    axes[0].legend(
        handles, labels,
        title="Layout Method",
        loc="upper left",   # you can adjust this
        fontsize=16, title_fontsize=16,
        frameon=True
    )

    #plt.title(f"SWAP Overhead on {backend_label} Architecture", fontsize=16, y=1.12, fontweight='bold', ha='left')
    #plt.tight_layout()
    plt.savefig("data/" + backend_label + "_swap_overhead.pdf", format="pdf", bbox_inches="tight")
    plt.close()

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
    palette = sns.color_palette("pastel", n_colors=len(layout_methods))
    hatches = ['/', '\\', '//', 'o', '-']
    layout_styles = {
        layout: (palette[i % len(palette)], hatches[i % len(hatches)])
        for i, layout in enumerate(layout_methods)
    }

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(18, 4),
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
                mean = entry["swap_overhead_mean"].values[0] if not entry.empty else 0
                var = entry["swap_overhead_var"].values[0] if not entry.empty else 0

                og_val = og_gate_dict.get(code, np.nan)
                if not np.isnan(og_val) and og_val > 0:
                    mean = (mean / og_val) * 100  # convert to percent
                    var = (var / (og_val ** 2)) * (100 ** 2)  # scale variance for percentage
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

        ax.set_title(f"({chr(100 + i)}) {routing.capitalize()}",
                     fontsize=18, fontweight='bold', loc='left')
        ax.set_xticks(x + (bar_width * (len(layout_methods) - 1)) / 2)
        ax.set_xticklabels(codes, rotation=75, ha="center", fontsize=16)
        axes[0].tick_params(axis='y', labelsize=16)

        if i % n_cols == 0:
            ax.set_ylabel("SWAP Overhead [%]", fontsize=16)

        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Add "lower is better" note
    ylim = axes[0].get_ylim()
    xlim = axes[0].get_xlim()
    for k in range(min(3, len(axes))):
        axes[k].text(
            xlim[1],
            ylim[1] * 1.1,
            "Lower is better ↓",
            fontsize=16, fontweight='bold', color="blue", va='top', ha='right'
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

    plt.savefig(f"data/{backend_label}_swap_overhead_norm.pdf",
                format="pdf", bbox_inches="tight")
    plt.close()


def generate_plot_variance(df_path):
    df = pd.read_csv(df_path)  # Update with the correct file path

    # Ensure `backend` is categorical with proper order
    backend_order = ["variance_low", "variance_mid", "variance_high"]
    df["backend"] = pd.Categorical(df["backend"], categories=backend_order, ordered=True)
    df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))

    # Calculate standard deviation using Bernoulli trial std formula
    df["std"] = np.sqrt(df["logical_error_rate"] * (1 - df["logical_error_rate"]) / df["num_samples"])

    # Plot settings
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 3))

    # Unique codes sorted
    codes = sorted(df["code"].unique())
    x = np.arange(len(codes))
    bar_width = 0.2

    # Color palette
    palette = sns.color_palette("pastel", n_colors=3)
    hatches = ['/', '\\', '//']

    # Plot bars for each backend
    for i, backend in enumerate(backend_order):
        subset = df[df["backend"] == backend]
        means = []
        stds = []

        for code in codes:
            row = subset[subset["code"] == code]
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
            color=palette[i],
            hatch=hatches[i],
            edgecolor="black",
            label=backend.replace("variance_", "").capitalize()
        )

    # Axes and labels
    plt.xticks(x + bar_width, codes, rotation=0, ha="center", fontsize=12)
    #plt.xlabel("Quantum Error Correction Code")
    plt.ylabel("Logical Error Rate", fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Logical Error Rate by Code", loc='left', fontweight='bold', fontsize=14)
    plt.legend(title="Variance", loc='lower center' , bbox_to_anchor=(1.055, 0.45), fontsize=12, frameon=True)
    plt.text(1.00, 1.2, 'Lower is better ↓', transform=plt.gca().transAxes,
             fontsize=12, fontweight='bold', color="blue", va='top', ha='right')
    plt.tight_layout()

    # Save / Show
    plt.savefig("data/plot_variance.pdf", format="pdf", bbox_inches="tight")
    plt.close()

def generate_normalized_gate_ovehead(df_path):
    df = pd.read_csv(df_path)
    method_label_map = {
        "tket": "tket",
        "qiskit": "qiskit",
        "qiskit_optimized": "qiskit_optimized",
    }
    df["translating_method"] = df["translating_method"].map(method_label_map)
    
    gate_sets = ["ibm_heron", "h2"]
    translation_methods = ["tket", "qiskit", "qiskit_optimized"]
    df = df[df["gate_set"].isin(gate_sets) & df["translating_method"].isin(translation_methods)]

    df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))

    # Plot settings
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 4), sharey=True)

    # Set pastel palette base color
    base_palette = sns.color_palette("pastel", n_colors=2)
    qiskit_color = base_palette[0]  # shared base for both Qiskits
    tket_color = base_palette[1]
    optimized_qiskit_color = tuple(min(1, c + 0.2) for c in qiskit_color)

    # Colors and hatches
    color_map = {
        "tket": tket_color,
        "qiskit": qiskit_color,
        "qiskit_optimized": optimized_qiskit_color,
    }
    hatches = ["\\", "/", "//"]  # different hatches per method

    # Sort and format code labels
    codes = sorted(df["code"].unique())

    # Plot bars with hatching
    bar_width = 0.25
    x = np.arange(len(codes))
    method_labels = {
        "tket": "TKET",
        "qiskit": "Qiskit",
        "qiskit_optimized": "Qiskit Optimized"
    }
    for i, gate_set in enumerate(gate_sets):
        ax = axes[i]
        subset = df[df["gate_set"] == gate_set]

        pivot = subset.pivot(index="code", columns="translating_method", values="gate_overhead_mean")
        total_gates = subset.pivot(index="code", columns="translating_method", values="original_total_gates")

        # Normalize gate overhead by total gates
        for method in translation_methods:
            if method in pivot.columns and method in total_gates.columns:
                #pivot[method] = pivot[method] / total_gates[method]
                pivot[method] = pivot[method]

        for j, method in enumerate(translation_methods):
            values = pivot[method].values
            bars = ax.bar(
                x + j * bar_width,
                values,
                width=bar_width,
                color=color_map[method],
                hatch=hatches[j % len(hatches)],
                edgecolor="black",
                label=method_labels[method]
            )

        ax.set_title(f"({chr(97 + i)}) Gate Overhead ({'IBM Heron' if gate_set == 'ibm_heron' else 'H2'})", fontsize=18, fontweight='bold', loc='left')
        #ax.set_xlabel("Quantum Error Correction Code")
        axes[0].set_ylabel("Gate Overhead", fontsize=16)
        axes[0].tick_params(axis='y', labelsize=16)
        
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(codes, rotation=0, ha="center", fontsize=16)

        # Add "Lower is better ↓" in top-left corner
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        axes[0].text(
            xlim[1],                
            ylim[1] * 1.07,         # near top
            "Lower is better ↓",
            fontsize=16,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="right",
            color="blue"
        )

        axes[1].text(
            xlim[1],                
            ylim[1] * 1.07,         # near top
            "Lower is better ↓",
            fontsize=16,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="right",
            color="blue"
        )

    # Legend and layout
    axes[1].legend(labels=['TKET', 'Qiskit', 'Qiskit Optimized'], fontsize=16, ncol=3)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    # Uncomment to save
    plt.savefig("data/translation.pdf", format="pdf", bbox_inches="tight")
    plt.close()

if __name__ == '__main__':
    size = "experiment_results/Size_full/results.csv"
    connectivity = "experiment_results/Connectivity_small/results.csv"
    topology = "experiment_results/Topology/results.csv"
    path = "experiment_results"
    df_grid = "experiment_results/Routing_grid/results.csv"
    df_hh = "experiment_results/Routing_hh/results.csv"
    plot_variance = "experiment_results/Variance/results.csv"
    gate_overhead = "experiment_results/Translation/results.csv"
    #generate_size_plot(size)
    #generate_connectivity_plot(connectivity)
    #generate_topology_plot(topology)
    #generate_technology_plot(path)
    #generate_dqc_plot(path)
    generate_swap_overhead_plot(df_grid, "Grid")
    #generate_swap_overhead_norm_plot(df_grid, "Grid")
    #generate_swap_overhead_plot(df_hh, "Heavy-Hex")
    #generate_plot_variance(plot_variance)
    #generate_normalized_gate_ovehead(gate_overhead)
