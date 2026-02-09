"""
ML²B Benchmark Pass Rate Visualization
Reads actual data from uploaded CSV and calculates pass rates
If cell has less than 3 values, marks as NaN
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import re

# Set up publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def parse_cell(cell_value):
    """
    Parse a cell like "0.904, 0.906, None" and return (pass_count, total_runs)

    Rules:
    - Count valid numeric values as passes.
    - If total_runs < 3, still return (successful_runs, total_runs) so we can show 0/3, 1/3, 2/3.

    Returns:
        tuple: (successful_runs, total_runs)
    """
    if pd.isna(cell_value) or cell_value == '' or cell_value is None:
        return (0, 0)

    # Convert to string if not already
    cell_str = str(cell_value).strip()

    # Split by comma (single value without comma gives total_runs=1)
    parts = [p.strip() for p in cell_str.split(',')]
    total_runs = len(parts)
    successful_runs = 0

    for part in parts:
        # Check if it's None or contains 'None' or error messages
        part_lower = part.lower()
        if ('none' in part_lower or
            'nan' in part_lower or
            'error' in part_lower or
            'es' in part_lower and len(part) < 20 or  # None(es)
            part == '' or
            'unexpected' in part_lower or
            'unk' in part_lower):
            continue

        # Try to parse as number
        try:
            # Remove any parenthetical notes
            clean_part = re.sub(r'\([^)]*\)', '', part).strip()
            if clean_part:
                val = float(clean_part)
                if not np.isnan(val):
                    successful_runs += 1
        except (ValueError, TypeError):
            continue

    return (successful_runs, total_runs)


def calculate_pass_rate(cell_value):
    """
    Calculate pass rate from a cell value.
    Empty/no data → 0 (display as 0/3). If total < 3 runs, rate is successful/3.
    """
    successful, total = parse_cell(cell_value)
    if total == 0:
        return 0.0  # Empty cell → 0/3
    if total < 3:
        return successful / 3.0  # Show as 0/3, 1/3, 2/3 scale
    return successful / total


def calculate_pass_fraction(cell_value):
    """
    Return pass fraction as string like "2/3" for display.
    Empty cell → "0/3". If total < 3, show as "0/3", "1/3", or "2/3".
    """
    successful, total = parse_cell(cell_value)
    if total == 0:
        return "0/3"
    if total < 3:
        return f"{successful}/3"
    return f"{successful}/{total}"


def load_data(filepath):
    """
    Load and process the CSV file with multi-level headers
    """
    # Read raw to understand structure
    df_raw = pd.read_csv(filepath, header=[0, 1])

    print("Columns found:", df_raw.columns.tolist()[:10], "...")
    print("Shape:", df_raw.shape)

    return df_raw


def process_model_data(filepath, model_name='gpt-4.1-mini'):
    """
    Extract data for a specific model from the multi-header CSV
    """
    # Read with multi-level header
    df = pd.read_csv(filepath, header=[0, 1])

    # Get competition names from first columns
    # The structure is: №, comp_name, then model columns

    # Find columns for the specified model
    model_cols = [col for col in df.columns if model_name.lower() in str(col[0]).lower()]

    print(f"\nProcessing model: {model_name}")
    print(f"Found {len(model_cols)} columns for this model")

    # Extract competition names
    comp_name_col = [col for col in df.columns if 'comp_name' in str(col).lower()][0]
    comp_names = df[comp_name_col].dropna().tolist()

    # Build data dictionary
    languages = ['Arab', 'Belarus', 'Chinese', 'English', 'Italian', 'Japanese',
                 'Kazakh', 'Polish', 'Romanian', 'Spanish', 'Russian', 'French', 'Turkish']

    data_dict = {'comp_name': comp_names}

    for lang in languages:
        lang_col = [col for col in model_cols if lang in str(col[1])]
        if lang_col:
            data_dict[lang] = df[lang_col[0]].tolist()[:len(comp_names)]
        else:
            data_dict[lang] = [None] * len(comp_names)

    return pd.DataFrame(data_dict)


# === MAIN DATA LOADING ===
print("=" * 60)
print("Loading ML²B Data from CSV")
print("=" * 60)

# Read the CSV file
filepath = 'final_version - chosen (final).csv'

# Read raw CSV to manually parse
df_raw = pd.read_csv(filepath, header=None)
print(f"\nRaw shape: {df_raw.shape}")
print(f"First few rows:")
print(df_raw.head(3))

# The structure is:
# Row 0: Model names (gpt-oss-120, gemini-2.5-flash, gpt-4.1-mini, gpt-oss (ReAct))
# Row 1: baseline_score, domain, metric (skip for heatmaps), then language names (Arab, Belarus, Chinese, Ukranian, ...)
# Row 2+: Data

# Extract model info from row 0
model_row = df_raw.iloc[0].tolist()
lang_row = df_raw.iloc[1].tolist()

# Find column ranges for each model
models = {}
current_model = None
current_start = None

for i, val in enumerate(model_row):
    if pd.notna(val) and val not in ['№', 'comp_name', '']:
        if current_model is not None:
            models[current_model] = (current_start, i)
        current_model = val
        current_start = i
if current_model is not None:
    models[current_model] = (current_start, len(model_row))

print(f"\nModels found: {list(models.keys())}")
for m, (s, e) in models.items():
    print(f"  {m}: columns {s}-{e}")

# Extract data for each model
# Order matches group division: high (5) → medium (6) → low (3)
languages_order = ['English', 'French', 'Spanish', 'Italian', 'Chinese',
                   'Japanese', 'Russian', 'Polish', 'Turkish', 'Arab', 'Ukranian',
                   'Romanian', 'Kazakh', 'Belarus']
lang_display = ['EN', 'FR', 'ES', 'IT', 'ZH', 'JA', 'RU', 'PL', 'TR', 'AR', 'UK', 'RO', 'KK', 'BE']

def extract_model_data(df_raw, model_name, models_dict, lang_row):
    """Extract data for a specific model"""
    start, end = models_dict[model_name]

    # Get language positions within this model's columns
    lang_positions = {}
    for i in range(start, end):
        lang = lang_row[i]
        if pd.notna(lang) and lang != '':
            lang_positions[lang] = i

    print(f"\n{model_name} languages: {list(lang_positions.keys())}")

    # Extract data rows (skip header rows 0 and 1)
    data_rows = df_raw.iloc[2:].copy()

    # Get competition names (column 1)
    comp_names = data_rows.iloc[:, 1].tolist()

    # Build result dataframe
    result = {'comp_name': comp_names}
    for lang in languages_order:
        if lang in lang_positions:
            col_idx = lang_positions[lang]
            result[lang] = data_rows.iloc[:, col_idx].tolist()
        else:
            result[lang] = [None] * len(comp_names)

    return pd.DataFrame(result)


# Process each model
all_models_data = {}
for model_name in models.keys():
    df_model = extract_model_data(df_raw, model_name, models, lang_row)
    df_model = df_model.set_index('comp_name')
    all_models_data[model_name] = df_model
    print(f"\n{model_name} data shape: {df_model.shape}")


def create_pass_rate_matrices(df_model):
    """Calculate pass rate and fraction matrices"""
    df_pass_rate = df_model.map(calculate_pass_rate)
    df_pass_fraction = df_model.map(calculate_pass_fraction)

    # Rename columns to short form
    rename_map = dict(zip(languages_order, lang_display))
    df_pass_rate.columns = [rename_map.get(c, c) for c in df_pass_rate.columns]
    df_pass_fraction.columns = [rename_map.get(c, c) for c in df_pass_fraction.columns]

    return df_pass_rate, df_pass_fraction


# Calculate for each model
pass_rate_data = {}
pass_frac_data = {}

for model_name, df_model in all_models_data.items():
    df_rate, df_frac = create_pass_rate_matrices(df_model)
    pass_rate_data[model_name] = df_rate
    pass_frac_data[model_name] = df_frac

    # Count valid entries (not NaN)
    valid_count = df_rate.notna().sum().sum()
    total_count = df_rate.size
    print(f"\n{model_name}: {valid_count}/{total_count} valid entries (with 3+ runs)")


# === VISUALIZATION FUNCTIONS ===

def create_pass_rate_heatmap(df_rate, df_frac, output_path, title):
    """Create a heatmap showing pass rates with fraction annotations"""

    # Filter out rows with all NaN
    valid_rows = df_rate.dropna(how='all').index
    df_rate_filtered = df_rate.loc[valid_rows]
    df_frac_filtered = df_frac.loc[valid_rows]

    if len(df_rate_filtered) == 0:
        print(f"No valid data for {title}, skipping...")
        return None

    fig, ax = plt.subplots(figsize=(12, max(6, len(df_rate_filtered) * 0.5)))

    # Custom colormap: red -> yellow -> green
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    cmap.set_bad(color='lightgray')

    # Create heatmap
    sns.heatmap(
        df_rate_filtered,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=1,
        center=0.67,
        annot=df_frac_filtered.values,
        fmt='',
        annot_kws={'size': 9, 'fontweight': 'bold'},
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Pass Rate', 'shrink': 0.8}
    )

    ax.set_title(title, fontweight='bold', pad=15, fontsize=14)
    ax.set_xlabel('Language', fontweight='bold')
    ax.set_ylabel('Competition', fontweight='bold')

    # Add language group separators (high=5, medium=6, low=3)
    ax.axvline(x=5, color='black', linewidth=2)   # After high-resource
    ax.axvline(x=11, color='black', linewidth=2)   # After medium-resource

    # Add group labels
    ax.text(2.5, -0.8, 'High-resource', ha='center', fontsize=9, fontweight='bold')
    ax.text(8, -0.8, 'Medium-resource', ha='center', fontsize=9, fontweight='bold')
    ax.text(12.5, -0.8, 'Low-resource', ha='center', fontsize=9, fontweight='bold')

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.pdf', '.png'))
    print(f"Saved: {output_path}")
    return fig


def create_gap_vs_english_heatmap(df_rate, output_path):
    """Create heatmap of performance gap relative to English (Δ vs EN)."""
    if 'EN' not in df_rate.columns:
        print(f"Skipping gap heatmap (no EN column): {output_path}")
        return None

    valid_rows = df_rate.dropna(how='all').index
    df_rate_filtered = df_rate.loc[valid_rows]
    if len(df_rate_filtered) == 0:
        print(f"No valid data for gap heatmap, skipping: {output_path}")
        return None

    # Gap = pass_rate_lang - pass_rate_EN (per row)
    df_gap = df_rate_filtered.subtract(df_rate_filtered['EN'], axis=0)
    # Drop EN column so it's all zeros and doesn't clutter
    df_gap = df_gap.drop(columns=['EN'], errors='ignore')
    if df_gap.empty:
        return None

    # Annotations: format as +0.12 / -0.15
    annot_gap = df_gap.apply(lambda s: s.map(lambda x: f'{x:+.2f}' if pd.notna(x) else ''))

    fig, ax = plt.subplots(figsize=(12, max(6, len(df_gap) * 0.5)))
    cmap = sns.diverging_palette(10, 240, as_cmap=True)
    cmap.set_bad(color='lightgray')

    sns.heatmap(
        df_gap,
        ax=ax,
        cmap=cmap,
        vmin=-0.5,
        vmax=0.2,
        center=0,
        annot=annot_gap.values,
        fmt='',
        annot_kws={'size': 8, 'fontweight': 'bold'},
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Δ vs English', 'shrink': 0.8}
    )

    ax.set_xlabel('Language', fontweight='bold')
    ax.set_ylabel('Competition', fontweight='bold')

    # Language group separators (EN removed, so 4 high left → x=4, then 6 med → x=10)
    ax.axvline(x=4, color='black', linewidth=2)
    ax.axvline(x=10, color='black', linewidth=2)
    ax.text(2, -0.8, 'High-resource', ha='center', fontsize=9, fontweight='bold')
    ax.text(7, -0.8, 'Medium-resource', ha='center', fontsize=9, fontweight='bold')
    ax.text(11.5, -0.8, 'Low-resource', ha='center', fontsize=9, fontweight='bold')

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.pdf', '.png'))
    print(f"Saved: {output_path}")
    return fig


def create_pass_rate_bar(df_rate, output_path, title):
    """Bar chart showing average pass rate per language"""

    # Filter valid data
    df_valid = df_rate.dropna(how='all')
    if len(df_valid) == 0:
        print(f"No valid data for bar chart, skipping...")
        return None

    # Calculate mean and std
    means = df_valid.mean()
    stds = df_valid.std()

    # Colors by language group
    high_resource = ['EN', 'FR', 'ES', 'IT', 'ZH']
    medium_resource = ['JA', 'RU', 'PL', 'TR', 'AR', 'UK']
    low_resource = ['RO', 'KK', 'BE']

    colors = []
    for lang in means.index:
        if lang in high_resource:
            colors.append('#2ecc71')
        elif lang in medium_resource:
            colors.append('#3498db')
        else:
            colors.append('#e74c3c')

    fig, ax = plt.subplots(figsize=(11, 5))

    x = np.arange(len(means))
    bars = ax.bar(x, means * 100, yerr=stds * 100, capsize=4,
                  color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, means)):
        if not np.isnan(val):
            height = bar.get_height()
            std_val = stds.iloc[i] if not np.isnan(stds.iloc[i]) else 0
            ax.text(bar.get_x() + bar.get_width()/2., height + std_val*100 + 2,
                    f'{val*100:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(means.index)
    ax.set_ylabel('Pass Rate (%)', fontweight='bold')
    ax.set_xlabel('Language', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=12)

    # Reference lines
    ax.axhline(y=100, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=66.7, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(len(means) - 0.5, 68, '2/3 threshold', fontsize=8, ha='right', color='gray')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='High-resource'),
        Patch(facecolor='#3498db', label='Medium-resource'),
        Patch(facecolor='#e74c3c', label='Low-resource'),
    ]
    ax.legend(handles=legend_elements, loc='lower left')

    ax.set_ylim(0, 115)
    ax.axvline(x=4.5, color='black', linewidth=1, linestyle='-', alpha=0.3)
    ax.axvline(x=10.5, color='black', linewidth=1, linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.pdf', '.png'))
    print(f"Saved: {output_path}")
    return fig


def _short_model_name(model_name):
    """Return display name for model."""
    return (model_name.replace('gpt-oss-120', 'GPT-oss-120')
            .replace('gemini-2.5-flash', 'Gemini-2.5')
            .replace('gpt-4.1-mini', 'GPT-4.1-mini')
            .replace('qwen2.5-coder', 'Qwen2.5-Coder')
            .replace('gpt-oss (ReAct)', 'GPT-oss (ReAct)'))


def create_models_collage(pass_rate_dict, pass_frac_dict, model_names, output_path, suptitle=None):
    """Create side-by-side heatmaps for a subset of models (same style as pass_rate_all_models)."""
    valid_models = [m for m in model_names if m in pass_rate_dict]
    if len(valid_models) == 0:
        print(f"No valid models found for collage: {model_names}, skipping...")
        return None

    n_models = len(valid_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 8), sharey=True)
    if n_models == 1:
        axes = [axes]

    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    cmap.set_bad(color='lightgray')

    for idx, model_name in enumerate(valid_models):
        df_rate = pass_rate_dict[model_name].dropna(how='all')
        df_frac = pass_frac_dict[model_name].reindex_like(df_rate).fillna('0/3')
        sns.heatmap(
            df_rate,
            ax=axes[idx],
            cmap=cmap,
            vmin=0,
            vmax=1,
            center=0.67,
            annot=df_frac.values,
            fmt='',
            annot_kws={'size': 7, 'fontweight': 'bold'},
            linewidths=0.3,
            linecolor='white',
            cbar=idx == n_models - 1,
            cbar_kws={'label': 'Pass Rate', 'shrink': 0.6} if idx == n_models - 1 else {}
        )
        axes[idx].set_title(_short_model_name(model_name), fontweight='bold', fontsize=11)
        axes[idx].set_xlabel('Language' if idx == n_models // 2 else '')
        if idx == 0:
            axes[idx].set_ylabel('Competition', fontweight='bold')
        axes[idx].axvline(x=5, color='black', linewidth=1.5)
        axes[idx].axvline(x=11, color='black', linewidth=1.5)

    if suptitle:
        plt.suptitle(suptitle, fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.pdf', '.png'))
    print(f"Saved: {output_path}")
    return fig


def create_combined_models_heatmap(pass_rate_dict, pass_frac_dict, output_path):
    """Create side-by-side heatmaps for all models"""

    # Get models with valid data
    valid_models = []
    for model_name, df_rate in pass_rate_dict.items():
        df_valid = df_rate.dropna(how='all')
        if len(df_valid) > 0:
            valid_models.append(model_name)

    if len(valid_models) == 0:
        print("No valid models found!")
        return None

    n_models = len(valid_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8), sharey=True)

    if n_models == 1:
        axes = [axes]

    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    cmap.set_bad(color='lightgray')

    for idx, model_name in enumerate(valid_models):
        df_rate = pass_rate_dict[model_name].dropna(how='all')
        df_frac = pass_frac_dict[model_name].reindex_like(df_rate).fillna('0/3')

        sns.heatmap(
            df_rate,
            ax=axes[idx],
            cmap=cmap,
            vmin=0,
            vmax=1,
            center=0.67,
            annot=df_frac.values,
            fmt='',
            annot_kws={'size': 7, 'fontweight': 'bold'},
            linewidths=0.3,
            linecolor='white',
            cbar=idx == n_models - 1,
            cbar_kws={'label': 'Pass Rate', 'shrink': 0.6} if idx == n_models - 1 else {}
        )

        axes[idx].set_title(_short_model_name(model_name), fontweight='bold', fontsize=11)
        axes[idx].set_xlabel('Language' if idx == n_models // 2 else '')

        if idx == 0:
            axes[idx].set_ylabel('Competition', fontweight='bold')

        # Add separators (high=5, medium=6, low=3)
        axes[idx].axvline(x=5, color='black', linewidth=1.5)
        axes[idx].axvline(x=11, color='black', linewidth=1.5)

    plt.suptitle('Pass Rate Across Models (15 competitions × 14 languages)',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.pdf', '.png'))
    print(f"Saved: {output_path}")
    return fig


def print_summary_stats(pass_rate_dict):
    """Print summary statistics for all models"""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    high = ['EN', 'FR', 'ES', 'IT', 'ZH']
    med = ['JA', 'RU', 'PL', 'TR', 'AR', 'UK']
    low = ['RO', 'KK', 'BE']

    for model_name, df_rate in pass_rate_dict.items():
        df_valid = df_rate.dropna(how='all')
        if len(df_valid) == 0:
            continue

        print(f"\n{model_name}:")
        print(f"  Valid competitions: {len(df_valid)}")
        print(f"  Overall pass rate: {df_valid.mean().mean()*100:.1f}%")

        high_cols = [c for c in df_valid.columns if c in high]
        med_cols = [c for c in df_valid.columns if c in med]
        low_cols = [c for c in df_valid.columns if c in low]

        if high_cols:
            print(f"  High-resource: {df_valid[high_cols].mean().mean()*100:.1f}%")
        if med_cols:
            print(f"  Medium-resource: {df_valid[med_cols].mean().mean()*100:.1f}%")
        if low_cols:
            print(f"  Low-resource: {df_valid[low_cols].mean().mean()*100:.1f}%")

        print(f"  Per language:")
        for col in df_valid.columns:
            mean_val = df_valid[col].mean()
            if not np.isnan(mean_val):
                print(f"    {col}: {mean_val*100:.1f}%")


# === GENERATE VISUALIZATIONS ===
print("\n" + "=" * 60)
print("Generating Visualizations")
print("=" * 60)

# Generate for each model
for model_name in pass_rate_data.keys():
    safe_name = model_name.replace('.', '_').replace('-', '_')

    create_pass_rate_heatmap(
        pass_rate_data[model_name],
        pass_frac_data[model_name],
        f'content/pass_rate_{safe_name}.pdf',
        f'Pass Rate: {model_name}'
    )

    create_pass_rate_bar(
        pass_rate_data[model_name],
        f'content/pass_rate_bar_{safe_name}.pdf',
        f'Average Pass Rate by Language: {model_name}'
    )

    create_gap_vs_english_heatmap(
        pass_rate_data[model_name],
        f'content/pass_rate_gap_english_{safe_name}.pdf'
    )

# Combined view (all models)
create_combined_models_heatmap(
    pass_rate_data,
    pass_frac_data,
    'content/pass_rate_all_models.pdf'
)

# Collage 1: gpt-oss-120 + gpt-oss (ReAct)
create_models_collage(
    pass_rate_data,
    pass_frac_data,
    ['gpt-oss-120', 'gpt-oss (ReAct)'],
    'content/pass_rate_collage_1.pdf',
    suptitle='Pass Rate: GPT-oss-120 & GPT-oss (ReAct) (15 competitions × 14 languages)'
)

# Collage 2: gemini-2.5-flash + gpt-4.1-mini
create_models_collage(
    pass_rate_data,
    pass_frac_data,
    ['gemini-2.5-flash', 'gpt-4.1-mini'],
    'content/pass_rate_collage_2.pdf',
    suptitle='Pass Rate: Gemini-2.5 & GPT-4.1-mini (15 competitions × 14 languages)'
)

# Print stats
print_summary_stats(pass_rate_data)

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)