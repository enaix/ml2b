import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import seaborn as sns

HIGHER_IS_BETTER_METRICS = ["roc_auc", "f1_score", "accuracy", "r2_score", "FScoreMacro"]
LOWER_IS_BETTER_METRICS = ["rmse", "wmae", "log_loss"]

df_raw = pd.read_csv('final_version - chosen (final).csv', header=None)

languages = df_raw.iloc[1, 5:19].tolist()
models = []
all_languages = []

model_names = ['gpt-oss-120', 'gemini-2.5-flash', 'gpt-4.1-mini', 'gpt-oss (ReAct)']
for i in range(4):
    for j in range(14):
        models.append(model_names[i])
        all_languages.append(languages[j])

multi_index = pd.MultiIndex.from_arrays([models, all_languages], names=['model', 'language'])
data = df_raw.iloc[2:, 5:61].reset_index(drop=True)
df = pd.DataFrame(data.values, columns=multi_index)
df.index = df_raw.iloc[2:, 1].values
df.index.name = 'comp_name'
df['№'] = df_raw.iloc[2:, 0].values

# Add baseline_score from column 2
baseline_values = df_raw.iloc[2:, 2].values
df['baseline_score'] = pd.to_numeric(baseline_values, errors='coerce')

metric_values = df_raw.iloc[2:, 4].values
df['metric'] = metric_values


def clean_value(val):
    scores = []
    if pd.isna(val) or str(val).strip() in ['None', 'nan', 'NaN', '']:
        return []
    try:
        # If value contains multiple numbers separated by comma
        if ',' in str(val):
            values = str(val).split(',')
            for value in values:
                if pd.isna(value) or str(value).strip() in ['None', 'nan', 'NaN', '']:
                    continue
                else:
                    scores.append(value)
            return scores
        else:
            return [float(val)]
    except:
        return []


for col in df.columns:
    # When df.columns is a MultiIndex, added scalar columns become tuples like ('metric', '')
    # Clean only the (model, language) columns; leave metadata columns untouched.
    if isinstance(col, tuple):
        if col[0] in {'№', 'metric', 'baseline_score'}:
            continue
        df[col] = df[col].apply(clean_value)
    else:
        if col in {'№', 'metric', 'baseline_score'}:
            continue
        df[col] = df[col].apply(clean_value)

base_models = ['gpt-4.1-mini', 'gemini-2.5-flash', 'gpt-oss-120', 'gpt-oss (ReAct)']

result_tables = {}
metrics = ['AUP_std', 'AUP_ln', 'AUP_invln']


def count_valid_numeric_runs(value):
    """Count how many of the up-to-3 runs in a cell are valid numeric values (for pass rate)."""
    if not isinstance(value, list) or len(value) == 0:
        return 0
    count = 0
    for v in value:
        v_str = str(v).strip()
        if pd.isna(v) or v_str in ['None', 'nan', 'NaN', ''] or \
                'None' in v_str or 'nan' in v_str or 'NaN' in v_str or \
                'Error' in v_str or 'err' in v_str.lower():
            continue
        try:
            float(v)
            count += 1
        except (ValueError, TypeError):
            continue
    return count


def compute_rp(values, baseline_score, metric_name):
    if isinstance(baseline_score, pd.Series):
        baseline_score = baseline_score.iloc[0]

    if len(values) == 0:
        return (0.0, 0.0)

    rp_list = []
    for value in values:
        value_str = str(value).strip()

        # Skip invalid values
        if (pd.isna(value) if not isinstance(value, (list, np.ndarray)) else False) or \
                value_str in ['None', 'nan', 'NaN', ''] or \
                'None' in value_str or 'nan' in value_str or 'NaN' in value_str or \
                'Error' in value_str or 'err' in value_str.lower():
            continue

        # Try to convert to float, skip if conversion fails
        try:
            value = float(value)
        except (ValueError, TypeError):
            continue

        if metric_name.isin(HIGHER_IS_BETTER_METRICS).any():
            # For higher is better: rp = result / baseline
            rp = value / baseline_score if baseline_score != 0 else 0
        else:
            # For lower is better: rp = baseline / result
            rp = baseline_score / value if value != 0 else 0

        rp_list.append(rp)

    rp_list += [0] * (3 - len(rp_list))

    return (np.array(rp_list).mean(), np.array(rp_list).std())


def compute_rho(r_vals, tau):
    return np.mean(r_vals <= tau) if len(r_vals) > 0 else 0.0


def compute_AUP_std(r_vals):
    if len(r_vals) == 0:
        return 0.0
    taus = np.linspace(0, 1, 1001)
    rhos = [compute_rho(r_vals, tau) for tau in taus]
    return np.trapezoid(rhos, taus)


def compute_AUP_ln(r_vals):
    if len(r_vals) == 0:
        return 0.0

    def integrand(tau):
        if tau <= 1e-100:
            return 0.0
        return compute_rho(r_vals, tau) * (-np.log(tau))

    result, _ = integrate.quad(integrand, 0, 1, limit=500, epsabs=1e-6, epsrel=1e-6)
    return result


def compute_AUP_invln(r_vals, eps=1e-8):
    if len(r_vals) == 0:
        return 0.0

    def integrand(tau):
        return compute_rho(r_vals, tau) * (-1.0 / np.log(tau))

    def weight(tau):
        return -1.0 / np.log(tau)

    a, b = eps, 1 - eps
    n_points = 1000
    taus = np.linspace(a, b, n_points)

    numerator = np.trapezoid([integrand(tau) for tau in taus], taus)
    denominator = np.trapezoid([weight(tau) for tau in taus], taus)

    return numerator / denominator if denominator != 0 else 0.0


df_aup_std = df.copy()
df_aup_ln = df.copy()
df_aup_invln = df.copy()

for base_model in base_models:
    model_columns = [col for col in df.columns if col[0] == base_model]

    if not model_columns:
        continue

    model_data = df[model_columns]

    metrics_data = {}

    for lang in languages:
        lang_columns = [col for col in model_columns if col[1] == lang]

        r_values = []
        pass_rates_per_cell = []  # (successful runs / 3) * 100 per (comp, lang) cell
        for col in lang_columns:
            comp_vals = model_data[col].dropna().values

            for idx in model_data.index:
                value = model_data.loc[idx, col]
                baseline_score = df.loc[idx, 'baseline_score']
                metric = df.loc[idx, 'metric']

                # Pass rate: each cell has up to 3 runs; only numeric values count as successful
                n_valid = count_valid_numeric_runs(value)
                pass_rates_per_cell.append((n_valid / 3.0) * 100)

                rp, rp_std = compute_rp(value, baseline_score, metric)
                # print(rp_std)

                r_values.append(rp)

        r_values = np.array(r_values)
        pass_rate = np.mean(pass_rates_per_cell) if pass_rates_per_cell else 0.0

        aup_std = compute_AUP_std(r_values)
        aup_ln = compute_AUP_ln(r_values)
        aup_invln = compute_AUP_invln(r_values)
        win_rate = np.mean(r_values <= 1) * 100

        metrics_data[lang] = {
            'AUP_std': aup_std,
            'AUP_ln': aup_ln,
            'AUP_invln': aup_invln,
            'win_rate': win_rate,
            'pass_rate': pass_rate
        }

    avg_std = np.mean([metrics_data[lang]['AUP_std'] for lang in languages])
    avg_ln = np.mean([metrics_data[lang]['AUP_ln'] for lang in languages])
    avg_invln = np.mean([metrics_data[lang]['AUP_invln'] for lang in languages])
    avg_win_rate = np.mean([metrics_data[lang]['win_rate'] for lang in languages])
    avg_pass_rate = np.mean([metrics_data[lang]['pass_rate'] for lang in languages])

    metrics_data['Avg.'] = {
        'AUP_std': avg_std,
        'AUP_ln': avg_ln,
        'AUP_invln': avg_invln,
        'win_rate': avg_win_rate,
        'pass_rate': avg_pass_rate
    }

    result_tables[base_model] = metrics_data

# Display names in table vs keys from CSV
LANG_DISPLAY_TO_KEY = {'Arabic': 'Arab', 'Belarusian': 'Belarus'}

def format_table(model_name, metric):
    data = []
    languages_order = ['English', 'French', 'Spanish', 'Italian', 'Chinese', 'Japanese',
                       'Russian', 'Polish', 'Turkish', 'Arab', 'Ukranian', 'Romanian', 'Kazakh', 'Belarus', 'Avg.']

    for lang in languages_order:
        key = LANG_DISPLAY_TO_KEY.get(lang, lang)
        if key in result_tables[model_name]:
            rd = result_tables[model_name][key]
            if metric == 'AUP_std':
                value = rd['AUP_std']
                pr = rd['pass_rate']
                formatted = f"{value:.5f} / {pr:.0f}%"
            elif metric == 'AUP_ln':
                value = rd['AUP_ln']
                pr = rd['pass_rate']
                formatted = f"{value:.5f} / {pr:.0f}%"
            else:
                value = rd['AUP_invln']
                pr = rd['pass_rate']
                formatted = f"{value:.5f} / {pr:.0f}%"
            data.append([lang, formatted])

    return pd.DataFrame(data, columns=['Language', f'{model_name}'])


def format_table_numeric(model_name, metric):
    """Same as format_table but numeric AUP only (for heatmaps)."""
    data = []
    languages_order = ['English', 'French', 'Spanish', 'Italian', 'Chinese', 'Japanese',
                       'Russian', 'Polish', 'Turkish', 'Arab', 'Ukranian', 'Romanian', 'Kazakh', 'Belarus', 'Avg.']

    for lang in languages_order:
        key = LANG_DISPLAY_TO_KEY.get(lang, lang)
        if key in result_tables[model_name]:
            rd = result_tables[model_name][key]
            if metric == 'AUP_std':
                value = rd['AUP_std']
            elif metric == 'AUP_ln':
                value = rd['AUP_ln']
            else:
                value = rd['AUP_invln']
            data.append([lang, value])
        else:
            data.append([lang, np.nan])

    return pd.DataFrame(data, columns=['Language', f'{model_name}'])


std_table = pd.DataFrame({'Language': []})
ln_table = pd.DataFrame({'Language': []})
invln_table = pd.DataFrame({'Language': []})

for model in base_models:
    std_table = std_table.merge(format_table(model, 'AUP_std'), on='Language', how='outer')
    ln_table = ln_table.merge(format_table(model, 'AUP_ln'), on='Language', how='outer')
    invln_table = invln_table.merge(format_table(model, 'AUP_invln'), on='Language', how='outer')

std_table.fillna('', inplace=True)
ln_table.fillna('', inplace=True)
invln_table.fillna('', inplace=True)

# Print order: languages then models
print_languages_order = ['English', 'French', 'Spanish', 'Italian', 'Chinese', 'Japanese',
                        'Russian', 'Polish', 'Turkish', 'Arab', 'Ukranian', 'Romanian', 'Kazakh', 'Belarus', 'Avg.']
print_models_order = ['gpt-4.1-mini', 'gemini-2.5-flash', 'gpt-oss-120', 'gpt-oss (ReAct)']

def reorder_table_for_print(tbl):
    """Reorder rows by language order and columns by model order."""
    tbl = tbl.copy()
    tbl['_row_order'] = tbl['Language'].map({l: i for i, l in enumerate(print_languages_order)})
    tbl = tbl.sort_values('_row_order').drop(columns=['_row_order'])
    cols = ['Language'] + [c for c in print_models_order if c in tbl.columns]
    return tbl[cols]

std_table = reorder_table_for_print(std_table)
ln_table = reorder_table_for_print(ln_table)
invln_table = reorder_table_for_print(invln_table)

print("Table for metric AUP_std:")
print(std_table.to_string(index=False))
print("\nTable for metric AUP_ln:")
print(ln_table.to_string(index=False))
print("\nTable for metric AUP_invln:")
print(invln_table.to_string(index=False))


def invln_table_to_latex(tbl):
    """Print invln_table in LaTeX format (AUP_{-1/ln tau} / PR)."""
    lang_latex_name = {'Arab': 'Arabic', 'Belarus': 'Belarusian'}
    midrule_after_indices = {4, 10, 13}  # after Chinese, Ukranian, Belarus
    lines = [
        "& \\textbf{GPT-4.1-mini} & \\textbf{Gemini-2.5-Flash} & \\textbf{GPT-OSS-120b} & \\textbf{GPT-OSS (ReAct)} \\\\",
        "& AIDE & AIDE & AIDE & Our ReAct \\\\",
        "\\cmidrule(lr){2-5}",
        "& \\multicolumn{4}{c}{$\\mathrm{AUP}_{-1/\\ln\\tau}$ / PR} \\\\",
        "\\midrule",
    ]
    model_cols = [c for c in print_models_order if c in tbl.columns]
    for pos, (_, row) in enumerate(tbl.iterrows()):
        lang = row['Language']
        name = lang_latex_name.get(lang, lang)
        if lang == 'Avg.':
            name = "\\textbf{Avg.}"
        cells = []
        for c in model_cols:
            val = row[c]
            if isinstance(val, str) and ' / ' in val:
                aup_pr = val.replace('%', '').split(' / ')
                if len(aup_pr) == 2:
                    try:
                        aup = float(aup_pr[0].strip())
                        pr = int(float(aup_pr[1].strip()))
                        s = f"{aup:.5f}"
                        aup_str = s[1:] if s.startswith('0') and len(s) > 1 and s[1] == '.' else s
                        cells.append(f"{aup_str} / {pr}")
                    except (ValueError, TypeError):
                        cells.append(val.replace('%', ''))
                else:
                    cells.append(val.replace('%', ''))
            else:
                cells.append(str(val).replace('%', '') if isinstance(val, str) else str(val))
        lines.append(name + " & " + " & ".join(cells) + " \\\\")
        if pos in midrule_after_indices:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    return "\n".join(lines)


print("\n--- invln_table (LaTeX) ---")
print(invln_table_to_latex(invln_table))


lang_code_mapping = {
    'English': 'EN',
    'French': 'FR',
    'Spanish': 'ES',
    'Italian': 'IT',
    'Chinese': 'ZH',
    'Japanese': 'JA',
    'Russian': 'RU',
    'Polish': 'PL',
    'Turkish': 'TR',
    'Arab': 'AR',
    'Arabic': 'AR',  # Alias for compatibility
    'Romanian': 'RO',
    'Kazakh': 'KK',
    'Belarus': 'BE',
    'Belarusian': 'BE',  # Alias for compatibility
    'Ukranian': 'UK',
    'Ukrainian': 'UK',  # Alias for compatibility
    'Avg.': 'Avg',
    'Avg': 'Avg'  # Alias without period
}


def create_heatmap_style1(df, output_path='heatmap_style1.pdf', title='AUP_invln Average Performance'):
    """
    Style 1: Clean academic heatmap with annotations
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Custom colormap: white for NaN, red-yellow-green for scores
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    cmap.set_bad(color='lightgray')  # NaN values

    # Create heatmap
    sns.heatmap(
        df,
        ax=ax,
        cmap=cmap,
        vmin=0.3,
        vmax=1.0,
        center=0.65,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 7},
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Normalized Score', 'shrink': 0.8}
    )

    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel('Language', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')

    # Rotate x labels
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    # Add language group separators
    ax.axvline(x=4, color='black', linewidth=2)  # After high-resource
    ax.axvline(x=9, color='black', linewidth=2)  # After medium-resource

    # Add group labels
    ax.text(2, -0.8, 'High-resource', ha='center', fontsize=8, fontweight='bold')
    ax.text(6.5, -0.8, 'Medium', ha='center', fontsize=8, fontweight='bold')
    ax.text(11, -0.8, 'Low', ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.pdf', '.png'))
    print(f"Saved: {output_path}")
    return fig


def prepare_table_for_heatmap(table):
    """
    Prepare table for heatmap: set Language as index, convert to numeric, transpose, and map language names to codes
    """
    # Create a copy
    df_heatmap = table.copy()

    # Remove Avg row before processing
    df_heatmap = df_heatmap[~df_heatmap['Language'].isin(['Avg.', 'Avg'])]

    # Map language names to codes
    df_heatmap['Language'] = df_heatmap['Language'].map(lang_code_mapping).fillna(df_heatmap['Language'])

    # Define the order of languages (matching languages_order from format_table, excluding Avg)
    languages_order_codes = ['EN', 'FR', 'ES', 'IT', 'ZH', 'JA', 'RU', 'PL', 'TR', 'AR', 'UK', 'RO', 'KK', 'BE']

    # Set Language as index
    df_heatmap = df_heatmap.set_index('Language')

    # Reorder rows according to languages_order_codes (only keep languages that exist)
    existing_languages = [lang for lang in languages_order_codes if lang in df_heatmap.index]
    df_heatmap = df_heatmap.reindex(existing_languages)

    # Convert all columns to numeric, coercing errors to NaN
    for col in df_heatmap.columns:
        df_heatmap[col] = pd.to_numeric(df_heatmap[col], errors='coerce')

    # Transpose so models are rows and languages are columns
    df_heatmap = df_heatmap.T

    # Reorder columns to match the desired order
    existing_cols = [col for col in languages_order_codes if col in df_heatmap.columns]
    df_heatmap = df_heatmap[existing_cols]

    return df_heatmap


# Prepare numeric tables for heatmaps (display tables have "AUP / PR" strings)
std_table_numeric = pd.DataFrame({'Language': []})
ln_table_numeric = pd.DataFrame({'Language': []})
invln_table_numeric = pd.DataFrame({'Language': []})
for model in base_models:
    std_table_numeric = std_table_numeric.merge(format_table_numeric(model, 'AUP_std'), on='Language', how='outer')
    ln_table_numeric = ln_table_numeric.merge(format_table_numeric(model, 'AUP_ln'), on='Language', how='outer')
    invln_table_numeric = invln_table_numeric.merge(format_table_numeric(model, 'AUP_invln'), on='Language', how='outer')

std_table_heatmap = prepare_table_for_heatmap(std_table_numeric)
ln_table_heatmap = prepare_table_for_heatmap(ln_table_numeric)
invln_table_heatmap = prepare_table_for_heatmap(invln_table_numeric)

# Create heatmaps
create_heatmap_style1(std_table_heatmap, output_path='avg_baseline/heatmap_AUP_std.pdf', title='AUP_std Average Performance')
create_heatmap_style1(ln_table_heatmap, output_path='avg_baseline/heatmap_AUP_ln.pdf', title='AUP_ln Average Performance')
create_heatmap_style1(invln_table_heatmap, output_path='avg_baseline/heatmap_AUP_invln.pdf', title='AUP_invln Average Performance')