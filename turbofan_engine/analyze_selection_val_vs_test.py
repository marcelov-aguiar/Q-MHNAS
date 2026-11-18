"""
Validation-based vs. test-based architecture selection (turbofan / C-MAPSS).

Question answered
-----------------
The architecture reported in the results table was selected by the LOWEST TEST
RMSE. Would the same architecture have been selected using only VALIDATION data?
If not, is the test-RMSE difference between the two architectures significant?

Experimental structure
----------------------
One config file (e.g. ``config_turbofan_FD002_v23.txt``) defines one experimental
configuration, identified by ``id = "<dataset>/<exp>"`` (e.g. ``FD002/exp_v23``).
Each configuration ran ``repeat`` independent searches; each search produced one
architecture, which was retrained N times with different seeds. For each retrain:

    best_rmse -> lowest validation RMSE over the epochs (the saved checkpoint)
    test_rmse -> test RMSE of that same checkpoint

Selection criteria (which search/architecture is "the best")
------------------------------------------------------------
    1) teste      : search containing the lowest individual test_rmse (table value)
    2) val_media  : search with the lowest MEAN best_rmse over its retrains
    3) val_melhor : search with the lowest individual best_rmse

Validation is only used to CHOOSE a search. When criterion 2 or 3 picks a search
different from criterion 1, both searches are compared on the TEST set using the
mean test RMSE of their retrains (Welch's t-test, see ``welch_comparison``).

Outputs (in ``turbofan_engine/selection_analysis/``)
----------------------------------------------------
    buscas.csv             : one row per search (mean, std, best of validation and test)
    resumo_val_media.csv   : one row per experiment, criterion 2 vs criterion 1
    resumo_val_melhor.csv  : one row per experiment, criterion 3 vs criterion 1

Usage (from the repository root)
--------------------------------
    python turbofan_engine/analyze_selection_val_vs_test.py
    python turbofan_engine/analyze_selection_val_vs_test.py config_turbofan_FD002_v23.txt config_turbofan_FD003_v27.txt
"""
import re
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util import load_yaml  # noqa: E402

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "selection_analysis"

# Default configs analysed when no config name is passed on the command line
# (experiments reported in the dissertation table).
CONFIG_FILES = [
    # SCH (shared head architecture)
    "config_turbofan_FD001_v29.txt",
    "config_turbofan_FD002_v17.txt",
    "config_turbofan_FD003_v12.txt",
    "config_turbofan_FD004_v13.txt",
    # ICH (independent head architecture)
    "config_turbofan_FD001_v54.txt",
    "config_turbofan_FD002_v23.txt",
    "config_turbofan_FD003_v27.txt",
    "config_turbofan_FD004_v35.txt",

    # Table "tab:amostra_scheduler_completa" (TFScheduler impact): without vs with scheduler.
    # Pairs already listed above are kept as comments to avoid duplicates.
    # FD001
    "config_turbofan_FD001_v5.txt",  # vs v29 (above)
    "config_turbofan_FD001_v8.txt", "config_turbofan_FD001_v32.txt",
    "config_turbofan_FD001_v10.txt", "config_turbofan_FD001_v34.txt",
    "config_turbofan_FD001_v59.txt", "config_turbofan_FD001_v60.txt",
    "config_turbofan_FD001_v17.txt", "config_turbofan_FD001_v41.txt",
    "config_turbofan_FD001_v19.txt", "config_turbofan_FD001_v43.txt",
    # FD002
    "config_turbofan_FD002_v7.txt",  # vs v17 (above)
    "config_turbofan_FD002_v5.txt", "config_turbofan_FD002_v15.txt",
    "config_turbofan_FD002_v30.txt", "config_turbofan_FD002_v31.txt",
    "config_turbofan_FD002_v10.txt", "config_turbofan_FD002_v20.txt",
    "config_turbofan_FD002_v4.txt", "config_turbofan_FD002_v14.txt",
    "config_turbofan_FD002_v6.txt", "config_turbofan_FD002_v16.txt",
    # FD003
    "config_turbofan_FD003_v2.txt",  # vs v12 (above)
    "config_turbofan_FD003_v6.txt", "config_turbofan_FD003_v16.txt",
    "config_turbofan_FD003_v29.txt", "config_turbofan_FD003_v30.txt",
    "config_turbofan_FD003_v9.txt", "config_turbofan_FD003_v19.txt",
    "config_turbofan_FD003_v4.txt", "config_turbofan_FD003_v14.txt",
    "config_turbofan_FD003_v5.txt", "config_turbofan_FD003_v15.txt",
    # FD004
    "config_turbofan_FD004_v2.txt",  # vs v13 (above)
    "config_turbofan_FD004_v6.txt", "config_turbofan_FD004_v17.txt",
    "config_turbofan_FD004_v7.txt", "config_turbofan_FD004_v18.txt",
    "config_turbofan_FD004_v11.txt", "config_turbofan_FD004_v22.txt",
    "config_turbofan_FD004_v4.txt", "config_turbofan_FD004_v15.txt",
    "config_turbofan_FD004_v5.txt", "config_turbofan_FD004_v16.txt",

    # Table "tab:scheduler_evolucao" (TFScheduler during evolution): without vs with scheduler.
    # Configs already listed above are kept as comments to avoid duplicates.
    # FD001
    "config_turbofan_FD001_v55.txt",  # v29 (above) vs v55
    "config_turbofan_FD001_v61.txt", "config_turbofan_FD001_v62.txt",
    "config_turbofan_FD001_v52.txt", "config_turbofan_FD001_v63.txt",
    # FD002
    "config_turbofan_FD002_v24.txt",  # v17 (above) vs v24
    "config_turbofan_FD002_v19.txt", "config_turbofan_FD002_v28.txt",
    "config_turbofan_FD002_v21.txt", "config_turbofan_FD002_v29.txt",
    # FD003
    "config_turbofan_FD003_v23.txt",  # v12 (above) vs v23
    "config_turbofan_FD003_v20.txt", "config_turbofan_FD003_v26.txt",
    "config_turbofan_FD003_v18.txt",  # v18 vs v27 (above)
    # FD004
    "config_turbofan_FD004_v25.txt",  # v13 (above) vs v25
    "config_turbofan_FD004_v21.txt", "config_turbofan_FD004_v32.txt",
    "config_turbofan_FD004_v23.txt", "config_turbofan_FD004_v33.txt",

    # Table "tab:comparacao_batch" (batch size): batch 64 vs batch 400.
    # Configs already listed above are kept as comments to avoid duplicates.
    # FD001
    "config_turbofan_FD001_v57.txt",  # v57 vs v29 (above)
    # FD002
    "config_turbofan_FD002_v3.txt", "config_turbofan_FD002_v2.txt",
    "config_turbofan_FD002_v13.txt",  # v13 vs v17 (above)
    # FD003
    "config_turbofan_FD003_v13.txt",  # v13 vs v12 (above)
    # FD004
    "config_turbofan_FD004_v14.txt",  # v14 vs v13 (above)
    "config_turbofan_FD004_v31.txt",  # v16 (above) vs v31

    # Table SCH vs ICH (same hyperparameters). Only configs not listed above.
    "config_turbofan_FD001_v46.txt",  # SCH v29 | ICH v46
    "config_turbofan_FD001_v47.txt",  # SCH v32 | ICH v47
    "config_turbofan_FD001_v48.txt",  # SCH v34 | ICH v48
    "config_turbofan_FD001_v49.txt", "config_turbofan_FD001_v51.txt",  # SCH v49 | ICH v51
    "config_turbofan_FD001_v50.txt",  # SCH v50 | ICH v52
    "config_turbofan_FD002_v18.txt",  # SCH v18 | ICH v19
    "config_turbofan_FD003_v21.txt",  # SCH v21 | ICH v18
    "config_turbofan_FD004_v20.txt",  # SCH v20 | ICH v21

    # Table ICH: without vs with retrain scheduler. Only configs not listed above.
    "config_turbofan_FD001_v66.txt", "config_turbofan_FD001_v64.txt",  # v66 vs v64
    "config_turbofan_FD002_v22.txt",  # v28 vs v22
    "config_turbofan_FD003_v28.txt",  # v27 vs v28
    "config_turbofan_FD003_v22.txt",  # v26 vs v22
    "config_turbofan_FD004_v34.txt",  # v32 vs v34
    "config_turbofan_FD004_v24.txt",  # v33 vs v24

    # Table population size (SCH). Only configs not listed above.
    "config_turbofan_FD001_v56.txt",  # v56 vs v49
    "config_turbofan_FD001_v31.txt", "config_turbofan_FD001_v33.txt",  # v31 vs v33
    "config_turbofan_FD002_v27.txt",  # v27 vs v18
    "config_turbofan_FD003_v25.txt",  # v25 vs v21
    "config_turbofan_FD004_v30.txt",  # v30 vs v20
]

NUM_RETRAINS = 5  # reads retrain_results_F12_{NUM_RETRAINS}_<scheduler>.txt (holds all retrains)
ALPHA = 0.05      # significance level of Welch's t-test


# ==========================================================
# ==================== DATA LOADING ========================
# ==========================================================

def load_experiment(config_name: str):
    """
    Load the validation and test RMSE of every retrain of one experiment.

    Parameters
    ----------
    config_name : str
        Config file name, e.g. ``"config_turbofan_FD002_v23.txt"``. The dataset is
        parsed from the name; ``exp``, ``lr_scheduler``, ``repeat`` and the head
        architecture are read from the file itself.

    Returns
    -------
    exp_id : str
        Unique experiment identifier ``"<dataset>/<exp>"``, e.g. ``"FD002/exp_v23"``.
        The dataset is needed because the same ``exp_vXX`` exists in several datasets.
    strategy : str
        ``"SCH"`` (shared head architecture) or ``"ICH"`` (independent heads).
    df : pandas.DataFrame
        One row per retrain with columns ``busca`` (search number), ``retreino``,
        ``val`` (``best_rmse``) and ``teste`` (``test_rmse``).
    """
    match = re.match(r"config_turbofan_(FD\d{3})_v\d+\.txt$", config_name)
    if match is None:
        raise ValueError(f"Unexpected config name: {config_name}")
    dataset = match.group(1)

    config = load_yaml(BASE / dataset / "config_files" / config_name)
    exp_name = config["train"]["exp"]
    lr_scheduler = config["train"]["lr_scheduler"]
    num_repeats = int(config["train"]["repeat"])
    shared = config["train"].get("extra_params", {}).get("shared_head_architecture")
    strategy = {True: "SCH", False: "ICH"}.get(shared, "?")

    rows = []
    for search in range(1, num_repeats + 1):
        f = (BASE / dataset / f"exp_{dataset}" / f"{exp_name}_repeat_{search}"
             / f"retrain_results_F12_{NUM_RETRAINS}_{lr_scheduler}.txt")
        if not f.exists():
            print(f"⚠️  file not found: {f}")
            continue
        for retrain_id, v in json.load(open(f)).items():
            rows.append({"busca": search, "retreino": retrain_id,
                         "val": float(v["best_rmse"]), "teste": float(v["test_rmse"])})
    return f"{dataset}/{exp_name}", strategy, pd.DataFrame(rows)


# ==========================================================
# ================= STATISTICAL FUNCTIONS ==================
# ==========================================================

def summarize_searches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Describe each search (architecture) by the distribution of its retrains.

    Each search was retrained with N seeds, so each architecture has N validation
    and N test RMSE values. They are summarised by:

    - mean : expected performance of the architecture (robust to a lucky seed);
    - std  : sample standard deviation (``ddof=1``), i.e. seed-to-seed variability;
    - best : minimum value, i.e. the luckiest seed.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of ``load_experiment`` (one row per retrain).

    Returns
    -------
    pandas.DataFrame
        One row per search with columns ``busca``, ``n``, ``val_media``,
        ``val_desvio``, ``val_melhor``, ``teste_media``, ``teste_desvio``,
        ``teste_melhor``.
    """
    g = df.groupby("busca")
    return pd.DataFrame({
        "n": g.size(),
        "val_media": g["val"].mean(), "val_desvio": g["val"].std(ddof=1), "val_melhor": g["val"].min(),
        "teste_media": g["teste"].mean(), "teste_desvio": g["teste"].std(ddof=1), "teste_melhor": g["teste"].min(),
    }).reset_index()


def select_searches(s: pd.DataFrame) -> dict:
    """
    Pick the "best" search according to each selection criterion.

    Parameters
    ----------
    s : pandas.DataFrame
        Output of ``summarize_searches``.

    Returns
    -------
    dict
        ``{"teste": int, "val_media": int, "val_melhor": int}`` with the chosen
        search number for each criterion:

        - ``teste``      : lowest individual test RMSE (the value in the table);
        - ``val_media``  : lowest mean validation RMSE;
        - ``val_melhor`` : lowest individual validation RMSE.
    """
    return {
        "teste": int(s.loc[s.teste_melhor.idxmin(), "busca"]),
        "val_media": int(s.loc[s.val_media.idxmin(), "busca"]),
        "val_melhor": int(s.loc[s.val_melhor.idxmin(), "busca"]),
    }


def welch_comparison(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Test whether two architectures differ in mean test RMSE beyond seed noise.

    Each mean comes from only a few seeds; retraining with other seeds would
    change it. The question is whether the observed difference is larger than
    what this seed noise alone would produce.

    Steps
    -----
    1. Difference of means::

           diff = mean(a) - mean(b)

    2. Standard error (SE) of the difference, i.e. how much ``diff`` would vary
       just by redrawing the seeds, if both architectures were equally good::

           SE = sqrt(s_a^2 / n_a + s_b^2 / n_b)

       where ``s`` is the sample std and ``n`` the number of retrains.
       ``s / sqrt(n)`` is the standard error of each mean; the two errors add up
       in quadrature because both means are noisy.

    3. Signal-to-noise ratio (Welch's t statistic)::

           t = diff / SE

       It counts how many "units of noise" the difference measures. If the two
       architectures were equal, ``|t|`` would rarely exceed ~2 (normal curve,
       95%). With few seeds the std itself is uncertain, so the t distribution
       (heavier tails) is used and the threshold is higher (~2.3 for n=5 each).

    4. p-value: probability of observing ``|t|`` at least this large if both
       architectures had the same mean test RMSE. ``p < ALPHA`` -> significant.

    Welch's version does not assume equal variances in ``a`` and ``b``.

    Parameters
    ----------
    a : numpy.ndarray
        Test RMSE of the retrains of the search chosen by validation.
    b : numpy.ndarray
        Test RMSE of the retrains of the search chosen by test.

    Returns
    -------
    dict
        ``diferenca`` (diff; > 0 means ``a`` is worse on average), ``EP`` (SE),
        ``diferenca_sobre_EP`` (t), ``p_valor_welch`` (two-sided p-value) and
        ``significativa`` (``p < ALPHA``).

    Notes
    -----
    "Not significant" means the difference cannot be distinguished from seed
    noise; it does NOT prove the architectures are equal. With ~5 seeds per
    architecture the test only detects large differences (low statistical power).

    Examples
    --------
    FD003/exp_v27 (validation picks search 1, test picks search 2):

    >>> a = np.array([12.45, 11.70, 12.03, 12.87, 12.45])  # mean 12.30, std 0.45
    >>> b = np.array([11.64, 11.34, 11.60, 12.15, 12.20])  # mean 11.79, std 0.38
    >>> r = welch_comparison(a, b)  # diff=+0.52, SE=0.26, t=1.97, p=0.09 -> not significant
    """
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    p = stats.ttest_ind(a, b, equal_var=False).pvalue
    return {"diferenca": diff, "EP": se, "diferenca_sobre_EP": diff / se,
            "p_valor_welch": p, "significativa": p < ALPHA}


def compare_with_test_choice(df: pd.DataFrame, s: pd.DataFrame, val_arch: int, test_arch: int) -> dict:
    """
    Build one summary row: search chosen by validation vs. search chosen by test.

    If the experiment has a single search there is no choice to make, so it is
    flagged as not comparable (``mesma_busca`` is NaN) and must not be counted as
    an agreement between criteria. If both criteria pick the same search there is
    nothing to test. Otherwise the two searches are compared on the TEST set with
    ``welch_comparison``.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of ``load_experiment`` (one row per retrain).
    s : pandas.DataFrame
        Output of ``summarize_searches``.
    val_arch : int
        Search chosen by a validation criterion (``busca_val``).
    test_arch : int
        Search chosen by the lowest test RMSE (``busca_teste``).

    Returns
    -------
    dict
        Number of searches, search numbers, mean/std test RMSE of both searches,
        the outputs of ``welch_comparison`` (NaN when there is nothing to test)
        and a readable ``conclusao``.
    """
    v = s[s.busca == val_arch].iloc[0]
    t = s[s.busca == test_arch].iloc[0]
    n_searches = len(s)
    row = {
        "n_buscas": n_searches,
        "busca_teste": test_arch,
        "busca_val": val_arch,
        "mesma_busca": (val_arch == test_arch) if n_searches > 1 else np.nan,
        "rmse_teste_media_busca_teste": t.teste_media,
        "rmse_teste_desvio_busca_teste": t.teste_desvio,
        "rmse_teste_media_busca_val": v.teste_media,
        "rmse_teste_desvio_busca_val": v.teste_desvio,
    }
    no_test = {"diferenca": np.nan, "EP": np.nan, "diferenca_sobre_EP": np.nan,
               "p_valor_welch": np.nan, "significativa": np.nan}
    if n_searches == 1:
        return {**row, **no_test, "conclusao": "busca única: sem comparação"}
    if val_arch == test_arch:
        return {**row, **no_test, "conclusao": "mesma busca: sem diferença"}

    w = welch_comparison(df.loc[df.busca == val_arch, "teste"].to_numpy(),
                         df.loc[df.busca == test_arch, "teste"].to_numpy())
    conclusao = (f"busca diferente: diferença {'' if w['significativa'] else 'NÃO '}significativa "
                 f"(dif={w['diferenca']:+.2f}, p={w['p_valor_welch']:.2f})")
    return {**row, **w, "conclusao": conclusao}


# ==========================================================
# ========================= MAIN ===========================
# ==========================================================

def main(config_files):
    OUT_DIR.mkdir(exist_ok=True)
    all_searches = []
    summaries = {"val_media": [], "val_melhor": []}

    for config_name in config_files:
        exp_id, strategy, df = load_experiment(config_name)
        if df.empty:
            print(f"⚠️  no retrain data for {config_name}")
            continue
        s = summarize_searches(df)
        choice = select_searches(s)

        s.insert(0, "estrategia", strategy)
        s.insert(0, "id", exp_id)
        for crit, arch in choice.items():
            s[f"escolhida_{crit}"] = s.busca == arch
        all_searches.append(s)

        for crit in summaries:
            summaries[crit].append({
                "id": exp_id, "estrategia": strategy,
                "rmse_teste_melhor": float(s.teste_melhor.min()),
                **compare_with_test_choice(df, s, choice[crit], choice["teste"]),
            })

    pd.concat(all_searches, ignore_index=True).to_csv(OUT_DIR / "buscas.csv", index=False, float_format="%.4f")

    with pd.option_context("display.width", 250, "display.max_columns", None):
        for crit, rows in summaries.items():
            d = pd.DataFrame(rows)
            d.to_csv(OUT_DIR / f"resumo_{crit}.csv", index=False, float_format="%.4f")
            comparable = d[d.n_buscas > 1]  # single-search experiments have no choice to compare
            print(f"\n=== resumo_{crit}.csv — mesma busca em {int(comparable.mesma_busca.sum())}/{len(comparable)} "
                  f"experimentos com mais de uma busca ({len(d) - len(comparable)} com busca única excluídos) ===")
            print(d[["id", "estrategia", "busca_teste", "busca_val", "conclusao"]].to_string(index=False))
    print(f"\n📄 CSVs salvos em {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation-based vs. test-based architecture selection.")
    parser.add_argument("configs", nargs="*", default=CONFIG_FILES,
                        help="config file names, e.g. config_turbofan_FD002_v23.txt (default: CONFIG_FILES)")
    main(parser.parse_args().configs)
