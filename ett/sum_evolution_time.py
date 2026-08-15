"""Soma o tempo total de evolução (QNAS) por variável de predição do ETT.

Varre recursivamente `ett/<serie>/<variavel>/**/log_QNAS.txt`, extrai a linha
"Total evolution time: X hours and Y minutes" de cada arquivo e agrega o
tempo (em horas) por variável (hufl, hull, lufl, ot, ...).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ETT_ROOT = Path(__file__).resolve().parent
OUTPUT_CSV = ETT_ROOT / "evolution_time_by_variable.csv"

TIME_PATTERN = re.compile(
    r"Total evolution time:\s*(\d+)\s*hours?\s*and\s*(\d+)\s*minutes?"
)


def find_log_files(root: Path):
    return sorted(root.rglob("log_QNAS.txt"))


def extract_total_hours(log_path: Path) -> float | None:
    text = log_path.read_text(errors="ignore")
    matches = TIME_PATTERN.findall(text)
    if not matches:
        return None
    hours, minutes = matches[-1]
    return int(hours) + int(minutes) / 60


def main():
    rows = []
    for log_path in find_log_files(ETT_ROOT):
        rel_parts = log_path.relative_to(ETT_ROOT).parts
        # esperado: <serie>/<variavel>/.../exp_v*_repeat_*/log_QNAS.txt
        if len(rel_parts) < 2:
            continue
        series, variable = rel_parts[0], rel_parts[1]

        total_hours = extract_total_hours(log_path)
        if total_hours is None:
            print(f"AVISO: sem 'Total evolution time' em {log_path}")
            continue

        rows.append(
            {
                "series": series,
                "variable": variable,
                "experiment": log_path.parent.name,
                "hours": total_hours,
                "log_path": str(log_path.relative_to(ETT_ROOT)),
            }
        )

    if not rows:
        print("Nenhum log_QNAS.txt com tempo de evolução encontrado.")
        return

    details_df = pd.DataFrame(rows)

    summary_df = (
        details_df.groupby(["series", "variable"], as_index=False)
        .agg(n_experiments=("hours", "count"), total_hours=("hours", "sum"))
        .sort_values(["series", "variable"])
        .reset_index(drop=True)
    )
    summary_df["total_days"] = summary_df["total_hours"] / 24

    summary_df.to_csv(OUTPUT_CSV, index=False)

    print(summary_df.to_string(index=False))
    print(f"\nCSV salvo em: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
