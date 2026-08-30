from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd


def timestamped_output_directory(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(root) / stamp
    (output / "figures").mkdir(parents=True, exist_ok=False)
    return output


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_manifest(config: dict, metadata: dict) -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "package_versions": {
            name: _package_version(name)
            for name in [
                "finlab",
                "pandas",
                "numpy",
                "scipy",
                "statsmodels",
                "openpyxl",
            ]
        },
        "git_commit_hash": _git_commit(),
        "study_version": "0.1.0",
        "config": config,
        **metadata,
    }


def write_json(path: Path, obj: dict) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def write_research_summary(
    path: Path,
    group_results: pd.DataFrame,
    data_quality_failures: int,
) -> None:
    significant = 0
    if "zero_p_value_significant_fdr_05" in group_results:
        significant = int(group_results["zero_p_value_significant_fdr_05"].sum())
    conclusion = "無法判定" if group_results.empty else "修改後再測"
    text = f"""# 三大法人現貨資金流研究摘要

## A. 目前假設

d0 盤後法人資金流強度可能與 d1 開盤後的 0050 未來報酬相關。

## B. 市場機制

法人可能因拆單、資訊優勢或避險需求，使買進、賣出及 Net flow 對後續報酬呈現不同關係。

## C. 本次分析

- 統計結果列數：{len(group_results)}
- FDR 後相對零報酬顯著列數：{significant}
- 資料公式驗證失敗筆數：{data_quality_failures}

## D. 主要結果

請以 `Main_Results` 與 HAC/FDR 欄位為準；本摘要不以單一 p-value 自動宣稱策略有效。

## E. 穩健性結果

需進一步檢查年度、子期間、市場狀態與極端值貢獻。

## F. 主要風險

多日 forward return 重疊、資料制度改變、全域分組前視偏誤，以及法人資金流可能只是同期動能的代理變數。

## G. 反對者觀點

即使統計顯著，效果也可能太小、集中在少數年份，或在 d1 開盤時已被價格充分反映。

## H. 結論

**{conclusion}**
"""
    path.write_text(text, encoding="utf-8")
