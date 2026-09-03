from __future__ import annotations

import argparse
import getpass

from config import StudyConfig
from data_loader import authenticate_finlab, load_finlab_data
from pipeline import run_study
from phase2_pipeline import run_phase2_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="三大法人現貨買賣金額對0050未來報酬研究"
    )
    parser.add_argument(
        "--browser-login",
        action="store_true",
        help="使用 FinLab 瀏覽器登入；預設安全輸入 API Token",
    )
    parser.add_argument(
        "--phase2",
        action="store_true",
        help="執行精簡 Phase 2 機制驗證，不重跑舊版完整 grid",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[1/8] 下載 FinLab 資料")
    if args.browser_login:
        authenticate_finlab()
    else:
        token = getpass.getpass("請輸入 FinLab API Token：")
        authenticate_finlab(token)
    raw = load_finlab_data(ticker="0050")
    if args.phase2:
        output = run_phase2_study(
            raw, StudyConfig(study_mode="phase2_flow_mechanism")
        )
    else:
        output = run_study(raw, StudyConfig())
    print(f"完成：{output.resolve()}")


if __name__ == "__main__":
    main()
