"""Load bundled structured financial samples into data/financial.db."""

from pathlib import Path

from src.ingestion.financial_store import import_research_reports_index

DEFAULT_REPORT_INDEX = Path(
    "data/raw/real_securities_data/financials/research_reports_index.csv"
)


def main() -> int:
    count = import_research_reports_index(DEFAULT_REPORT_INDEX)
    print(f"research_reports_index: {count} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
