from pathlib import Path

from inspect_swe.reliability.analysis import (
    CampaignAnalysisResult,
    CampaignResourceSummary,
    PhaseAnalysisSummary,
)
from inspect_swe.reliability.reporting import write_campaign_markdown_report


def test_campaign_report_includes_token_breakdown(tmp_path: Path) -> None:
    result = CampaignAnalysisResult(
        benchmark="gaia_level1",
        campaign_id="campaign_x",
        resources=CampaignResourceSummary(
            input_tokens=1000,
            output_tokens=200,
            cache_read_tokens=3000,
            cache_write_tokens=50,
            reasoning_tokens=75,
            total_tokens=4200,
        ),
        phase_summaries={
            "baseline": PhaseAnalysisSummary(
                phase="baseline",
                sidecar_path="records.jsonl",
                source="eval",
                total_records=10,
                sample_count=5,
                repeat_count=2,
                input_tokens=100,
                output_tokens=20,
                cache_read_tokens=300,
                cache_write_tokens=5,
                reasoning_tokens=7,
                total_tokens=420,
            )
        },
    )

    output_path = tmp_path / "report.md"
    write_campaign_markdown_report(result=result, output_path=output_path)
    content = output_path.read_text(encoding="utf-8")

    assert "- Input Tokens: 1,000" in content
    assert "- Cache Read Tokens: 3,000" in content
    assert "| Phase | Source | Records | Samples | Repeats | Accuracy | Started | Completed | Wall Time | Cost | In | Out | Cache Read | Cache Write | Reasoning | Total |" in content
    assert "| baseline | eval | 10 | 5 | 2 | n/a | `n/a` | `n/a` | n/a | n/a | 100 | 20 | 300 | 5 | 7 | 420 |" in content
