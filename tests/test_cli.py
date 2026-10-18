from research_agent.cli import main


def test_demo_mode_runs_offline(capsys):
    code = main(["--demo"])
    out = capsys.readouterr().out
    assert code == 0
    assert "1967.15" in out
    assert "citations verified: 0 of 1" in out
    assert "made-up-source" in out


def test_demo_writes_report_and_trace(tmp_path, capsys):
    report = tmp_path / "report.md"
    trace = tmp_path / "trace.jsonl"
    assert main(["--demo", "--out", str(report), "--trace", str(trace)]) == 0
    assert report.read_text(encoding="utf-8").startswith("# ")
    assert len(trace.read_text().splitlines()) >= 5


def test_missing_question_exits_with_usage(capsys):
    assert main([]) == 2


def test_missing_api_key_is_a_clean_error(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["hello"]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
