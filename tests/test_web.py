from demand_forecasting.web import DEFAULT_CASE_STUDY_URL, case_study_url


def test_case_study_url_defaults_to_local_portfolio(monkeypatch):
    monkeypatch.delenv("DEMO_CASE_STUDY_URL", raising=False)
    assert case_study_url() == DEFAULT_CASE_STUDY_URL


def test_case_study_url_accepts_https(monkeypatch):
    expected = "https://portfolio.example/demand-forecasting.html"
    monkeypatch.setenv("DEMO_CASE_STUDY_URL", expected)
    assert case_study_url() == expected


def test_case_study_url_rejects_unsafe_or_malformed_values(monkeypatch):
    for value in (
        "javascript:alert(1)",
        "https://user:password@portfolio.example/case",
        "https://portfolio.example/case\r\nX-Test: injected",
        "not-a-url",
    ):
        monkeypatch.setenv("DEMO_CASE_STUDY_URL", value)
        assert case_study_url() == DEFAULT_CASE_STUDY_URL
