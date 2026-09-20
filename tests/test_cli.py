from reflexroute.cli import main


def test_cli_accepts_models_before_quoted_query(capsys):
    # One candidate makes this a fully local CLI smoke test.
    code = main(["route", "--models", "private/model", "hello world"])
    output = capsys.readouterr()
    assert code == 0
    assert "Selected: private/model" in output.out
    assert "Confidence: 1.00" in output.out
