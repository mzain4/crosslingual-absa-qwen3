"""Model-free checks for the full-test inference path."""

from clabsa.inference import iter_shard


def test_qlora_uses_no_demonstrations(tmp_path):
    test_file = tmp_path / "data" / "laptop" / "en" / "test.txt"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "nice keyboard .####[['keyboard', 'KEYBOARD#GENERAL', 'positive']]\n",
        encoding="utf-8",
    )
    rows = list(iter_shard(tmp_path, "laptop", "en", "tasd", "qlora"))
    assert len(rows) == 1
    assert rows[0][0].sentence == "nice keyboard ."
    assert rows[0][1] == []
