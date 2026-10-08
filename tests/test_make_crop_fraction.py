import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common" / "analysis"))
from make_crop_fraction import build


def test_fraction_stratified_and_deterministic(tmp_path):
    src = tmp_path / "src"
    for c, n in (("A", 20), ("B", 8)):
        (src / "train" / c).mkdir(parents=True)
        for i in range(n):
            (src / "train" / c / f"{i}.jpg").write_bytes(b"x")
    (src / "val" / "A").mkdir(parents=True)
    r1 = build(src, tmp_path / "d1", 0.25, 0)
    r2 = build(src, tmp_path / "d2", 0.25, 0)
    assert r1 == {"A": 5, "B": 2}
    assert sorted(p.name for p in (tmp_path / "d1/train/A").iterdir()) == sorted(p.name for p in (tmp_path / "d2/train/A").iterdir())
    assert not any(p.is_symlink() for p in (tmp_path / "d1").rglob("*"))
    assert (tmp_path / "d1" / "val" / "A").is_dir()
