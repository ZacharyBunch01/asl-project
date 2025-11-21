import visualization.export as export
from pathlib import Path


def test_get_output_dir_creates_directory(tmp_path):
    """
    get_output_dir should create the directory if it does not exist,
    and return a Path object pointing to it.
    """
    folder = tmp_path / "Figures"
    assert not folder.exists()

    out_dir = export.get_output_dir(folder)

    assert out_dir == folder
    assert out_dir.exists()
    assert out_dir.is_dir()


def test_save_plot_writes_file(tmp_path, monkeypatch):
    """
    save_plot should call fig.savefig() with the correct filepath.
    We mock fig to avoid using real matplotlib.
    """
    # Mock figure object with a fake savefig
    calls = []

    class FakeFig:
        def savefig(self, path, dpi=None, bbox_inches=None):
            calls.append({
                "path": Path(path),
                "dpi": dpi,
                "bbox_inches": bbox_inches
            })

    fake_fig = FakeFig()

    # Run save_plot
    export.save_plot(fake_fig, "test_plot", folder=str(tmp_path))

    # Exactly one call expected
    assert len(calls) == 1

    saved = calls[0]
    assert saved["dpi"] == 300
    assert saved["bbox_inches"] == "tight"

    # Check the output file name
    assert saved["path"].name == "test_plot.png"

    # Check that the folder matches tmp_path
    assert saved["path"].parent == tmp_path
