import visualization.export as export
from pathlib import Path


def test_get_output_dir_creates_directory(tmp_path):
    # Choose a folder inside pytest's temporary directory
    folder = tmp_path / "Figures"

    # It should NOT exist yet
    assert not folder.exists()

    # Call helper that should create it
    out_dir = export.get_output_dir(folder)

    # Returned path should match the requested folder
    assert out_dir == folder

    # Directory should now exist
    assert out_dir.exists()
    assert out_dir.is_dir()


def test_save_plot_writes_file(tmp_path, monkeypatch):
    # Track savefig() calls from our fake figure
    calls = []

    # Fake object to replace a matplotlib Figure
    class FakeFig:
        def savefig(self, path, dpi=None, bbox_inches=None):
            # Instead of writing a file, just record the call
            calls.append({
                "path": Path(path),
                "dpi": dpi,
                "bbox_inches": bbox_inches
            })

    fake_fig = FakeFig()

    # Execute the save_plot function using the fake figure
    export.save_plot(fake_fig, "test_plot", folder=str(tmp_path))

    # Exactly one savefig call should have occurred
    assert len(calls) == 1
    saved = calls[0]

    # dpi and bbox arguments must match the implementation
    assert saved["dpi"] == 300
    assert saved["bbox_inches"] == "tight"

    # Filename must match expected output
    assert saved["path"].name == "test_plot.png"

    # Check that the folder matches tmp_path
    assert saved["path"].parent == tmp_path
