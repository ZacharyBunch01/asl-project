'''

dataPrep.py

PURPOSE : Provide unit tests for data prep.

'''

import pandas as pd
from pathlib import Path

from data_prep.prepareData import read_csv, attemptRowGet
from data_prep.parseData import buildSigns
from data_prep import prepareData
from utils.classes import Sign


'''
	--------------------
	- Verify Filepaths -
	--------------------
'''

def test_read_csv_success(tmp_path):
    # create a temporary CSV
    csv_path = tmp_path / "test.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv_path, index=False)

    df = read_csv(csv_path)

    assert isinstance(df, pd.DataFrame)
    assert list(df["a"]) == [1, 2, 3]


def test_read_csv_missing_file():
    missing = Path("this_file_does_not_exist.csv")
    df = read_csv(missing)
    assert df is None


'''
	------------------------------
	- Verify Preparing Functions -
	------------------------------
'''
	
def test_attemptRowGet_prefers_base_column():
    row = pd.Series({
        "Movement.2.0": "base_value",
        "MovementM2.2.0": "alt_value",
    })
    assert attemptRowGet(row, "Movement") == "base_value"


def test_attemptRowGet_uses_fallback_M_column():
    row = pd.Series({
        "MovementM3.2.0": "fallback_value",
    })
    assert attemptRowGet(row, "Movement") == "fallback_value"


def test_attemptRowGet_returns_none_if_missing(capfd):
    row = pd.Series({})
    result = attemptRowGet(row, "Movement")
    assert result is None
    # optional: verify it printed the warning
    out, _ = capfd.readouterr()
    assert "Movement attribute does not exist" in out


'''
	----------------------------
	- Verify Parsing Functions -
	----------------------------
'''

def test_data_files_exist():
    """
    Verify that the main data files exist in the project-level Data directory.
    Works regardless of whether pytest is run from project root or from src/.
    """
    # __file__ = .../src/tests/test_data_prep.py
    # parents[0] = tests
    # parents[1] = src
    # parents[2] = project root (my-package-name)
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "Data"

    assert (data_dir / "signdata.csv").exists()
    assert (data_dir / "signdataKEY.csv").exists()

def test_signData_loaded_if_files_present():
    # if the data files are there, signData should be a DataFrame
    if prepareData.signData is not None:
        assert isinstance(prepareData.signData, pd.DataFrame)


def test_build_signs_and_sign_wrapper():
    # tiny fake dataset with the same structure your code expects
    df = pd.DataFrame(
        [
            {
                "LemmaID": "hello",
                "Movement.2.0": "straight",
                "MajorLocation.2.0": "Neutral",
                "MinorLocation.2.0": "Neutral",
                "Handshape.2.0": "5",
            }
        ]
    )

    signs = buildSigns(df)

    assert "hello" in signs

    s = Sign("hello", signs["hello"])
    assert s.movement == "straight"
    assert s.majorLocation == "Neutral"
    assert s.minorLocation == "Neutral"
    assert s.handshape == "5"







