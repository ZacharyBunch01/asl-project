'''

parseData.py

PURPOSE : Parse the dataset, then pass for visualization

'''

# Import packages
import pandas as pd
import numpy as np
import os
import re
import sys

from pathlib import Path
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep.prepareData import signData, attemptRowGet
from utils.classes import Sign


'''	
	---------------------
	- Process Sign Data -
	---------------------
'''

# If we have no sign data, the program cannot continue
if signData is None:
	raise RuntimeError("signdata.csv failed to load")

# Sort signs by name so we can look up by name
indexedSignData = signData.set_index('LemmaID')

# Initiate object containing all signs in dataset
signs = {}

# Append signs into our signs object
for name, row in indexedSignData.iterrows():
	signs[name] = {
		"Movement":        attemptRowGet(row, "Movement"),
		"MajorLocation":   attemptRowGet(row, "MajorLocation"),
		"MinorLocation":   attemptRowGet(row, "MinorLocation"),
		"Handshape":       attemptRowGet(row, "Handshape"),
		**row.to_dict()
	}	


'''
	--------
	- Test -
	--------	
'''

def buildSigns(sign_df):
    if sign_df is None:
        raise RuntimeError("signdata.csv failed to load")

    indexed = sign_df.set_index("LemmaID")
    signs = {}
    for name, row in indexed.iterrows():
        signs[name] = {
            "Movement":      attemptRowGet(row, "Movement"),
            "MajorLocation": attemptRowGet(row, "MajorLocation"),
            "MinorLocation": attemptRowGet(row, "MinorLocation"),
            "Handshape":     attemptRowGet(row, "Handshape"),
            **row.to_dict(),
        }
    return signs


# Only run the demo when executing as a script
if __name__ == "__main__":
    signs = build_signs(signData)

    tree = Sign("hello", signs["hello"])
    print(f"Movement: {tree.movement}")
    print(f"Major Location: {tree.majorLocation}")
    print(f"Minor Location: {tree.minorLocation}")
    print(f"Handshape: {tree.handshape}")

# Print all sign variables
# print(sorted(signData.columns.tolist()))

