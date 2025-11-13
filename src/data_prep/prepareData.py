'''

prepareData.py

PURPOSE : Preprocess data, then pass for parsing.

'''

# Import packages
import pandas as pd
import numpy as np
import os
import re

from pathlib import Path
from dataclasses import dataclass

# Attempt to read csv file
# Throw an exception if unable to
def read_csv(path):
	try:
		print(f"Loading {path}")
		data = pd.read_csv(path, encoding="latin1", on_bad_lines="skip", low_memory=False)
		print(f"Successfully opened {path}")
		return data
	except Exception as e:
		print(f"Cannot open data: {path}. Error: {e}")
		# If an exception is thrown, return nothing
		return None


'''
	-----------------
	-  Access Data  -
	-----------------
'''

# Path to dataset folder
dataPath        = Path("../Data")

# Sign handshape and movement data
signData        = read_csv(dataPath / "signdata.csv")
signDataKey     = read_csv(dataPath / "signdataKEY.csv")


'''
# Reaction time and sign familiarity
aslLEXR	        = read_csv(dataPath / "Frequency/ASLLEXR.csv")
aslLEXRKey      = read_csv(dataPath / "Frequency/ASLLEXRkey.csv") 
# freqStatus    = read_csv(dataPath / "Frequency/Freq_ASLstatus.R")

# Iconic rating of signs
iconDTrial      = read_csv(dataPath / "Iconicity/IconD_trial.csv")
iconicityTrial  = read_csv(dataPath / "Iconicity/IconicityTrial.csv")

# Pairs of similar-looking signs
neigborPairs    = read_csv(dataPath / "Phonology/NeigborPairs.csv")
''' 

'''
	---------------------------
	- Preprocessing Functions -
	---------------------------
'''

# Test availability of extensions for attributes
# Variables are either stored under ".2.0", "M2.2.0", "M3.2.0", ...
def attemptRowGet(row: pd.Series, prefix: str):

	# Init correctValue
	correctValue = ""

	# Account for varying stems for each attribute
	correctPathCandidates = [f"{prefix}.2.0"] + [f"{prefix}M{i}.2.0" for i in range(2, 7)]

	# Check if attribute is assigned value according to stem
	for candidate in correctPathCandidates:	
		if (correctValue := row.get(candidate)) is not None:
			return correctValue
	
	print(f"{prefix} attribute does not exist")
	return None 

