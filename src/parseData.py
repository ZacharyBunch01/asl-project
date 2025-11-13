'''

parseData.py

PURPOSE : Parse the dataset, then pass for visualization

'''

# Import packages
import pandas as pd
import numpy as np
import os

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
	---------------------
	- Define Sign Class -
	---------------------
'''

# Class for storing parsed sign data
# Data : Movement, Location, Handshape
class Sign:
	# Constructor
	def __init__(self, name, attrs):
		self.name  = name
		self.attrs = attrs

	# Property getter function
	def get(self, field, default=None):
		return self.attrs.get(field, default)

	# Define sign properties
	@property
	def movement(self):  return self.attrs.get("Movement.2.0")
	@property
	def location(self):  return self.attrs.get("Location.2.0")
	@property
	def handshape(self): return self.attrs.get("Handshape.2.0")


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
		"Movement":  row.get("Movement.2.0"),
		"Location":  row.get("Location.2.0"),
		"Handshape": row.get("Handshape.2.0"),
		**row.to_dict()
	}	

'''
	--------
	- Test -
	--------	
'''

# Test word parsing
tree = Sign("tree", signs["tree"])
print(f"Movement: {tree.movement}")
print(f"Location: {tree.location}")
print(f"Handshape: {tree.handshape}")

# print(sorted(signData.columns.tolist()))

