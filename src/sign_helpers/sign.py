'''

classes.py

PURPOSE : Repository for general-use classes.

'''

# Import packages
import pandas as pd
import numpy as np
import os
import re

from pathlib import Path
from dataclasses import dataclass


'''
	---------------------
	- Define Sign Class -
	---------------------
'''

# Class for storing parsed sign data
# Data : Movement, Major Location, Minor Location, Handshape
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
	def movement(self):  
		return self.attrs.get("Movement")
	@property
	def majorLocation(self):  
		return self.attrs.get("MajorLocation")
	@property
	def minorLocation(self):
		return self.attrs.get("MinorLocation")
	@property
	def handshape(self): 
		return self.attrs.get("Handshape")



