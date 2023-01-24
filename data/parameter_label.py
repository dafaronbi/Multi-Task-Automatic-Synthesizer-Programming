import numpy as np

serum_labels = ["vol","vol","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","vol","osc","osc","osc",
"osc","osc","osc","osc","osc","osc","osc","osc","osc","vol","osc","osc","osc","osc","osc","vol","osc","env","env","env","env",
"env","fil","fil","fil","fil","fil","fil","fil","fil","fil","fil","fil","env","env","env","env","env","env","env","env","env","env",
"lfo", "lfo", "lfo", "lfo","pitch", "pitch","mod","mod","mod","mod","env","env","env","env","env","env","env","env","env","pitch","fx",
"fx","fx","fx","fx","fx","fx","fil","fil","fil","fil","fil","fil","fil","fil","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx",
"fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx",
"fx","fx","fx","fx","fx","fx","fil","fil","fil","fil","fil","fil","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx",
"fil","fil","fx","osc","osc","pitch","pitch","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","mod","mod","mod",
"mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod",
"mod","mod","mod","mod","mod","mod","osc","osc","osc","osc","fil","mod","mod","mod","mod","mod","vol","lfo","lfo","lfo","lfo","pitch",
"mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod",
"mod","mod","mod","mod","mod","mod","mod","mod","mod","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","fil","fx","fx","fx","fx","lfo","lfo",
"lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","fx","fx","fx","fx","fx","fx","fx","fx","fil","fx","lfo",
"lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo"]

diva_labels = ['vol',"fx","fx","fx","pitch","pitch","pitch","pitch","pitch","pitch","pitch","pitch","pitch","pitch","pitch","pitch","pitch",
"pitch","fx","fx","pitch","fil","pitch","osc","env","mod","mod","mod","mod","mod","mod","mod","mod","env","env","env","env","env","env","env",
"env","env","env","env","env","env","env","env","env","env","env","env","env","env","env","lfo", "lfo", "lfo", "lfo", "lfo", "lfo", "lfo", "lfo",
"lfo", "lfo", "lfo", "lfo", "lfo", "lfo", "lfo", "lfo", "lfo", "lfo", "lfo", "lfo","mod","mod","mod","mod","mod","mod","mod","mod","mod","mod",
"osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc",
"osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc","osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc",
"osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc", "osc","fil","fil", "fil","fil","fil","fil","fil","fil","fil","fil",
"fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "fil", "vol", "vol","vol", "vol", "vol",
"vol","vol","vol","vol","fx","fx", "fx", "fx", "fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx",
"fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx",
"fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx","fx",
"fx","osc","osc","osc","fil","fil","fil","fil","fx","lfo","lfo","fx","fx","fx","fx","osc","osc","osc"]

tyrell_labels = ["vol", "mod","mod","mod","mod","mod","mod","mod","mod","fx","fx","fx","fx","fx","fx","fx","fx","fx",
"env", "env", "env", "env", "env", "env", "env", "env","env","env","env","env","env","env","env","env","lfo","lfo","lfo","lfo",
"osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","osc","vol","vol","vol","vol","osc","osc"
,"fil","fil","fil","fil","fil","fil","fil","fil","fil","fil","mod","mod","mod","mod","fx","fx","fx","fx","lfo","lfo","lfo","lfo","lfo"
,"lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo","lfo"]

