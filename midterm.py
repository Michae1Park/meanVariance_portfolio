"""
python3.7
numpy
pyplot
pip install xlrd==1.2.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#***************************************************
# Problem 1
#***************************************************

# Read Data from Excel
filename = "ISYE6227_project_dataset.xlsx"

df = pd.read_excel(filename, sheet_name = None)
allsheetnames = list(df.keys())
sheetnames = allsheetnames[:-3] #-3
print(sheetnames)

col = "Adj Close"
sfile = {}
for sheet in sheetnames:
	terms = []
	df_terms = pd.read_excel(filename, sheet_name = sheet)
	for index, row in df_terms.iterrows():
		terms.append(row[col])
	sfile[sheet] = terms

#========
# Part A
#========
meanstd_pair = []

for stock in sfile:
	mean = sum(sfile[stock])/len(sfile[stock])
	var = sum([((x - mean) ** 2) for x in sfile[stock]])/len(sfile[stock])
	std = var ** 0.5
	meanstd_pair.append([mean, std])
	# print("stock, "+str(mean)+" "+str(std))
# print(meanstd_pair)

meanstd_pair = np.array(meanstd_pair)
plt.figure()
plt.scatter(meanstd_pair[:,0], meanstd_pair[:,1])
plt.show()


#========
# Part B
#========
