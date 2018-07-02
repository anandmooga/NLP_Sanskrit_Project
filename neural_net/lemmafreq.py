import csv,os

print("enumerating DCS_pick")
x = os.listdir('DCS_pick')
x = set(x)
x_10k = set()
print(len(x))
print('opening csv file')
with open('filelist.csv','r') as csvfile:
	rd = csv.reader(csvfile)
	for row in rd:
		x_10k.add(row[0])
print(len(x))
print(len(x_10k))
uniqlem = set()
import pickle
from DCS import *
from romtoslp import *
fc = 0
for file in x_10k:
    if(fc%1000==0):
        print(fc)
    fc+=1
    with open('DCS_pick/'+file,'rb') as pfile:
        obj = pickle.load(pfile)
        for lem in obj.lemmas:
            uniqlem.add(rom_slp(lem))
