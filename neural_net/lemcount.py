import csv,os
import pickle
from DCS import *
from romtoslp import *
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

final_files = x-x_10k
print(len(final_files))
fc = 0
for file in x_10k:
    if(fc%1000==0):
        print(fc)
    fc+=1
    with open('DCS_pick/'+file,'rb') as pfile:
        obj = pickle.load(pfile)
        for leml in obj.lemmas:
            for lem in leml:
                uniqlem.add(rom_slp(lem))
print(len(uniqlem))

print("Iterating over 440k-10k files")
d = dict()
fc = 0
for file in final_files:
    if(fc%1000==0):
        print(fc)
    fc+=1
    with open('DCS_pick/'+file,'rb') as pfile:
        try:
            obj = pickle.load(pfile)
            for leml in obj.lemmas:
                for lem in leml:
                    l = rom_slp(lem)
                    d[l] = 1+d.get(l,0)
        except Exception as e:
            print(e)
            print("exception")
            continue
print("Dumping into csv file")
with open('lem_count_final_440.csv','w') as csvfile:
    rd = csv.writer(csvfile)
    rd.writerow(['Lemma','Count'])
    for k in uniqlem:
        rd.writerow([k,d.get(k,0)])

print("ALl done")
