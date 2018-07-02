import os
for i in range(2):
	os.system("nohup python -u edge1500generate.py {} > meta{}.log &".format(i,i))