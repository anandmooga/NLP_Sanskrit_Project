import threading
import os,pickle,bz2,csv,json
import torch,sys
import numpy as np
from networkx import *
import pandas as pd
import networkx as nx
from collections import defaultdict



def threadfun(startf,endf,pid):
	## Individual and Co-Occurence Counts
	print("Started Process : {} , PID :{}, Startf:{}  Endf: {}".format(pid,os.getpid(),startf,endf))
	print("Loading Individual count files..")
	with open('CNG_count.pickle', 'rb') as handle:
		CNG_count = pickle.load(handle)
	with open('CNGG_count.pickle', 'rb') as handle:
		CNGG_count = pickle.load(handle)
	with open('Word_count.pickle', 'rb') as handle:
		Word_count = pickle.load(handle)
	with open('Lemma_count.pickle', 'rb') as handle:
		Lemma_count = pickle.load(handle)
	def from_dict(Type_1, Type_2):
		with open(Type_1 + '|' + Type_2 + '.json', 'r') as fp:
			d = json.load(fp)
		return d
	print("Loading Co-Occurence count files...")
	CNG_Distinct = len(CNG_count.keys())
	graph = { 'LemmaLemma': from_dict('Lemma', 'Lemma'), 'LemmaWord' : from_dict('Lemma', 'Word'), 'LemmaCNG' : from_dict('Lemma', 'CNG'), 'LemmaCNG_Group' : from_dict('Lemma', 'CNG_Group'), 'WordLemma' : from_dict('Word', 'Lemma'), 'WordWord' : from_dict('Word', 'Word'), 'WordCNG' : from_dict('Word', 'CNG'), 'WordCNG_Group' : from_dict('Word', 'CNG_Group'), 'CNGLemma' : from_dict('CNG', 'Lemma'), 'CNGWord' : from_dict('CNG', 'Word'), 'CNGCNG' : from_dict('CNG', 'CNG'), 'CNGCNG_Group' : from_dict('CNG', 'CNG_Group'), 'CNG_GroupLemma' : from_dict('CNG_Group', 'Lemma'), 'CNG_GroupWord' : from_dict('CNG_Group', 'Word'), 'CNG_GroupCNG' : from_dict('CNG_Group', 'CNG'), 'CNG_GroupCNG_Group' : from_dict('CNG_Group', 'CNG_Group') } 
	savedir = 'features/'

	## Reading Metapaths
	# metapaths = []
	# with open('feature_ranklist_BM2_t2.txt','r') as file:
	#     rd = file.readlines()
	#     for row in rd:
	#         metapaths.append(row.split(',')[1])
			
	# print(len(metapaths))

	# Reading Metapaths
	df = pd.read_csv("featureStats.csv")
	metapaths = list(df[df["p2_4K_bigram_mir"]==1]['FeatureName'])
	print(len(metapaths))
	##Some utility functions    
	def checktype(el):
		if(el.lstrip("-").isdigit()):
			return "CNG"
		elif(el=='C'):
			return 'C'
		elif(el=='T'):
			return "T"
		elif(el=='L'):
			return "L"
		else:
			return "CNG_Group"
		
	def denfun(el,eltype):
		if(eltype=='CNG' or eltype=='C'):
			return CNG_count.get(int(el),0)
		elif(eltype=='L'):
			return Lemma_count.get(el,0)
		elif(eltype=='W'):
			return Word_count.get(el,0)
		else:
			return CNGG_count.get(el,0)

	def changetype(typ):
		if(typ=='L'):
			return "Lemma"
		elif(typ=='C'):
			return "CNG"
		elif(typ=='T'):
			return "Word"
		else:
			return typ

	##Actual Work Starts here
	gdir = 'After_graphml/'
	x = os.listdir(gdir)
	x.sort()
	fc = 0

	print("Started Iterating over files :{} - {}".format(startf,endf))


	for gfile in x[startf:endf]:
		##iterating over 119k files
		try:
			G = read_graphml(gdir+gfile)
			cur = []
			for i in range(1+G.number_of_nodes()):
				cur.append([])
				for j in range(1+G.number_of_nodes()):
					cur[i].append(0)

			glemma = nx.get_node_attributes(G,'lemma')
			gword = nx.get_node_attributes(G,'word')
			gcng = nx.get_node_attributes(G,'cng')
			ec = 0
			for snode,enode,d in G.edges_iter(data=True):
				##iterating over all edges 
				# print(snode,enode)
				ar = np.zeros(1500)
				r = 0
				c = 0
				w = 0
				l = 0
				g = 0
				o = 0    
				for row in metapaths:
					##iterating over 1500 metapaths
					row = row.split('*')
					if(len(row)==2):
						node1 = row[0]
						type1 = checktype(node1)
						if(type1=='T'):
							node1 = glemma[snode]+'_'+str(gcng[snode])
						elif(type1=='L'):
							node1 = glemma[snode]
						elif(type1=='C'):
							node1 = gcng[snode]
						den1 = denfun(node1,type1)
						node2 = row[1]
						type2 = checktype(node2)
						if(type2=='T'):
							node2 = glemma[enode]+'_'+str(gcng[enode])
						elif(type2=='L'):
							node2 = glemma[enode]
						elif(type2=='C'):
							node2 = gcng[enode]
						type1 = changetype(type1)
						type2 = changetype(type2)
						type12 = type1+type2
						num12 = graph[type12].get(str(node1) + '|' + str(node2),0)
						prob12 = (float(num12) + 1)/(den1 + CNG_Distinct)
						prob = prob12
						
					elif(len(row)==3):
						node1 = row[0]
						type1 = checktype(node1)
						if(type1=='T'):
							node1 = glemma[snode]+'_'+str(gcng[snode])
						elif(type1=='L'):
							node1 = glemma[snode]
						elif(type1=='C'):
							node1 = gcng[snode]
						den1 = denfun(node1,type1)
						node2 = row[1]
						type2 = checktype(node2)
						den2 = denfun(node2,type2)
						node3 = row[2]
						type3 = checktype(node3)
						if(type3=='T'):
							node3 = glemma[enode]+'_'+str(gcng[enode])
						elif(type3=='L'):
							node3 = glemma[enode]
						elif(type3=='C'):
							node3 = gcng[enode]
						type1 = changetype(type1)
						type2 = changetype(type2)
						type3 = changetype(type3)
						type12 = type1+type2
						type23 = type2+type3
						 
						num12 = graph[type12].get(str(node1) + '|' + str(node2),0)
						num23 = graph[type23].get(str(node2) + '|' + str(node3),0)
						prob12 = (float(num12) + 1)/(den1 + CNG_Distinct)
						prob23 = (float(num23) + 1)/(den2 + CNG_Distinct)
						prob = prob12*prob23
						
					elif(len(row)==4):
						node1 = row[0]
						type1 = checktype(node1)
						if(type1=='T'):
							node1 = glemma[snode]+'_'+str(gcng[snode])
						elif(type1=='L'):
							node1 = glemma[snode]
						elif(type1=='C'):
							node1 = gcng[snode]
						den1 = denfun(node1,type1)
						node2 = row[1]
						type2 = checktype(node2)
						den2 = denfun(node2,type2)
						node3 = row[2]
						type3 = checktype(node3)
						den3 = denfun(node3,type3)
						node4 = row[3]
						type4 = checktype(node4)
						if(type4=='T'):
							node4 = glemma[enode]+'_'+str(gcng[enode])
						elif(type4=='L'):
							node4 = glemma[enode]
						elif(type4=='C'):
							node4 = gcng[snode]
						den4 = denfun(node4,type4)
						type1 = changetype(type1)
						type2 = changetype(type2)
						type3 = changetype(type3)
						type4 = changetype(type4)
						type12 = type1+type2
						type23 = type2+type3
						type34 = type3+type4
						num12 = graph[type12].get(str(node1) + '|' + str(node2),0)
						num23 = graph[type23].get(str(node2) + '|' + str(node3),0)
						num34 = graph[type34].get(str(node3) + '|' + str(node4),0)
						prob12 = (float(num12) + 1)/(den1 + CNG_Distinct)
						prob23 = (float(num23) + 1)/(den2 + CNG_Distinct)
						prob34 = (float(num34) + 1)/(den3 + CNG_Distinct)
						prob = prob12*prob23*prob34
					else:
						print("Invalid Metapath length")
					ar[r] = prob
					r+=1

				cur[int(snode)][int(enode)] = ar
			fc+=1
			print("File Number :{}; pid: {}".format(fc,pid))
			print("fine till here")
			with bz2.open(str(savedir)+str(gfile.split(".graphml")[0])+'.bz2', 'wb') as f:
			    pickle.dump(cur,f)

			# with bz2.open(str(savedir)+str(gfile.split(".graphml")[0])+'.bz2', 'rb') as f:
			# 	y = pickle.load(f)


		except Exception as e:
			print(e)
			print("Error at file :{}".format(str(gfile)))
			continue
	print("All Done for pid :{}".format(pid))


# threadfun(0,1000,1)