import numpy as np
import pandas as pd
from datetime import datetime
from pytz import timezone
# import tensorflow as tf
# from nn_module_order import NN_Order
from Train_clique import *
import os,pickle
from word_definite import *
from edgeselection_module import *
from networkx import read_graphml
"""
################################################################################################
#############################   TRAINER CLASS DEFINITION  ######################################
################################################################################################
"""
class Trainer:
	def __init__(self, modelFile = None):
		if modelFile is None:
			singleLayer = True
			self._edge_vector_dim = 1500
			if singleLayer:
				self.hidden_layer_size = 1200
				keep_prob = 0.6
				self.neuralnet = NN(self._edge_vector_dim, self.hidden_layer_size, outer_relu=True, keep_prob=keep_prob)
			else:
				# DeepR Network
				self.hidden_layer_size = 800
				self.hidden_layer_size2 = 800
				self.neuralnet = NN_2(self._edge_vector_dim, self.hidden_layer_size,\
									  hidden_layer_2_size = self.hidden_layer_size2, outer_relu=True)
				self.history = defaultdict(lambda: list())
		else:
			loader = pickle.load(open(filename, 'rb'))
			
			self.neuralnet.n = loader['n']
			self.neuralnet.d = loader['d']

			self.neuralnet = NN(self._edge_vector_dim, self.hidden_layer_size, outer_relu=True)

			self.neuralnet.U = loader['U']
			self.neuralnet.W = loader['W']
			self.neuralnet.B1 = loader['B1']
			self.neuralnet.B2 = loader['B2']
			
			self.history = defaultdict(lambda: list())
			
		# SET LEARNING RATES
		if self.neuralnet.version == 'h1':
			self.neuralnet.etaW = 3e-5
			self.neuralnet.etaB1 = 1e-5

			self.neuralnet.etaU = 1e-5
			self.neuralnet.etaB2 = 1e-5
		elif self.neuralnet.version == 'h2':
			self.neuralnet.etaW1 = 3e-4
			self.neuralnet.etaB1 = 1e-4

			self.neuralnet.etaW2 = 1e-4
			self.neuralnet.etaB2 = 1e-4
			
			self.neuralnet.etaU = 1e-4
			self.neuralnet.etaB3 = 1e-4
			
			
	def Reset(self):
		self.neuralnet = NN(self._edge_vector_dim, self.hidden_layer_size)
		self.history = defaultdict(lambda: list())
		
	def Save(self, filename):
		print('Weights Saved: ', filename)
		if self.neuralnet.version == 'h1':
			pickle.dump({
					'U': self.neuralnet.U,
					'W': self.neuralnet.W,
					'n': self.neuralnet.n,
					'd': self.neuralnet.d,
					'B1': self.neuralnet.B1,
					'B2': self.neuralnet.B2,
					'keep_prob': self.neuralnet.keep_prob,
					'version': self.neuralnet.version
				}, open(filename, 'wb'))
			return
		elif self.neuralnet.version == 'h2':
			pickle.dump({
					'U': self.neuralnet.U,
					'B3': self.neuralnet.B3,
					'W2': self.neuralnet.W2,
					'B2': self.neuralnet.B2,
					'W1': self.neuralnet.W1,
					'B1': self.neuralnet.B1,
					'h1': self.neuralnet.h1,
					'h2': self.neuralnet.h2,
					'd': self.neuralnet.d,
					'version': self.neuralnet.version
				}, open(filename, 'wb'))
			return
		
	
	def Load(self, filename):
		loader = pickle.load(open(filename, 'rb'))
		if 'version' not in loader: # means 1 hidden layer
			self.neuralnet = NN(self._edge_vector_dim, self.hidden_layer_size, outer_relu=True)
			self.neuralnet.U = loader['U']
			self.neuralnet.W = loader['W']
			self.neuralnet.B1 = loader['B1']
			self.neuralnet.B2 = loader['B2']
			self.neuralnet.hidden_layer_size = loader['n']
			self.neuralnet._edge_vector_dim = loader['d']
			if 'keep_prob' in loader:
				self.neuralnet.keep_prob = loader['keep_prob']
				self.neuralnet.dropout_prob = 1 - loader['keep_prob']
			print('Keep Prob = {}, Dropout = {}'.format(self.neuralnet.keep_prob, self.neuralnet.dropout_prob))
		else:
			if loader['version'] == 'h1':
				self.neuralnet = NN(self._edge_vector_dim, self.hidden_layer_size, outer_relu=True)
				self.neuralnet.U = loader['U']
				self.neuralnet.W = loader['W']
				self.neuralnet.B1 = loader['B1']
				self.neuralnet.B2 = loader['B2']
				self.neuralnet.hidden_layer_size = loader['n']
				self.neuralnet._edge_vector_dim = loader['d']
				if 'keep_prob' in loader:
					self.neuralnet.keep_prob = loader['keep_prob']
					self.neuralnet.dropout_prob = 1 - loader['keep_prob']
				print('Keep Prob = {}, Dropout = {}'.format(self.neuralnet.keep_prob, self.neuralnet.dropout_prob))
			elif loader['version'] == 'h2':
				self.neuralnet = NN_2(self._edge_vector_dim, self.hidden_layer_size, outer_relu=True)
				
				self.neuralnet.U = loader['U']
				self.neuralnet.B3 = loader['B3']
				
				self.neuralnet.W2 = loader['W2']
				self.neuralnet.B2 = loader['B2']
				
				self.neuralnet.W1 = loader['W1']
				self.neuralnet.B1 = loader['B1']
				
				self.neuralnet.h1 = loader['h1']
				self.neuralnet.h2 = loader['h2']
				self.neuralnet.d = loader['d']
		
	def CalculateLoss_n_Grads(self, WScalarMat, min_st_adj_worst, max_st_adj_gold, loss_type = 0, min_marginalized_energy = None):
		doBpp = True
		
		# Claculate the enrgies
		etg = np.sum(WScalarMat[max_st_adj_gold])
		etq = np.sum(WScalarMat[min_st_adj_worst])
		
		if loss_type == 0:
			# Variable Hinge Loss - CHECKED
			L = etg - min_marginalized_energy
			if L > 0:
				# print("Do BackProp")
				dLdOut = np.zeros_like(WScalarMat)
				dLdOut[max_st_adj_gold&(~min_st_adj_worst)] = 1
				dLdOut[(~max_st_adj_gold)&min_st_adj_worst] = -1
			else:
				doBpp = False
				return (L, None, doBpp)
		elif loss_type == 1:
			# LOg Loss
			a = etg - etq
			b = np.exp(a)
			L = np.log(1 + b)
			
			dLdOut = np.zeros_like(WScalarMat)
			dLdOut[max_st_adj_gold&(~min_st_adj_worst)] = 1
			dLdOut[(~max_st_adj_gold)&min_st_adj_worst] = -1
			
			dLdOut *= (b/(1 + b))
		elif loss_type == 2:
			# Square exponential loss
			gamma = 1
			b = np.exp(-etq)
			
			L = etg**2 + gamma*b
			
			dLdOut = np.zeros_like(WScalarMat)
			dLdOut[max_st_adj_gold&(~min_st_adj_worst)] = 2*etg
			dLdOut[(~max_st_adj_gold)&min_st_adj_worst] = -gamma*b
			pass
		return (L, dLdOut, doBpp)
	def Test(self, sentenceObj, dcsObj, dsbz2_name, _dump = False, _outFile = None):
		if _dump:
			if _outFile is None:
				raise Exception('WTH r u thinking! pass me outFolder')
		if self.neuralnet.version == 'h1':
			self.neuralnet.ForTesting()

		# with open('gt_cngs.csv','a') as fh:
		#     for i in dcsObj.cng:
		#         for j in i:
		#             print(str(sentenceObj.sent_id)+":"+str(j))
		#             wr = csv.writer(fh)
		#             wr.writerow([sentenceObj.sent_id,j])

		# return
		neuralnet = self.neuralnet
		minScore = np.inf
		minMst = None
		
		# dsbz2_name = sentenceObj.sent_id + '.ds.bz2'
		(nodelist_correct, conflicts_Dict_correct, featVMat_correct, nodelist_to_correct_mapping,\
			nodelist, conflicts_Dict, featVMat) = open_dsbz2(dsbz2_name)
		
		# if len(nodelist) > 50:
		#     return None

		if not self.neuralnet.outer_relu:
			(WScalarMat, SigmoidGateOutput) = Get_W_Scalar_Matrix_from_FeatVect_Matrix(featVMat, nodelist, conflicts_Dict, neuralnet)
		else:
			WScalarMat = Get_W_Scalar_Matrix_from_FeatVect_Matrix(featVMat, nodelist, conflicts_Dict, neuralnet)
		
		# print('NeuralNet Time: ', time.time() - startT)
		# startT = time.time()
		
		# Get all MST
		# print('before getting all cliques')
		cliqset  = set()
		for source in range(len(nodelist)):
			(mst_nodes, mst_adj_graph, mb) = clique(nodelist, WScalarMat, conflicts_Dict, source)
			# print('.', end = '')
			cst = ''
			for i in mb:
				if(i):
					cst=cst+'1'
				else:
					cst=cst+'0'
			cliqset.add(cst)

			score = GetMSTWeight(mst_adj_graph, WScalarMat)
			if(score < minScore):
				minScore = score
				minMst = mst_nodes
		# print('Enumerated number of cliques : '+str(len(cliqset)))
		# print('after getting all cliques')
		dcsLemmas = [[rom_slp(l) for l in arr]for arr in dcsObj.lemmas]
		word_match = 0
		lemma_match = 0
		n_output_nodes = 0
		
		if _dump:
			predicted_lemmas = [sentenceObj.sent_id]
			predicted_cngs = [sentenceObj.sent_id]
			predicted_chunk_id = [sentenceObj.sent_id]
			predicted_pos = [sentenceObj.sent_id]
			predicted_id = [sentenceObj.sent_id]
		
		for chunk_id, wdSplit in minMst.items():
			for wd in wdSplit:
				if _dump:
					predicted_lemmas.append(wd.lemma)
					predicted_cngs.append(wd.cng)
					predicted_chunk_id.append(wd.chunk_id)
					predicted_pos.append(wd.pos)
					predicted_id.append(wd.id)
					
				n_output_nodes += 1
				# Match lemma
				search_result = [i for i, j in enumerate(dcsLemmas[chunk_id]) if j == wd.lemma]
				if len(search_result) > 0:
					lemma_match += 1
				# Match CNG
				for i in search_result:
					if(dcsObj.cng[chunk_id][i] == str(wd.cng)):
						word_match += 1
						# print(wd.lemma, wd.cng)
						break
		dcsLemmas = [l for arr in dcsObj.lemmas for l in arr]
		
		if _dump:
			with open(_outFile, 'a') as fh:
				dcsv = csv.writer(fh)
				dcsv.writerow(predicted_lemmas)
				dcsv.writerow(predicted_cngs)
				dcsv.writerow(predicted_chunk_id)
				dcsv.writerow(predicted_pos)
				dcsv.writerow(predicted_id)
				dcsv.writerow([sentenceObj.sent_id, word_match, lemma_match, len(dcsLemmas), n_output_nodes])
		
		# print('All MST Time: ', time.time() - startT)
		# print('Node Count: ', len(nodelist))
#         print('\nFull Match: {}, Partial Match: {}, OutOf {}, NodeCount: {}, '.\
#               format(word_match, lemma_match, len(dcsLemmas), len(nodelist)))
		return (word_match, lemma_match, len(dcsLemmas), n_output_nodes)
	
	def Train(self, featVMat, gold_edges_mask, edges_to_consider, node_list, total_nodes,mode ='nearest_neighbour',_debug = True):
		self.neuralnet.ForTraining()
		self.neuralnet.new_dropout() # renew dropout setting
		# Hyperparameter for hinge loss: m
		m_hinge_param = 14
		
				
		""" FORM MAXIMUM(ENERGY) SPANNING TREE OF THE GOLDEN GRAPH : WORST GOLD STRUCTURE """
		# WScalarMat_correct = Get_W_Scalar_Matrix_from_FeatVect_Matrix(featVMat_correct, self.neuralnet)
		# WScalarMat_correct = WScalarMat_correct[edges_to_consider]
		
		WScalarMat = Get_W_Scalar_Matrix_from_FeatVect_Matrix(featVMat, self.neuralnet)
		# WScalarMat  = WScalarMat[edges_to_consider]
		# print("WscalarMat")		
		# print(WScalarMat)
		

		
		# print("WscalarMat_correct")
		# print(WScalarMat_correct)

		energy_gold_max_ST = np.sum(WScalarMat[gold_edges_mask])
		
		""" Delta(Margin) Function : MASK FOR WHICH CORRECT EDGES"""
		margin_f = lambda edges_mask: np.sum(edges_mask&(~gold_edges_mask))**2
		
		""" FOR ALL POSSIBLE MST FROM THE COMPLETE GRAPH """
		
		""" For each node - Find MST with that source"""
		min_STx = None # Min Energy spanning tree with worst margin with gold_STx
		min_marginalized_energy = np.inf
		
		# Generate random set of nodes from which mSTs are to be considered
		
		best_node_diff = np.Inf 
		best_energy = np.inf
		# print('before###')
		if(mode=='least_edge_first'):
			# print("In Least Edge mode")
			mst_adj_graph = least_edge_first(WScalarMat,edges_to_consider,node_list, total_nodes) # T_X
			# print("Predicted Edges")
			# print(mst_adj_graph)
			en_st = np.sum(WScalarMat[mst_adj_graph])
			delta_st = margin_f(mst_adj_graph)

			if _debug:
				if best_energy > en_st:
					best_node_diff = delta_st
					best_energy = en_st
				
			# Minimum marginalized energy calculation
			marginalized_en = en_st - delta_st
			if marginalized_en < min_marginalized_energy:
				min_marginalized_energy = marginalized_en
				min_STx = mst_adj_graph
			# Energy diff should all be negative
			# if _debug:
				# print('Source: [{}], Node_Diff:{}, Max_Gold_En: {:.3f}, Energy: {:.3f}'.\
				# 	  format( np.sum((~gold_nodes_mask)&mst_nodes_bool), energy_gold_max_ST,  np.sum(WScalarMat[mst_adj_graph])))
		

		if _debug:
			print('Best Node diff: {} with EN: {}'.format(np.sqrt(best_node_diff), best_energy))
		""" Gradient Descent """
		# LOSS TYPES -> hinge(0), log-loss(1), square-exponential(2)

		# print("Actual: ")
		# print(gold_edges_mask)
		# print("Predicted")
		# print(min_STx)

		print("Total Edges ",np.sum(gold_edges_mask)," Correctly Predicted Edges ",np.sum(min_STx&gold_edges_mask))
		Total_Loss, dLdOut, doBpp = self.CalculateLoss_n_Grads(WScalarMat, min_STx, gold_edges_mask,\
																 loss_type = 0, min_marginalized_energy = min_marginalized_energy)
		# print("doBPP : ",doBpp,"Total Loss",Total_Loss)
		# print("dLdOut")
		# print(dLdOut)
		# print("Fine Till Backpropagation")
		if doBpp:
			
			self.neuralnet.Back_Prop(dLdOut, total_nodes, featVMat, _debug)
		print("After BackPropagation")
		# else:
			# trainingStatus[sentenceObj.sent_id] = True
		# if _debug:
		# 	print("\nFileKey: %s, Loss: %6.3f" % (sentenceObj.sent_id, Total_Loss))

TrainFiles = None
trainer = None
p_name = ''
odir = ''
	
def register_nnet(nnet):
	if not os.path.isdir(odir):
		print("Creating a directory : "+str(odir))
		os.mkdir(odir)
	if not os.path.isfile('outputs/nnet_LOGS.csv'):
		with open('outputs/nnet_LOGS.csv', 'a') as fh:
			csv_r = csv.writer(fh)
			csv_r.writerow(['odir', 'p_name', 'hidden_layer_size', '_edge_vector_dim'])
	with open('outputs/nnet_LOGS.csv', 'a') as fh:
		csv_r = csv.writer(fh)
		if nnet.version == 'h1':
			csv_r.writerow([odir, p_name, nnet.n, nnet.d])
		elif nnet.version == 'h2':
			csv_r.writerow([odir, p_name, nnet.h1, nnet.h2, nnet.d])
p_name = None
odir = None
trainer = None


def train_generator(train_files,node_dict, feat_dir):
	register_nnet(trainer.neuralnet)
	print("No. of training files: {}".format(len(train_files)))
	fc = 0
	for t_file in train_files:
		fc+=1
		t_file = str(t_file)
		print(t_file)
		# t_file = "105433"
		# t_file = '285222'
		d = node_dict[t_file]
		with bz2.open(str(feat_dir)+str(t_file)+".bz2", 'rb') as f:
			featVMat = pickle.load(f)
		G = read_graphml('After_graphml/'+str(t_file)+".graphml")
		total_nodes = G.number_of_nodes()
		if(total_nodes<2):
			print("Only {} nodes".format(total_nodes))
			continue
		# print(t_file,total_nodes)
		# if(G.number_of_nodes()>10 or len(d)<4):
		# 	continue
		# for u,v,d in G.edges_iter(data=True):
		# 	print("Edge: ",u,v)
		# for n,d in G.nodes_iter(data = True):
		# 	print("Node: ",n,d)

		# featVMat_correct = np.zeros((1+total_nodes,1+total_nodes))
		gold_edges_mask = np.ndarray((1+total_nodes,1+total_nodes), np.bool)*False	
		prev = -1		

		node_list = []
		for i in node_dict[t_file]:
			i = int(i)
			node_list.append(i)
			# print(i)
			if(prev!=-1):
				# featVMat_correct[prev][i] = featVMat[prev][i]
				gold_edges_mask[prev][i] = True
			prev = int(i)

		edges_to_consider = np.ndarray((1+total_nodes,1+total_nodes), np.bool)*False
		for i in node_list:
			for j in node_list:
				if(i==j):
					continue
				edges_to_consider[i][j] = True


		# print(gold_edges_mask)
		# print(np.sum(gold_edges_mask))
		try:
			trainer.Train(featVMat,gold_edges_mask,edges_to_consider,node_list,total_nodes = G.number_of_nodes(),mode = 'least_edge_first')
		except Exception as e:
			print(e)
			pass
		print("#"*50,"  ",fc)
		# break
		if(fc%2==0):
			print("Saving after 1000 files")
			trainer.Save(p_name.replace('.p', '_f{}.p'.format(fc)))
			break
	print("Saving For the last time")
	trainer.Save(p_name)

def main():

	global p_name, odir,trainer

	input_layer_dim = 1500
	hidden_layer_dim = 1200
	output_layer_dim = 1
	keep_prob = 0.6
	lambda1 = 0.00001
	lambda2 = 0.0001

	st = str(int((time.time() * 1e6) % 1e13))
	India = timezone('Asia/Kolkata')
	st = str(datetime.now(India).strftime('%Y-%m-%d_%H-%M-%S'))
	log_name = 'logs/train_nnet_t{}.out'.format(st)
	odir = 'outputs/train_t{}'.format(st)
	p_name = 'outputs/train_t{}/nnet.p'.format(st)
	nodecsvfile = 'gtnodeorder.csv'
	print('nEURAL nET wILL bE sAVED hERE: ', p_name)
	feat_dir = 'features/'

	trainer = Trainer()

	node_dict = dict()
	with open("gtnodeorder.csv", 'r') as csvfile:
		rd = csv.reader(csvfile)
		for row in rd:
			node_dict[row[0]] = row[1:] 

	df = pd.read_csv('train_files.csv')
	print(df.columns)
	## Call train function with appropriate list of files and also the directory to pick the files from
	train_generator(df['TrainFile'],node_dict, feat_dir)

if __name__ == '__main__':
	main()