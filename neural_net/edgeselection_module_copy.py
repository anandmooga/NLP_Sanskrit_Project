import numpy as np
import re
regex1 = re.compile('[ ]+')

# The asertion is that both the arguments must not be empty. this is by purpose
# Calculates both pairwise metric and Kendell-Tau
def Pairwise_Metric(ground_truth, predicted):
    # Find the skip bigram pairs
    assert(len(ground_truth) > 0 and len(predicted) > 0 )
    list_gt = ground_truth.strip().split()
    list_pred = predicted.strip().split()

    skip_bigrams_gt = [(list_gt[i], list_gt[j]) for i in range(len(list_gt)) for j in range(i+1, len(list_gt))]
    skip_bigrams_pred = [(list_pred[i], list_pred[j]) for i in range(len(list_pred)) for j in range(i+1, len(list_pred))]
    intersect = [i for i in skip_bigrams_pred if i in skip_bigrams_gt]


    # for kendell Tau
    skipSKipped = [i for i in skip_bigrams_pred if i not in skip_bigrams_gt]
    tau = 1 - (1.0*len(skipSKipped))/len(skip_bigrams_pred)


    # Calculate and return precision, recall, f-score
    P = len(intersect)*1.0/len(skip_bigrams_pred)
    R = len(intersect)*1.0/len(skip_bigrams_gt)
    try:
        F = 2*P*R/(P+R)
    except ZeroDivisionError:
        F = 0.0

    return P,R,F,tau

def edlcs(a, b):
    # The dynamic programming algo for longest common subsequence
    lengths = np.zeros((len(a) + 1, len(b) + 1))

    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1, j+1] = lengths[i, j] + 1
            else:
                lengths[i+1, j+1] = max(lengths[i+1, j], lengths[i, j+1])

    return lengths[-1, -1]

def LSR(ground_truth, predicted):
    # Longest sequence ratio
    list_gt = ground_truth.strip().split()
    list_pred = predicted.strip().split()
    lcs_match = edlcs(list_gt, list_pred)
    # LSR = lcs_match/len(list_pred)

    # Calculate and return precision, recall, f-score
    if len(list_pred) > 0:
        P = lcs_match/len(list_pred)
    else:
        P = 0

    if len(list_gt) > 0:
        R = lcs_match/len(list_gt)
    else:
        return [np.nan]*3

    if (P <= 0) and (R <= 0):
        F = 0
    else:
        F = 2*P*R/(P+R)

    return P,R,F

def Perfect_Match(ground_truth, predicted):
    # Perfect match b/w ground_truth sequence and predicted sequence
    list_gt = ground_truth.strip().split()
    list_pred = predicted.strip().split()
    l = min(len(list_gt), len(list_pred))
    list_gt = list_gt[:l]
    list_pred = list_pred[:l]
    pm = 0
    for i, gx in enumerate(list_gt):
        if gx == list_pred[i]:
            pm += 1
    pm = pm/float(len(list_pred))
    return pm


def least_edge_first(WScalarMat,edges_to_consider, total_nodes):
	ans_graph = np.ndarray(np.shape(edges_to_consider),np.bool)*False
	# print(len(edges_to_consider))

	ins = np.zeros(total_nodes)-np.ones(total_nodes)
	out = np.zeros(total_nodes)-np.ones(total_nodes)



	eset = set()


	for i in range(len(edges_to_consider)):
		for j in range(len(edges_to_consider)):
			if(edges_to_consider[i][j]):
				eset.add((WScalarMat[i][j],i,j))

	# print("traversing sorted edges")
	for e in sorted(eset):
		# print(e)
		sn = e[1]
		en = e[2]
		if(ins[en]==-1 and out[sn]==-1):
			ok = True
			y = out[en]

			while(y!=-1):
				if(y==sn):
					ok=False
					break
				# print(y)
				# print(type(y))
				y = out[int(y)]
			# print("before if")
			if(not ok):
				continue
			# print("after if")
			ins[en] = sn
			out[sn] = en
			ans_graph[sn][en] = True

			# print(sn,en)
	path = []
	# print("before loop")
	for i in range(total_nodes):
		if(ins[i]==-1 and out[i]!=-1):
			path.append(i)
			# print("len()")
			while(out[int(path[int(len(path))-1])]!=-1):
				# print(type(out[path[len(path)-1.0]]))
				path.append(out[int(path[int(len(path))-1])])
			break

	# print(out)
	# print(ins)
	# print(path)
	# print("after loop")
	return ans_graph, path
