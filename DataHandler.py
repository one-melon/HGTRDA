import pickle
import numpy as np
from scipy.sparse import csr_matrix,coo_matrix, dok_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log

def transpose(mat):
	coomat = coo_matrix(mat)
	return csr_matrix(coomat.transpose())

def negSamp(temLabel, sampSize, nodeNum):
	negset = [None] * sampSize
	cur = 0
	while cur < sampSize:
		rdmItm = np.random.choice(nodeNum)
		if temLabel[rdmItm] == 0:
			negset[cur] = rdmItm
			cur += 1
	return negset

def transToLsts(mat, mask=False, norm=False):
	shape = [mat.shape[0], mat.shape[1]]
	coomat = sp.coo_matrix(mat)
	indices = np.array(list(map(list, zip(coomat.row, coomat.col))), dtype=np.int32)
	data = coomat.data.astype(np.float32)

	if norm:
		rowD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=1) + 1e-8) + 1e-8)))
		colD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=0) + 1e-8) + 1e-8)))
		for i in range(len(data)):
			row = indices[i, 0]
			col = indices[i, 1]
			data[i] = data[i] * rowD[row] * colD[col]

	# half mask
	if mask:
		spMask = (np.random.uniform(size=data.shape) > 0.5) * 1.0
		data = data * spMask

	if indices.shape[0] == 0:
		indices = np.array([[0, 0]], dtype=np.int32)
		data = np.array([0.0], np.float32)
	return indices, data, shape

class DataHandler:
	def __init__(self):
		if args.data == 'nc_drug_relation':
			predir = 'Data/nc_drug_relation/'
		self.predir = predir
		if args.data == 'nc_drug_relation':
			self.trnfile = predir + 'train0'
			self.tstfile = predir + 'test0'

	def LoadData(self):
		with open(self.trnfile, 'rb') as fs:
			trnMat = (pickle.load(fs) != 0).astype(np.float32)
		# test set
		with open(self.tstfile, 'rb') as fs:
			tstMat = pickle.load(fs)
		tstLocs = [None] * tstMat.shape[0]
		tstncs = set()
		for i in range(len(tstMat.data)):
			row = tstMat.row[i]
			col = tstMat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstncs.add(row)
		tstncs = np.array(list(tstncs))

		self.trnMat = trnMat
		self.tstLocs = tstLocs
		self.tstncs = tstncs
		args.edgeNum = len(trnMat.data)
		args.nc, args.drug = self.trnMat.shape
