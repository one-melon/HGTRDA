import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate
from DataHandler import negSamp, transpose, DataHandler, transToLsts
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle

class Recommender:
	def __init__(self, sess, handler):
		self.sess = sess
		self.handler = handler

		print('ncRNA', args.nc, 'DRUG', args.drug)
		print('NUM OF INTERACTIONS', len(self.handler.trnMat.data))
		self.metrics = dict()
		mets = ['Loss', 'preLoss']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
		else:
			stloc = 0
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
		for ep in range(stloc, args.epoch):
			test = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if ep % args.tstEpoch == 0:
				self.saveHistory()
			print()
		log(self.makePrint('Test', args.epoch, reses, True))
		self.saveHistory()

	def GCN(self, nclat, dlat, adj, tpAdj):
		nclats = [nclat]
		dlats = [dlat]
		for i in range(args.gcn_hops):
			temnclat = tf.sparse.sparse_dense_matmul(adj, dlats[-1])
			temdlat = tf.sparse.sparse_dense_matmul(tpAdj, nclats[-1])
			nclats.append(temnclat)
			dlats.append(temdlat)
		return tf.add_n(nclats[1:]), tf.add_n(dlats[1:])

	def prepareKey(self, nodeEmbed):
		K = NNs.getOrDefineParam('KMapping', [args.latdim, args.latdim], reg=True, reuse=True)
		key = tf.reshape(nodeEmbed @ K, [-1, args.att_head, args.latdim//args.att_head])
		key = tf.transpose(key, perm=[1, 0, 2]) # Head * N * d'(lowrank)
		return key

	# def makeView(self, a, b):
	# 	return tf.nn.dropout(a, self.keepRate), tf.nn.dropout(b, self.keepRate)

	def propagate(self, lats, key, hyper):
		V = NNs.getOrDefineParam('VMapping', [args.latdim, args.latdim], reg=True, reuse=True)
		lstLat = tf.reshape(lats[-1] @ V, [-1, args.att_head, args.latdim // args.att_head])
		lstLat = tf.transpose(lstLat, perm=[1, 2, 0]) # Head * d' * N
		temlat1 = lstLat @ key # Head * d' * d'
		hyper = tf.reshape(hyper, [-1, args.att_head, args.latdim//args.att_head])
		hyper = tf.transpose(hyper, perm=[1, 2, 0]) # Head * d' * E
		temlat1 = tf.reshape(temlat1 @ hyper, [args.latdim, -1]) # d * E
		temlat2 = FC(temlat1, args.hyperNum, activation=self.actFunc) + temlat1
		temlat3 = FC(temlat2, args.hyperNum, activation=self.actFunc) + temlat2 # d * E

		preNewLat = tf.reshape(tf.transpose(temlat3) @ V, [-1, args.att_head, args.latdim//args.att_head])
		preNewLat = tf.transpose(preNewLat, [1, 0, 2])# Head * E * d'
		preNewLat = hyper @ preNewLat # Head * d'(lowrank) * d'(embed)
		newLat = key @ preNewLat # Head * N * d'
		newLat = tf.reshape(tf.transpose(newLat, [1, 0, 2]), [-1, args.latdim])
		lats.append(newLat)

	def edgeDropout(self, mat):
		def dropOneMat(mat):
			indices = mat.indices
			values = mat.values
			shape = mat.dense_shape
			# newVals = tf.to_float(tf.sign(tf.nn.dropout(values, self.keepRate)))
			newVals = tf.nn.dropout(values, self.keepRate)
			return tf.sparse.SparseTensor(indices, newVals, shape)
		return dropOneMat(mat)

	def meta(self, hyper):
		hyper_mean = tf.reduce_mean(hyper, axis=0, keep_dims=True)
		hyper = hyper_mean
		W1 = tf.reshape(FC(hyper, args.latdim * args.latdim, useBias=True, reg=True, name='W1_gen', reuse=True), [args.latdim, args.latdim])
		b1 = FC(hyper, args.latdim, useBias=True, reg=True, name='b1_gen', reuse=True)
		def mapping(key):
			ret = Activate(key @ W1 + b1, 'leakyRelu')
			return ret
		return mapping

	def label(self, ncKey, dkey, ncEmbed, dHyper):
		uMapping = self.meta(ncEmbed)
		iMapping = self.meta(dHyper)
		nclat = uMapping(ncKey)
		dlat = iMapping(dkey)
		lat = tf.concat([nclat, dlat], axis=-1)
		lat = FC(lat, args.latdim, activation='leakyRelu', useBias=True, reg=True) + nclat + dlat
		ret = tf.reshape(FC(lat, 1, activation='sigmoid', useBias=True, reg=True), [-1])
		return ret

	def ours(self):
		ncEmbed_ini = NNs.defineParam('ncEmbed_ini', [args.nc, args.latdim], reg=True)
		dEmbed_ini = NNs.defineParam('dEmbed_ini', [args.drug, args.latdim], reg=True)
		ncEmbed_gcn, dEmbed_gcn = self.GCN(ncEmbed_ini, dEmbed_ini, self.adj, self.tpAdj)
		ncEmbed0 = ncEmbed_ini + ncEmbed_gcn
		dEmbed0 = dEmbed_ini + dEmbed_gcn
		self.gcnNorm = (tf.reduce_sum(tf.reduce_sum(tf.square(ncEmbed_gcn), axis=-1)) + tf.reduce_sum(tf.reduce_sum(tf.square(dEmbed_gcn), axis=-1))) / 2
		self.iniNorm = (tf.reduce_sum(tf.reduce_sum(tf.square(ncEmbed_ini), axis=-1)) + tf.reduce_sum(tf.reduce_sum(tf.square(dEmbed_ini), axis=-1))) / 2

		ncEmbed = NNs.defineParam('ncEmbed', [args.hyperNum, args.latdim], reg=True)
		dHyper = NNs.defineParam('dHyper', [args.hyperNum, args.latdim], reg=True)
		ncKey = self.prepareKey(ncEmbed0)
		dKey = self.prepareKey(dEmbed0)

		nclats = [ncEmbed0]
		dlats = [dEmbed0]
		for i in range(args.gnn_layer):
			self.propagate(nclats, ncKey, ncEmbed)
			self.propagate(dlats, dKey, dHyper)

		nclat = tf.add_n(nclats)
		dlat = tf.add_n(dlats)

		pcknclat = tf.nn.embedding_lookup(nclat, self.ncids)
		pckdlat = tf.nn.embedding_lookup(dlat, self.dids)
		preds = tf.reduce_sum(pcknclat * pckdlat, axis=-1)

		idx = self.adj.indices
		ncs, drugs = tf.nn.embedding_lookup(idx[:, 0], self.edgeids), tf.nn.embedding_lookup(idx[:, 1], self.edgeids)
		ncKey = tf.reshape(tf.transpose(ncKey, perm=[1, 0, 2]), [-1, args.latdim])# N * d
		dKey = tf.reshape(tf.transpose(dKey, perm=[1, 0, 2]), [-1, args.latdim])
		ncKey = tf.nn.embedding_lookup(ncKey, ncs)
		dkey = tf.nn.embedding_lookup(dKey, drugs)
		scores = self.label(ncKey, dkey, ncEmbed, dHyper)
		_preds = tf.reduce_sum(tf.nn.embedding_lookup(ncEmbed0, ncs) * tf.nn.embedding_lookup(dEmbed0, drugs), axis=-1)

		self.pck_preds = _preds
		self.pck_labels = scores

		halfNum = tf.shape(scores)[0] // 2
		fstScores = tf.slice(scores, [0], [halfNum])
		scdScores = tf.slice(scores, [halfNum], [-1])
		fstPreds = tf.slice(_preds, [0], [halfNum])
		scdPreds = tf.slice(_preds, [halfNum], [-1])
		sslLoss = tf.reduce_sum(tf.maximum(0.0, 1.0 - (fstPreds - scdPreds) * args.mult * (fstScores - scdScores)))

		return preds, sslLoss, nclat, dlat

	def tstPred(self, nclat, dlat):
		pcknclat = tf.nn.embedding_lookup(nclat, self.ncids)
		allPreds = pcknclat @ tf.transpose(dlat)
		allPreds = allPreds * (1 - self.trnPosMask) - self.trnPosMask * 1e8
		vals, locs = tf.nn.top_k(allPreds, args.shoot)
		return locs

	def prepareModel(self):
		self.keepRate = tf.placeholder(dtype=tf.float32, shape=[])
		NNs.leaky = args.leaky
		self.actFunc = 'leakyRelu'
		adj = self.handler.trnMat
		idx, data, shape = transToLsts(adj, norm=True)
		self.adj = tf.sparse.SparseTensor(idx, data, shape)

		idx, data, shape = transToLsts(transpose(adj), norm=True)
		self.tpAdj = tf.sparse.SparseTensor(idx, data, shape)


		self.ncids = tf.placeholder(name='ncids', dtype=tf.int32, shape=[None])
		self.dids = tf.placeholder(name='dids', dtype=tf.int32, shape=[None])
		self.edgeids = tf.placeholder(name='edgeids', dtype=tf.int32, shape=[None])
		self.trnPosMask = tf.placeholder(name='trnPosMask', dtype=tf.float32, shape=[None, args.drug])

		self.preds, sslLoss, nclat, dlat = self.ours()
		self.topLocs = self.tstPred(nclat, dlat)

		sampNum = tf.shape(self.ncids)[0] // 2
		posPred = tf.slice(self.preds, [0], [sampNum])
		negPred = tf.slice(self.preds, [sampNum], [-1])
		self.preLoss = tf.reduce_sum(tf.maximum(0.0, 1.0 - (posPred - negPred))) / args.batch
		self.regLoss = args.reg * Regularize()
		self.sslLoss = args.ssl_reg * sslLoss
		self.loss = self.preLoss + self.regLoss + self.sslLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def sampleTrainBatch(self, batIds, labelMat):
		temLabel = labelMat[batIds].toarray()
		batch = len(batIds)
		temlen = batch * 2 * args.sampNum
		ncLocs = [None] * temlen
		dLocs = [None] * temlen
		cur = 0
		for i in range(batch):
			posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
			sampNum = min(args.sampNum, len(posset))
			if sampNum == 0:
				poslocs = [np.random.choice(args.drug)]
				neglocs = [poslocs[0]]
			else:
				poslocs = np.random.choice(posset, sampNum)
				neglocs = negSamp(temLabel[i], sampNum, args.drug)
			for j in range(sampNum):
				posloc = poslocs[j]
				negloc = neglocs[j]
				ncLocs[cur] = ncLocs[cur+temlen//2] = batIds[i]
				dLocs[cur] = posloc
				dLocs[cur+temlen//2] = negloc
				cur += 1
		ncLocs = ncLocs[:cur] + ncLocs[temlen//2: temlen//2 + cur]
		dLocs = dLocs[:cur] + dLocs[temlen//2: temlen//2 + cur]

		edgeSampNum = int(args.edgeSampRate * args.edgeNum)
		if edgeSampNum % 2 == 1:
			edgeSampNum += 1
		edgeids = np.random.choice(args.edgeNum, edgeSampNum)
		return ncLocs, dLocs, edgeids

	def trainEpoch(self):
		num = args.nc
		sfIds = np.random.permutation(num)[:args.trnNum]
		epochLoss, epochPreLoss, epochsslLoss = [0] * 3
		num = len(sfIds)
		steps = int(np.ceil(num / args.batch))

		for i in range(steps):
			st = i * args.batch
			ed = min((i+1) * args.batch, num)
			batIds = sfIds[st: ed]

			target = [self.optimizer, self.preLoss, self.regLoss, self.loss, self.sslLoss, self.iniNorm, self.gcnNorm]
			feed_dict = {}
			ncLocs, dLocs, edgeids = self.sampleTrainBatch(batIds, self.handler.trnMat)
			feed_dict[self.ncids] = ncLocs
			feed_dict[self.dids] = dLocs
			feed_dict[self.keepRate] = args.keepRate
			feed_dict[self.edgeids] = edgeids

			res = self.sess.run(target, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

			preLoss, regLoss, loss, sslLoss, iniNorm, gcnNorm = res[1:]

			epochLoss += loss
			epochPreLoss += preLoss
			epochsslLoss += sslLoss
			log('Step %d/%d: loss = %.2f, regLoss = %.2f, sslLoss = %.2f         ' % (i, steps, loss, regLoss, sslLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epochLoss / steps
		ret['preLoss'] = epochPreLoss / steps
		ret['sslLoss'] = epochsslLoss / steps
		return ret


	def calcRes(self, topLocs, tstLocs, batIds):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = 0
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.shoot))])
			recall = dcg = 0
			for val in temTstLocs:
				if val in temTopLocs:
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
			recall = recall / tstNum
			ndcg = dcg / maxDcg
			allRecall += recall
			allNdcg += ndcg
		return allRecall, allNdcg
	
	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + args.save_path)
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(sess, 'Models/' + args.load_model)
		with open('History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')	

if __name__ == '__main__':
	logger.saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	with tf.Session(config=config) as sess:
		recom = Recommender(sess, handler)
		recom.run()