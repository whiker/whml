#coding=utf-8


import numpy as np
import tensorflow as tf


# 对数据作归一化
# 每列是一个样本数据, 有多少列就有多少个样本
def normal(data):
	shift = np.min(data, 1)  # 计算每行(各维度)的最小值
	scale = np.max(data, 1) - np.min(data, 1)
	scale *= 0.5
	shift += scale
	scale /= 2  # 映射到[-2, 2]
	for i in xrange(1, len(data)):  # 对每行处理
		data[i] = (data[i] - shift[i]) / scale[i]
	return shift, scale


def adjustWeight(shift, scale, weight):
	w0 = weight[0]
	for i in xrange(1, len(weight)):
		weight[i] /= scale[i]
		w0 -= (weight[i] * shift[i])
	weight[0] = w0
	return weight


# 线性回归
def linearRegression(data_x, data_y, nIter, alpha=0.01, errDisp=None):
	shiftX, scale = normal(data_x)
	shiftY = (np.max(data_y) + np.min(data_y)) * 0.5
	data_y -= shiftY
	
	# 构造线性模型
	tf_w = tf.Variable( tf.zeros([1, len(data_x)]) )
	tf_y = tf.matmul(tf_w, data_x)
	loss = tf.reduce_mean( tf.square( tf_y - data_y ) )  # 损失函数是均方误差
	optimizer = tf.train.GradientDescentOptimizer(alpha)
	train = optimizer.minimize(loss)
	
	# 训练
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	cnt = 0
	for iter in xrange(0, nIter):
		sess.run(train)
		if errDisp != None and iter%errDisp == 0:
			cnt += errDisp
			print cnt, sess.run(loss)
			# print tf_w.eval(sess)[0]
	
	# 结果
	weight = tf_w.eval(sess)[0]
	weight = adjustWeight(shiftX, scale, weight)
	weight[0] += shiftY
	return weight


# 神经网络
def nn(din, dout, hideUnitNum, nIter, alpha=0.01, errDisp=None):
	nLevel = len(hideUnitNum)
	if nLevel < 3:
		print 'nn() >> 层数<3'
		return
	
	nSample = len(din)
	if len(din) < 1 or len(dout) < 1:
		print 'nn() >> 样本数<1'
		return
	if len(din) != len(dout):
		print 'nn() >> 输入和输出的样本数不相等'
		return
	
	if hideUnitNum[0] != len(din[0]) or hideUnitNum[nLevel-1] != len(dout[1]):
		print 'nn() >> 输入或输出的样本的维度与hideUnitNum不匹配'
		return
	
	tf_w = []
	tf_b = []
	for i_level in xrange(1, nLevel):
		nPrev = hideUnitNum[i_level-1]
		nCurr = hideUnitNum[i_level]
		w = tf.Variable( tf.zeros([nPrev, nCurr]) )
		b = tf.Variable( tf.zeros([nCurr]) )
		tf_w.append(w)
		tf_b.append(b)
	
	tf_y = tf.nn.relu(tf.matmul(din, tf_w[0]) + tf_b[0])
	for i in xrange(1, len(tf_w)):
		tf_y = tf.nn.relu( tf.matmul(tf_y, tf_w[i]) + tf_b[i] )
	loss = tf.reduce_mean( tf.square( tf_y - dout ) )
	optimizer = tf.train.GradientDescentOptimizer(0.01)
	train = optimizer.minimize(loss)
	
	# 训练
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	cnt = 0
	for iter in xrange(0, nIter):
		sess.run(train)
		if errDisp != None and iter%errDisp == 0:
			cnt += errDisp
			print cnt, sess.run(loss)