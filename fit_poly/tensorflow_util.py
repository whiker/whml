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


def tfTrain(data_x, data_y, nIter):
	shift, scale = normal(data_x)

	# 构造线性模型
	tf_w = tf.Variable( tf.zeros([1, len(data_x)]) )
	tf_y = tf.matmul(tf_w, data_x)
	loss = tf.reduce_mean( tf.square( tf_y - data_y ) )
	optimizer = tf.train.GradientDescentOptimizer(0.1)
	train = optimizer.minimize(loss)
	
	# 训练
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	for iter in xrange(0, nIter):
		sess.run(train)
		#print tf_w.eval(sess)[0]
	
	# 结果
	weight = tf_w.eval(sess)[0]
	return adjustWeight(shift, scale, weight)