#coding=utf-8


import copy
import tensorflow as tf

import lr_base


if __name__ == '__main__':
	# 产生数据
	x_mat, y_vec = lr_base.produceSampleData()
	x_mat_ori = copy.deepcopy(x_mat)
	x_shift, x_scale = lr_base.adjustSampleScale(x_mat)
	
	# 构造线性模型
	W = tf.Variable( tf.zeros([1, len(x_mat)]) )
	y = tf.matmul(W, x_mat)
	
	loss = tf.reduce_mean( tf.square(y-y_vec) )
	optimizer = tf.train.GradientDescentOptimizer(0.5)
	train = optimizer.minimize(loss)
	
	# tf 初始化
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	
	# 训练
	for iter in xrange(0, 50):
		sess.run(train)
	
	# 结果
	theta = W.eval(sess)[0]
	lr_base.adjustTheta(x_shift, x_scale, theta)
	print theta