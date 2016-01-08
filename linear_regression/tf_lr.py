#coding=utf-8


import copy
import tensorflow as tf

import lr_base


def tfTrain(data_x, data_y):
	# 构造线性模型
	tf_w = tf.Variable( tf.zeros([1, len(data_x)]) )
	tf_y = tf.matmul(tf_w, data_x)
	loss = tf.reduce_mean( tf.square( tf_y - data_y ) )
	optimizer = tf.train.GradientDescentOptimizer(0.5)
	train = optimizer.minimize(loss)
	
	# 训练
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	for iter in xrange(0, 50):
		sess.run(train)
	
	# 结果
	theta = tf_w.eval(sess)[0]
	return theta


if __name__ == '__main__':
	# 产生数据
	x_mat, y_vec = lr_base.produceSampleData()
	x_mat_ori = copy.deepcopy(x_mat)
	x_shift, x_scale = lr_base.adjustSampleScale(x_mat)
	
	theta = tfTrain(x_mat, y_vec)
	lr_base.adjustTheta(x_shift, x_scale, theta)
	print theta