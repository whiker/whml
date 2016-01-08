#coding=utf-8


import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf

import data_file as df


def readPoints(filename):
	f = open(filename, 'r')
	
	line = f.readline()
	if not line:
		print "%s文件, 空数据" %(filename)
		return -1, 0, 0, 0, 0
	lines = line.strip().split('\t')
	if len(lines) != 2:
		print "%s文件, 首行格式错误: %s" %(filename, line)
		return -1, 0, 0, 0, 0
	nRow = int(lines[0])
	nCol = int(lines[1])
	ptx = []
	pty = []
	
	for i in xrange(0, nRow):
		line = f.readline()
		if not line:
			print "行不足: %d<%d" %(i, nRow)
			return -2, nRow, nCol, ptx, pty
		lines = line.strip().split('\t')
		if len(lines) != 2*nCol:
			print "第%d行,列不足: %d<%d" %(i, len(lines), 2*nCol)
			return -1, nRow, nCol, ptx, pty
		
		x = []
		y = []
		for j in xrange(0, nCol):
			x.append(int(lines[2*j]))
			y.append(-int(lines[2*j+1]))
		ptx.append(x)
		pty.append(y)
	
	f.close()
	return 0, nRow, nCol, ptx, pty


def normal(data):  #每列是一个样本数据, 有多少列就有多少个样本
	shift = np.min(data, 1)  #计算每行(各维度)的最小值
	scale = np.max(data, 1) - np.min(data, 1)
	scale *= 0.5
	shift += scale
	scale /= 2  #映射到[-2, 2]
	for i in xrange(1, len(data)):  #对每行处理
		data[i] = (data[i] - shift[i]) / scale[i]
	return shift, scale


def adjustWeight(shift, scale, weight):
	w0 = weight[0]
	for i in xrange(1, len(weight)):
		weight[i] /= scale[i]
		w0 -= (weight[i] * shift[i])
	weight[0] = w0
	return weight


def tfTrain(data_x, data_y):
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
	for iter in xrange(0, 1000):
		sess.run(train)
		#print tf_w.eval(sess)[0]
	
	# 结果
	weight = tf_w.eval(sess)[0]
	return adjustWeight(shift, scale, weight)


def fitPoly(x, y, nPoly):
	if nPoly < 1:
		print "多项式次数错误: %d<1" %(nPoly)
		return
	x = np.float32(x)
	
	data_x = []
	for i in xrange(len(x)):
		px = [1, x[i]]
		for j in xrange(2, nPoly+1):
			px.append( px[j-1] * px[1] )
		data_x.append(px)
	data_x = np.transpose(data_x)  #每列是一个样本
	
	data_x = np.array(np.float32(data_x))
	data_y = np.array(np.float32(y))
	return tfTrain(data_x, data_y)
	

def calcPolyValue(x, coeff):
	y = 0
	t = 1
	for i in xrange(0, len(coeff)):
		y = y + t*coeff[i]
		t = t * x
	return y


def plotFitResult(x, y, coeff, xySwap):
	tx = range(np.min(x)-10, np.max(x)+10)
	ty = []
	for i in xrange(0, len(tx)):
		ty.append( calcPolyValue(tx[i], coeff) )
	if not xySwap:
		plt.plot(x, y, 'b.')
		plt.plot(tx, ty, 'r')
	else:
		plt.plot(y, x, 'b.')
		plt.plot(ty, tx, 'r')


if __name__ == '__main__':
	# 读取数据
	ret, nRow, nCol, ptx, pty = readPoints('all_pts.txt')
	if ret != 0:
		sys.exit(1)
	
	nPoly = 2
	
	rowCoeff = []
	for row in xrange(nRow):
		coeff = fitPoly(ptx[row], pty[row], nPoly)
		plotFitResult(ptx[row], pty[row], coeff, False)
		rowCoeff.append(coeff)
		print row
	df.saveMatrix(rowCoeff, "row_coeff.txt")
	
	colCoeff = []
	ptx = np.transpose(ptx)
	pty = np.transpose(pty)
	for col in xrange(nCol):
		coeff = fitPoly(pty[col], ptx[col], nPoly)
		plotFitResult(pty[col], ptx[col], coeff, True)
		colCoeff.append(coeff)
		print col
	df.saveMatrix(colCoeff, "col_coeff.txt")
	
	plt.xlim(50, 650)
	plt.ylim(-450, -200)
	plt.show()