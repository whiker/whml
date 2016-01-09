#coding=utf-8


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import point_file as ptfile
import data_file as dfile


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
	for iter in xrange(0, 5000):
		sess.run(train)
		#print tf_w.eval(sess)[0]
	
	# 结果
	weight = tf_w.eval(sess)[0]
	return adjustWeight(shift, scale, weight)


# 拟合多项式
def fitPoly(x, y, nPoly):
	if nPoly < 1:
		print "fitPoly() >> 多项式次数错误: %d<1" %(nPoly)
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


def plotFitResult(x, y, coeff, flag=None, color='b'):
	if len(x) < 1:
		print 'plotFitResult() >> 空数据'
		return
	if len(y) != len(x):
		print 'plotFitResult() >> x和y的维度不一致'
		return
	limit1 = int(np.min(x)) - 10
	limit2 = int(np.max(x)) + 10
	tx = range(limit1, limit2)
	ty = []
	for i in xrange(0, len(tx)):
		ty.append( calcPolyValue(tx[i], coeff) )
	if flag == None:
		plt.plot(x, y, color+'.')
		plt.plot(tx, ty, 'r')
	else:
		x1 = [-x[i] for i in xrange(len(x))]
		tx1 = [-tx[i] for i in xrange(len(tx))]
		plt.plot(y, x1, color+'.')
		plt.plot(ty, tx1, 'r')


# 计算行抛物线与每个列抛物线的交点
def calcRowCrossPoints(coeff, colCoeff, xMin, xMax, step=0.05):
	ptx = []
	x = np.linspace(xMin, xMax, 1+(xMax-xMin)/step)
	col = 0
	errPrev = float('inf')
	for i in xrange(len(x)):
		y = calcPolyValue(x[i], coeff)
		err = abs(calcPolyValue(-y, colCoeff[col]) - x[i])
		if err < errPrev:
			errPrev = err
		else:
			ptx.append(x[i-1])
			col += 1
			if col == len(colCoeff):
				break
			errPrev = abs(calcPolyValue(y, colCoeff[col]) + x[i])
	return ptx


def calcCoeff():
	ret, nRow, nCol, ptx, pty = ptfile.loadPoints('data/all_pts.txt')
	if ret != 0:
		print 'calcCoeff() >> 读取数据失败'
		return
	
	nPoly = 2  # 抛物线
	rowCoeff = []
	colCoeff = []
	
	for i in xrange(nRow):
		coeff = fitPoly(ptx[i], pty[i], nPoly)
		plotFitResult(ptx[i], pty[i], coeff)
		rowCoeff.append(coeff)
		print i
	dfile.saveMatrix(rowCoeff, 'data/row_coeff.txt')
	
	ptx1 = -np.transpose(pty)
	pty1 = np.transpose(ptx)
	for i in xrange(nCol):
		coeff = fitPoly(ptx1[i], pty1[i], nPoly)
		plotFitResult(ptx1[i], pty1[i], coeff, 1)
		colCoeff.append(coeff)
		print i
	dfile.saveMatrix(colCoeff, 'data/col_coeff.txt')
	
	plt.xlim(50, 650)
	plt.ylim(-400, -150)
	plt.show()


def calcTruePoints():
	ret, rowCoeff, nRow, _ = dfile.loadMatrix('data/row_coeff.txt', float)
	ret, colCoeff, nCol, _ = dfile.loadMatrix('data/col_coeff.txt', float)
	
	xLimit = [50, 650]
	ptxs = []
	ptys = []
	
	for i in xrange(nRow):
		ptx = calcRowCrossPoints(rowCoeff[i], colCoeff, xLimit[0], xLimit[1])
		if len(ptx) != nCol:
			print "calcTruePoints() >> 第%d行的交叉点数: %d<%d" %(i, len(ptx), nCol)
		pty = [calcPolyValue(x, rowCoeff[i]) for x in ptx]
		plotFitResult(ptx, pty, rowCoeff[i], None, 'g')
		ptxs.append(ptx)
		ptys.append(pty)
	ptfile.savePoints(ptxs, ptys, 'data/true_pts.txt')
	
	ptxs1 = -np.transpose(ptys)
	ptys1 = np.transpose(ptxs)
	for i in xrange(nCol):
		plotFitResult(ptxs1[i], ptys1[i], colCoeff[i], 1, 'g')
	
	plt.xlim(50, 650)
	plt.ylim(-400, -150)
	plt.show()


if __name__ == '__main__':
	#calcCoeff()
	calcTruePoints()