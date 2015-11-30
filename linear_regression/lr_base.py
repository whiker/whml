#coding=utf-8


import matplotlib.pyplot as plt
import numpy as np
import random as rand


TrueTheta = [ 2, 5, 100, 7 ]
Xmin = [ 0, 0, 0 ]
Xmax = [ 100, 30, 200 ]
Hrand = 50
Nsample = 200


# 产生单个随机数
def randRange(dmin, dmax):
	return ( dmin + rand.random() * (dmax-dmin) )


# 生成随机样本
def produceSampleData():
	x_mat = []
	y_vec = []
	n = len(Xmin)
	for i in xrange(Nsample):
		x = [1]
		for j in xrange(n):
			x.append(randRange(Xmin[j], Xmax[j]))
		h = np.dot(TrueTheta, x) + randRange(-Hrand, Hrand)
		x_mat.append(x)
		y_vec.append(h)
	x_mat = np.transpose(x_mat)  # 每列是一个样本
	x_mat = np.float32(x_mat)
	y_vec = np.float32(y_vec)
	return np.array(x_mat), np.array(y_vec)


# 把样本取值区间映射到[-2, 2]
def adjustSampleScale(x_mat):
	x_shift = np.min(x_mat, 1)  # 计算每行(各维度)的最小值
	x_scale = np.max(x_mat, 1) - np.min(x_mat, 1)
	x_scale *= 0.5
	x_shift += x_scale
	x_scale /= 2
	n = len(x_mat)
	for i in xrange(1,n):
		x_mat[i] = (x_mat[i] - x_shift[i]) / x_scale[i]
	return x_shift, x_scale


# 调整结果 theta
def adjustTheta(x_shift, x_scale, theta):
	t = theta[0]
	for i in xrange(1, len(theta)):
		theta[i] /= x_scale[i]
		t -= (theta[i] * x_shift[i])
	theta[0] = t


# 显示回归结果
def showLrLine(x_mat, y_vec, theta):
	if len(x_mat) != 2:
		return
	
	# 绘制原始数据
	plt.plot(x_mat[1], y_vec, 'b*')
	# 绘制回归线
	x1 = [ 1, Xmin[0] ]
	x2 = [ 1, Xmax[0] ]
	x = np.array([x1, x2])
	y = np.dot(x, theta)
	plt.plot([x1[1], x2[1]], y, 'r')
	
	plt.xlim(Xmin[0], Xmax[0])
	plt.show()