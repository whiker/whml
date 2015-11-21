#coding=utf-8

import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import random as rand


# 全局变量
IterNum = 30
alpha = 0.9
TrueTheta = [ 1, 5, 100, 7 ]
Xmin = [ 0, 0, 0 ]
Xmax = [ 100, 30, 200 ]
Hrand = 50
Nsample = 200


# 函数
def randRange(dmin, dmax):
	return ( dmin + rand.random() * (dmax-dmin) )

def show(x_mat, y_vec, theta, errors):
	if len(x_mat) == 2:
		plt.plot(x_mat[1], y_vec, 'b*')
		x1 = [ 1, Xmin[0] ]
		x2 = [ 1, Xmax[0] ]
		x = np.array([x1, x2])
		y = np.dot(x, theta)
		plt.plot([x1[1], x2[1]], y, 'r')
		plt.xlim(Xmin[0], Xmax[0])
		plt.show()
	plt.plot(range(IterNum), errors, 'b*')
	plt.plot(range(IterNum), errors, 'r')
	plt.xlim(0, IterNum)
	plt.show()

def produceOriginalTrainData():
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
	return np.array(x_mat), np.array(y_vec)

# 把特征的维度取值区间映射到[-2, 2]
def adjustScale(x_mat, y_vec):
	x_shift = np.min(x_mat, 1)  # 计算每行(各维度)的最小值
	x_scale = np.max(x_mat, 1) - np.min(x_mat, 1)
	x_scale *= 0.5
	x_shift += x_scale
	x_scale /= 2
	n = len(x_mat)
	for i in xrange(1,n):
		x_mat[i] = (x_mat[i] - x_shift[i]) / x_scale[i]
	return x_shift, x_scale

def calcErrVec(x_mat, y_vec, theta):
	err_vec = np.dot(theta, x_mat) - y_vec
	error = math.sqrt( np.dot(err_vec, err_vec) / Nsample )
	print error
	return err_vec, error

def gradientDescent(x_mat, y_vec, theta, errors):
	err_vec, error = calcErrVec(x_mat, y_vec, theta)
	errors.append(error)
	d_theta_vec = np.dot(x_mat, err_vec)
	d_theta_vec *= (alpha / Nsample)
	theta -= d_theta_vec


if __name__ == "__main__":
	#训练数据
	x_mat, y_vec = produceOriginalTrainData()
	x_mat_ori = copy.deepcopy(x_mat)
	x_shift, x_scale = adjustScale(x_mat, y_vec)
	print x_mat[1]
	
	# 初始值
	theta = np.zeros(len(TrueTheta))
	errors = []
	
	# 迭代
	for i in xrange(IterNum):
		gradientDescent(x_mat, y_vec, theta, errors)
	
	calcErrVec(x_mat, y_vec, theta)
	
	# '''
	t = theta[0]
	for i in xrange(1, len(TrueTheta)):
		theta[i] /= x_scale[i]
		t -= (theta[i] * x_shift[i])
	theta[0] = t
	# '''
	
	print theta
	show(x_mat_ori, y_vec, theta, errors)