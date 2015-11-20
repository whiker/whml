#coding=utf-8

import math
import numpy as np
import random as rand


# 全局变量
IterNum = 30
alpha = 0.05
TrueTheta = [ 1, 2 ]
Xmin = [ 0 ]
Xmax = [ 100 ]
Hrand = 0
Nsample = 40


# 函数
def randRange(dmin, dmax):
	return ( dmin + rand.random() * (dmax-dmin) );

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

def normalize(x_mat, y_vec):
	# 计算每行(各维度)的均值和尺度
	x_mean = np.mean(x_mat, 1)
	x_scale = np.max(x_mat, 1) - np.min(x_mat, 1)
	n = len(x_mat)
	for i in xrange(1,n):
		x_mat[i] = (x_mat[i] - x_mean[i]) / x_scale[i]
	return x_mean, x_scale

def calcErrVec(x_mat, y_vec, theta):
	err_vec = np.dot(theta, x_mat) - y_vec
	print math.sqrt( np.dot(err_vec, err_vec) / Nsample )
	return err_vec

def gradientDescent(x_mat, y_vec, theta):
	err_vec = calcErrVec(x_mat, y_vec, theta)
	d_theta_vec = np.dot(x_mat, err_vec) / Nsample
	d_theta_vec *= alpha
	return (theta - d_theta_vec)


if __name__ == "__main__":
	#训练数据
	x_mat, y_vec = produceOriginalTrainData()
	x_mean, x_scale = normalize(x_mat, y_vec)
	
	# 初始值
	theta = np.zeros(len(TrueTheta))
	
	# 迭代
	for i in xrange(IterNum):
		theta = gradientDescent(x_mat, y_vec, theta)
	
	calcErrVec(x_mat, y_vec, theta)
	print theta