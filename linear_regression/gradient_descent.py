#coding=utf-8

import copy
import math
import matplotlib.pyplot as plt
import numpy as np

import lr_base


IterNum = 30
alpha = 0.9


# 计算均方根误差
def calcErrVec(x_mat, y_vec, theta):
	err_vec = np.dot(theta, x_mat) - y_vec
	error = math.sqrt( np.dot(err_vec, err_vec) / lr_base.Nsample )
	return err_vec, error


# 梯度下降法
def gradientDescent(x_mat, y_vec, theta, errors):
	err_vec, error = calcErrVec(x_mat, y_vec, theta)
	errors.append(error)
	d_theta_vec = np.dot(x_mat, err_vec)
	d_theta_vec *= (alpha / lr_base.Nsample)
	theta -= d_theta_vec


# 显示误差变化
def showError(errors):
	plt.plot(range(IterNum), errors, 'b*')
	plt.plot(range(IterNum), errors, 'r')
	plt.xlim(0, IterNum)
	plt.show()


if __name__ == "__main__":
	# 产生数据
	x_mat, y_vec = lr_base.produceSampleData()
	x_mat_ori = copy.deepcopy(x_mat)
	x_shift, x_scale = lr_base.adjustSampleScale(x_mat)
	
	# 初始值
	theta = np.zeros(len(lr_base.TrueTheta))
	errors = []
	
	# 迭代
	for i in xrange(IterNum):
		gradientDescent(x_mat, y_vec, theta, errors)
	
	calcErrVec(x_mat, y_vec, theta)
	
	# 调整 theta
	lr_base.adjustTheta(x_shift, x_scale, theta)
	print theta
	
	lr_base.showLrLine(x_mat_ori, y_vec, theta)
	showError(errors)