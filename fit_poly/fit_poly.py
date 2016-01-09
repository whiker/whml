#coding=utf-8


import matplotlib.pyplot as plt
import numpy as np

import data_file as dfile
import point_file as ptfile
import tensorflow_util as tfu
import util


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
	return tfu.tfTrain(data_x, data_y, 5000)


def plotFitResult(x, y, coeff, flag=None, color='b', limit1=None, limit2=None):
	if len(x) < 1:
		print 'plotFitResult() >> 空数据'
		return
	if len(y) != len(x):
		print 'plotFitResult() >> x和y的维度不一致'
		return
	if limit1 == None:
		limit1 = int(np.min(x)) - 10
	if limit2 == None:
		limit2 = int(np.max(x)) + 10
	tx = range(limit1, limit2)
	ty = []
	for i in xrange(0, len(tx)):
		ty.append( util.polyValue(tx[i], coeff) )
	if flag == None:
		plt.plot(x, y, color+'.')
		plt.plot(tx, ty, 'r')
	else:
		x1 = [-x[i] for i in xrange(len(x))]
		tx1 = [-tx[i] for i in xrange(len(tx))]
		plt.plot(y, x1, color+'.')
		plt.plot(ty, tx1, 'r')


# 行抛物线与每个列抛物线的交点
def rowCrossPoints(coeff, colCoeff, xMin, xMax, step=0.05):
	ptx = []
	x = np.linspace(xMin, xMax, 1+(xMax-xMin)/step)
	col = 0
	errPrev = float('inf')
	for i in xrange(len(x)):
		y = util.polyValue(x[i], coeff)
		err = abs(util.polyValue(-y, colCoeff[col]) - x[i])
		if err < errPrev:
			errPrev = err
		else:
			ptx.append(x[i-1])
			col += 1
			if col == len(colCoeff):
				break
			errPrev = abs(util.polyValue(y, colCoeff[col]) + x[i])
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


def truePoints():
	ret1, rowCoeff, nRow, _ = dfile.loadMatrix('data/row_coeff.txt', float)
	ret2, colCoeff, nCol, _ = dfile.loadMatrix('data/col_coeff.txt', float)
	if ret1 != 0 or ret2 != 0:
		print 'truePoints() >> load coeff失败'
		return
	
	xLimit = [50, 650]
	ptxs = []
	ptys = []
	
	for i in xrange(nRow):
		ptx = rowCrossPoints(rowCoeff[i], colCoeff, xLimit[0], xLimit[1])
		if len(ptx) != nCol:
			print "truePoints() >> 第%d行的交叉点数: %d<%d" %(i, len(ptx), nCol)
		pty = [util.polyValue(x, rowCoeff[i]) for x in ptx]
		plotFitResult(ptx, pty, rowCoeff[i], None, 'g', -50, 850)
		ptxs.append(ptx)
		ptys.append(pty)
	ptfile.savePoints(ptxs, ptys, 'data/true_pts.txt')
	
	ptxs1 = -np.transpose(ptys)
	ptys1 = np.transpose(ptxs)
	for i in xrange(nCol):
		plotFitResult(ptxs1[i], ptys1[i], colCoeff[i], 1, 'g', -150, 850)
	
	plt.xlim(-50, 850)
	#plt.ylim(-400, -150)
	plt.show()


if __name__ == '__main__':
	#calcCoeff()
	truePoints()