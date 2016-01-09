#coding=utf-8


import math


# 多项式的值
def polyValue(x, coeff):
	y = 0
	t = 1
	for i in xrange(0, len(coeff)):
		y = y + t*coeff[i]
		t = t * x
	return y


# 一元二次方程的解
def solveQuadraticEquation(coeff1, coeff2):
	if len(coeff1) != 3 or len(coeff2) != 3:
		print 'solveQuadraticEquation() >> coeff1或coeff2的长度 != 2'
		return None, None
	[a, b, c] = [ (coeff1[i]-coeff2[i]) for i in range(2,-1,-1) ]
	if abs(a) < 1e-10:
		print 'solveQuadraticEquation() >> 方程无解'
		return None, None
	p = math.sqrt( (b*b - 4*a*c) / (4*a*a) )
	q = -0.5 * b / a
	return q-p, q+p


# 两个抛物线的交点
def parabolaCrossPoint(coeff1, coeff2):
	x1, x2 = solveQuadraticEquation(coeff1, coeff2)
	if x1 == None or x2 == None:
		return None, None, None, None
	y1, y2 = polyValue(x1, coeff1), polyValue(x2, coeff2)
	return x1, y1, x2, y2