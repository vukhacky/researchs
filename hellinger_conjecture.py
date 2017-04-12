#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:34:42 2017

@author: vu
"""
##############################################################################
''' Numerical simulations for visualizing the Hellinger conjecture
'''
##############################################################################
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# Single-variable Kullback–Leibler divergence
def single_diverg(x,y):
    if y == 0 or y == 1 or x == 0 or x == 1:
        return 0
    else:
        return x * np.log(x/y) + (1-x) * np.log((1-x)/(1-y))

def h_func(x):
    return np.sqrt(1 - x**2)

# The left-hand side of Hellinger conjecture
def LHS_conjecture(s,c,d):
    p = 1 - 2*d
    q = 2*c - 1
    return h_func(s*p + (1-s)*q)- s * h_func(p) - (1-s) * h_func(q)

# The right-hand side of Hellinger conjecture
def RHS_conjecture(s,c,d):
    s1 = 1 - s
    c1 = 1 - c
    d1 = 1 - d
    up_term = single_diverg(s*c+s1*d1, s*d +s1*c1)
    down_term = s * single_diverg(c,d) + s1 * single_diverg(d1,c1)
    if down_term ==0:
        return 0
    else:
        return 1 - np.sqrt(up_term/down_term)

# Hellinger conjecture: this function is positive for all s,c,d in [0,1]    
def Hellinger_conjecture(s,c,d):
    return RHS_conjecture(s,c,d) - LHS_conjecture(s,c,d)

c =0.3
d = 0.8
S = np.arange(0.1, 0.9, 0.001)
conj = [Hellinger_conjecture(s,c,d) for s in S]
plt.plot(S, conj)

#RHS_conj = [Hellinger_conjecture(s,c,d)/ RHS_conjecture(s,c,d) for s in S]
#LHS_conj = [LHS_conjecture(s,c,d) for s in S]
#plt.plot(S, LHS_conj)

RHS_conj = [LHS_conjecture(s,c,d)/ RHS_conjecture(s,c,d) for s in S]
plt.plot(S, RHS_conj)

##############################################################################
''' 
Conjecture:
f(s) = LHS_conjecture(s,c,d)/ RHS_conjecture(s,c,d)
is a concave function of s, when c and d are fixed.
Idea:
Take automatic differentiations to estimate the second derivative f''(s)
If it is much smaller than 0, we can expect to prove formally that 
f''(s) < 0
easily by interval computations.
Then it is left to prove that
f(s*) < 1 for s* be the unique s such that f'(s) = 0.
'''

##############################################################################
## Symbolic computation
from sympy import *

s, c, d = symbols('s c d')
# rewrite functions, because numpy functions can't be used in sympy
def single_diverg(x,y):
    return x * log(x/y) + (1-x) * log((1-x)/(1-y))

def h_func(x):
    return sqrt(1 - x**2)

def LHS_conjecture(s,c,d):
    p = 1 - 2*d
    q = 2*c - 1
    return h_func(s*p + (1-s)*q)- s * h_func(p) - (1-s) * h_func(q)

def RHS_conjecture(s,c,d):
    s1 = 1 - s
    c1 = 1 - c
    d1 = 1 - d
    up_term = single_diverg(s*c+s1*d1, s*d +s1*c1)
    down_term = s * single_diverg(c,d) + s1 * single_diverg(d1,c1)
    return 1 - sqrt(up_term/down_term)

def Hellinger_conjecture(s,c,d):
    return LHS_conjecture(s,c,d) / RHS_conjecture(s,c,d)

####################
expr = Hellinger_conjecture(s,c,d)
second_deriv = diff(expr, s, s)
second_deriv.evalf(subs={s:0.2, c:0.3, d:0.4})

## Plots
S = np.arange(0.1, 0.9, 0.001)
concave_conj = [second_deriv.evalf(subs={s:t, c:0.3, d:0.4}) for t in S]
plt.plot(S, concave_conj)





