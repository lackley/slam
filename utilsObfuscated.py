from __future__ import division
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkyFJ=False
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkySF=abs
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkySJ=min
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkyJF=tuple
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkyJS=len
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFyS=int
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFyJ=open
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFSy=float
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFSJ=buffer
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFJy=AttributeError
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFJS=ValueError
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkSyF=range
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkSyJ=map
import re
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkSJ=re.search
import numpy as np
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkJy=np.array
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFSkJ=np.unique
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFSyJ=np.exp
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJky=np.ones
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFSJk=np.loadtxt
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFSJy=np.max
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFSyk=np.transpose
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkJS=np.linalg
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJyS=np.concatenate
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFSky=np.random
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJyk=np.frombuffer
from numpy import dot
from numpy import identity 
from scipy.stats import norm as gaussian
import math
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJkS=math.pi
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJSk=math.sqrt
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJSy=math.pow
import sys
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkyFS=sys.float_info
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyFkS=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkyFS.max
gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyFkJ=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkyFJ
def calculate_likelihood_probability(measurement,predicted_measurement,covariance):
 distance,gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyFSk=measurement
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyFSJ,gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyFJk=predicted_measurement
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyFJS=(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyFSk-gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyFJk)%(2*gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJkS)
 if gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyFJS>gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJkS:
  gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyFJS=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyFJS-2*gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJkS
 return multivariate_gauss_prob((distance,gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyFJS),(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyFSJ,0),covariance) 
def multivariate_gauss_prob(observed,gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykFS,covariance):
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykFS=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkJy(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykFS)
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykFJ=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkJy(observed)
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykSF=covariance
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykSJ=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkJS.det(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykSF)
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykJF=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkJS.inv(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykSF)
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykJS=-0.5*gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFSyk(observed-gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykFS).dot(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykJF).dot(observed-gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykFS)
 if gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykSJ<0:
  gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykSJ=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkySF(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykSJ) 
 return gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkySJ((2*gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJkS)**-1*gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJSy(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykSJ,-0.5)*gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFSyJ(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykJS),gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyFkS)
def distance(p1,p2):
 return gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJSk((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
def calculate_jacobian(robot_position,landmark_pos):
 t=robot_position 
 j=landmark_pos
 q=(j[0]-t[0])**2+(j[1]-t[1])**2 
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySFk=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkJy([[(j[0]-t[0])/gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJSk(q),(j[1]-t[1])/gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJSk(q)],[-(j[1]-t[1])/q,(j[0]-t[0])/q]])
 return gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySFk
def compute_measurement_covariance(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySFk,oldCovariance,sigmaObservation):
 Q=dot(dot(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySFk,oldCovariance),gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFSyk(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySFk))+sigmaObservation
 return Q
def compute_initial_covariance(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySFk,sigmaObservation):
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySFJ=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkJS.inv(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySFk)
 S=dot(dot(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySFJ,sigmaObservation),gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFSyk(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySFJ))
 return S
def compute_kalman_gain(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySFk,oldCovariance,measurementCovariance):
 K=dot(dot(oldCovariance,gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFSyk(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySFk)),gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkJS.inv(measurementCovariance))
 return K
def compute_new_landmark(z,gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySkF,kalmanGain,old_landmark):
 z=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkJy(z)
 z[1]=z[1]
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySkF=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkJy(z)
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySkF[1]=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySkF[1]
 d=z-gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySkF
 d[1]=d[1]%(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJkS*2)
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySkJ=old_landmark+dot(kalmanGain,d)
 return gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkyJF(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySkJ)
def compute_new_covariance(kalmanGain,gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySFk,oldCovariance):
 I=identity(2)
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySJF=I-dot(kalmanGain,gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySFk)
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySJk=dot(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySJF,oldCovariance)
 return gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNySJk
def gauss_sample(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykFS,covariance):
 return gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFSky.multivariate_normal(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNykFS,covariance)
"""
The following are helper functions and not necessary to understand the project
"""
def load_map(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJkF):
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJFk=read_pgm(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJkF,byteorder='<')
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJFS=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFSkJ(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJFk)
 if(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkyJS(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJFS)==3):
  gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJFk=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJFk!=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFSJy(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJFS)
  gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJFk=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkJy(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJFk,dtype=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFyS,ndmin=2)
 elif(not(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkyJS(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJFS)==2 and 0 in gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJFS and 1 in gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJFS)):
  gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJFk=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJFk>150
  gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJFk=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkJy(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJFk,dtype=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFyS,ndmin=2)
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJFk=fulltrim(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJFk)
 return gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJFk
def load_csv(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJkF):
 return gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFSJk(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJkF,delimiter=",",ndmin=2)
def load_measurements(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJkF):
 with gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFyJ(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJkF)as f:
  gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJSF=f.readlines()
  gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJSk=[]
  for gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFykS in gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJSF:
   gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFykS=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFykS.strip()
   ms=[]
   for v in gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFykS.split(","):
    if gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkyJS(v)>0:
     ms.append([gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFSy(x)for x in v.split(":")])
   gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJSk.append(ms)
  return gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNyJSk
def read_pgm(filename,byteorder='>'):
 with gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFyJ(filename,'rb')as f:
  gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFSJ=f.read()
 try:
  gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFykJ,gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFySk,gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFySJ,gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFyJk=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkSJ(b"(^P5\s(?:\s*#.*[\r\n])*" b"(\d+)\s(?:\s*#.*[\r\n])*" b"(\d+)\s(?:\s*#.*[\r\n])*" b"(\d+)\s(?:\s*#.*[\r\n]\s)*)",gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFSJ).groups()
 except gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFJy:
  raise gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFJS("Not a raw PGM file: '%s'"%filename)
 return gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJyk(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFSJ,dtype='u1' if gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFyS(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFyJk)<256 else byteorder+'u2',count=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFyS(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFySk)*gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFyS(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFySJ),offset=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkyJS(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFykJ)).reshape((gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFyS(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFySJ),gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkFyS(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFySk)))
def trim(figure):
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkyS=figure
 for i in gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkSyF(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkyJS(figure)):
  if 0 in figure[i]:
   gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkyS=figure[i:]
   break
 l=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkyJS(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkyS)-1
 for i in gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkSyF(l+1):
  if 0 in gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkyS[(l-i)]:
   gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkyS=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkyS[:(l-i)]
   break
 return gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkyS
def fulltrim(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkSyJ):
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkSyJ=trim(trim(gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkSyJ.T).T)
 h,w=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkSyJ.shape
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkyJ=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJyS([gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJky((1,w)),gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNkSyJ,gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJky((1,w))],axis=0)
 gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkSy=gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJyS([gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJky((h+2,1)),gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkyJ,gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFJky((h+2,1))],axis=1)
 return gcDIiGwlhAuYOXxVUbMmfqpdTBPnjzWrQovRKLCHtaEseNFkSy
# Created by pyminifier (https://github.com/liftoff/pyminifier)

