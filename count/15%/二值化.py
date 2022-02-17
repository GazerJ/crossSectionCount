# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 14:57:44 2021

@author: 94917
"""

from sklearn.cluster import KMeans
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
def getimg():
    return Image.open("./15%CT截面1.jpg")
def showimg(img):
    plt.imshow(img)
    plt.show()
    
im=getimg()
im=np.array(im.convert("L"))
im=np.where(im[...,:]<100,0,255)
showimg(Image.fromarray(im))

where=[]
for i in range(len(im[:,0])):
    for j in range(len(im[0,:])):
        if im[i,j]==0:
            where.append([i,j])

from sklearn.cluster import DBSCAN
dbscan=DBSCAN(1.5,min_samples=1)



dbscan.fit(where)
labels=dbscan.labels_
where=np.array([where]+[labels]).T

aaxx=[]
for j in range(max(labels)):
    ax=[]
    for i in range(len(where[:,0])):
        if where[i,1]==j:
            ax.append(where[i,0])
    ax=np.array(ax)
    barx=ax[:,0].mean()
    bary=ax[:,1].mean()         
    aaxx.append([j,barx,bary])
aaxx=np.array(aaxx)
plt.scatter(aaxx[:,1],aaxx[:,2],s=1)
plt.show()

d=np.zeros((max(labels),max(labels)))
for i in range(max(labels)):
    for j in range(max(labels)):
        d[i,j]=np.sqrt((aaxx[i,1]-aaxx[j,1])**2+(aaxx[i,2]-aaxx[j,2])**2)


dmin=np.zeros(max(labels))
dmin[0]=d[0,1:].min()
for i in range(1,max(labels)-1):
    dmin[i]=min(d[i,:i].min(),d[i,i+1:].min())
dmin[-1]=d[0,:-2].min()


dmin=np.array(dmin)/dmin.mean()*10.77535
from jplt import distribution
distribution(dmin,delta=1,d=1.5,u=dmin.mean(),sig=dmin.std())

plt.bar(dmin)
plt.show()


np.savetxt("15%.txt",dmin)
plt.plot(dmin)
plt.show()
print('std:',dmin.std())
print('var:',dmin.var())
print('mean:',dmin.mean())
