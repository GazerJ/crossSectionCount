# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 18:13:24 2021

@author: 姜高晓
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
def func(x,u,sig):
    return 1/(sig*np.sqrt(2*np.pi))*np.exp(-(x-u)**2/(2*sig**2))
def distribution(a,d=0.01,x='d',P='P',delta=0.01,left=-1,right=1,u=1,sig=1):
    left=a.min()
    right=a.max()
    N=len(a)
    n=int(d/delta)
    t=list()
    y=list()
    for i in np.arange(left,right,delta):
        count=0
        for j in a:
            if  i-d <j < i+d:
                count+=1
        t.append(i)
        y.append(count)
    y=np.array(y)/N/(2*d)
    plt.figure()
    plt.title(' Fig: the distribution of '+x  )
    plt.xlabel(x)
    plt.ylabel(P)  
    plt.scatter(t[n:-n],y[n:-n],s=10,marker='*',label='Experiment')

    #plt.plot(t,-t,c='green',label='$y=0.98exp(-x)$')
    plt.xlim(left,right)
    '''
    #plt.show()
    popt, pcov = curve_fit(func,t[n:-n],y[n:-n]*30)
    x=np.arange(left,right,0.1)
    yfit=[func(i,popt[0],popt[1],popt[2]) for i in x]
    yfit=np.array(yfit)/30
    #plt.show()
    plt.plot(x,yfit,label='$y='+str(popt[0]/10)[:5]+'exp(-'+str(popt[1])[:5]+'(d'+str(popt[2])[:5]+')^2)$')
    '''
    t=np.arange(left,right,0.1)
    yfit=[func(i,u,sig) for i in t]
    plt.plot(t,yfit,label="Gauss fit(u="+str(u)[:5]+",sigma="+str(sig)[:5]+")")
    plt.xlim(left,right)
    plt.legend(loc='upper right')
    plt.show()
    return t[n:-n],y[n:-n]