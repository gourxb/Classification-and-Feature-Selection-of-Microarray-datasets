import numpy as np
import math
import matplotlib.pyplot as plt


def create_cats(pop_size, D, search_space):
    
    X=search_space[0]*np.ones((pop_size,D))+ np.random.rand(pop_size,D)*(np.ones((pop_size,D))*(search_space[1]-search_space[0]))
    
    
    V=np.random.rand(pop_size,D)
    #print X.shape
    return X,V
    
          
def objective_fucntion1(w):
    return sum(x**2 for x in w)

def objective_function2(w):
    return sum((x-2.0)**2 for x in w)          

def evaluate_fitness(X):
    fit=np.zeros((len(X),2))    
    
    for i in xrange(len(X)):
        fit[i,0]=objective_fucntion1(X[i,:])
        fit[i,1]=objective_function2(X[i,:])
        
    return fit


def Dominates(a,b):

      
    dom= np.all(a<=b) and np.any(a<b)
    
    return dom
  
def non_dominated_archive(X,V,fit,pop_size,D):
    
    npop=pop_size
    Dominated=np.zeros(npop)
    for i in xrange(npop):
        Dominated[i]=False
        for j in xrange(i-1):
            if not Dominated[i]:
                if Dominates(fit[i],fit[j]):
                    Dominated[j]=True
                elif Dominates(fit[j],fit[i]):
                    Dominated[j]=True
                    break
 
    nd_pop=X[np.where(Dominated==0)]
    
    return nd_pop
    
def seeking_mode(w,f,rep,SMP, SRD, CDC, D):
   
    T= w*np.ones((SMP,D))
    
    for i in xrange(SMP):
        if np.random.random() < CDC:
            T[i,:]=T[i,:] + (np.round(np.random.uniform(low=-1, high=1))*SRD*T[i,:])
                
    fit_T=evaluate_fitness(T)
    
    for i in xrange(SMP):

        if Dominates(fit_T[i], f):
            #print "hey"
            return T[i,:], fit_T[i]
    
    return w,f   

def tracing_mode(X,V,global_best_cat, w_cat,fit,pop_size,D,search_space):
    
    #print X.shape,V.shape
    V=np.add(w_cat*V,  0.2*np.random.random()*np.subtract(global_best_cat*np.ones(X.shape) ,X))
    X=np.add(X,V)
    
    ind1=np.where(X>search_space[1])
    X[ind1]=search_space[1]
    V[ind1]=V[ind1]*-1
    
    ind2=np.where(X<search_space[0])
    X[ind2]=search_space[0]
    V[ind2]=V[ind2]*-1
    
    #X,V=non_dominated_archive(X,V,fit,pop_size,D)
    
    return X,V
   
def CSO(MR,SRD,CDC,w_cat, pop_size, D, SMP, search_space):


    X, V =create_cats(pop_size, D, search_space)
    #print X
    fit=evaluate_fitness(X)
    #print fit
    rep=non_dominated_archive(X,V,fit,pop_size,D) 
    
    
    mode=np.zeros(len(X))
    
    epoch=0
    maxepoch=10

    while epoch< maxepoch:
        #print X.shape
        for i in xrange(len(X)):
            if np.random.random()<MR:            
                mode[i]=0
            else:
                mode[i]=1
                
        for i in xrange(len(X)):
            if mode[i]==0:
                #print "hi",X.shape, fit.shape
                #print X[i,:],fit[i]
                X[i,:],fit[i]=seeking_mode(X[i,:],fit[i],rep,SMP, SRD, CDC, D)
                #rep=np.vstack((rep,X[i,:]))                
                #print X[i,:],fit[i]
            else:
                global_best_cat=X[np.random.random_integers(0,len(X)-1),:]
                X,V=tracing_mode(X,V,global_best_cat, w_cat,fit,pop_size,D, search_space)
                
                rep=non_dominated_archive(X,V,fit,pop_size,D) 
                
                #print X
        epoch=epoch+1 
    fit=evaluate_fitness(rep)   
    #print fit
    #print rep
    colors=np.random.rand(pop_size)
    plt.scatter(fit[:,0],fit[:,1], c=colors)
            
       
def main():
    
    MR=0.3
    SRD=0.2
    CDC=0.3
    w_cat=0.5
    
    pop_size=20
    D=1
    SMP=int(0.25*pop_size)
   
    search_space=[-10,10]
    
    CSO(MR,SRD,CDC,w_cat, pop_size, D, SMP, search_space)
    #print gb, fgb
    
    
main()
    