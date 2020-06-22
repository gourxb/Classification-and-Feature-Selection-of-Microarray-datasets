import numpy as np
import math

def create_cats(pop_size, D, search_space):
    
    X=np.ones((pop_size,D))*search_space[0]+ np.random.rand(pop_size,D)*(np.ones((pop_size,D))*(search_space[1]-search_space[0]))
    
    
    V=np.random.rand(pop_size,D)
    
    return X,V
    
def objective_fucntion(w):
    
    #x=w[0]
    #y=w[1]
    #Ackley's function
    #return -20.0*math.exp(-0.2*math.sqrt(0.5*(x**2+y**2)))- math.exp(0.5*(math.cos(2*math.pi*x)+math.cos(2*math.pi*x)))+math.e+20 
    
    s=0
    for i in xrange(1,len(w)):
        s=s+ 100*(w[i]-w[i-1])**2 + (w[i-1]-1)**2
    return s

def evaluate_fitness(X):
    fit=np.zeros(len(X))    
    
    for i in xrange(len(X)):
        fit[i]=objective_fucntion(X[i,:])
        
    return fit
    
def sort_fitness(X,fit):    

    ind=np.argsort(fit)
    
    return X[ind], fit[ind]
    
    
def seeking_mode(w,f,SMP, SRD, CDC, D):
    
    T= w*np.ones((SMP,D))
    
    for i in xrange(SMP):
        if np.random.random() < CDC:
            T[i,:]=T[i,:] + (np.round(np.random.uniform(low=-1, high=1))*SRD*T[i,:])
                
    fit_T=evaluate_fitness(T)
    T,fit_T=sort_fitness(T,fit_T)
    
    if fit_T[0]< f:
        return T[0,:], fit_T[0]
        
    else:
        return w,f

def tracing_mode(X,V,global_best_cat, w_cat):
    
    #print V
    #print global_best_cat*np.ones(X.shape)
    V=np.add(w_cat*V,  0.2*np.random.random()*np.subtract(global_best_cat*np.ones(X.shape) ,X))
    #print V    
    X=np.add(X,V)
    fit=evaluate_fitness(X)
    X,fit=sort_fitness(X,fit)
    return X,V
   
def CSO(MR,SRD,CDC,w_cat, pop_size, D, SMP, search_space):


    X, V =create_cats(pop_size, D, search_space)
    
    fit=evaluate_fitness(X)
    X,fit=sort_fitness(X,fit)
    #print X.shape
    best_cat=X[0,:]
    fit_best=fit[0]
    global_best_cat=X[0,:]
    fit_global_best=fit[0]
    
    mode=np.zeros(pop_size)
    
    epoch=0
    maxepoch=100

    while epoch< maxepoch:
        
        for i in xrange(pop_size):
            if np.random.random()<MR:            
                mode[i]=0
            else:
                mode[i]=1
                
        for i in xrange(pop_size):
            if mode[i]==0:
                X[i,:],fit[i]=seeking_mode(X[i,:],fit[i],SMP, SRD, CDC, D)
            else:
                X,V=tracing_mode(X,V,global_best_cat, w_cat)
        
        
        
        best_cat=X[0,:]
        fit_best=fit[0]
        
        if fit_best <fit_global_best:
            global_best_cat=best_cat
            fit_global_best=fit_best
            
        epoch=epoch+1    
    
    print X
    print fit    
    print global_best_cat, fit_global_best
            
        
def main():
    
    MR=0.2
    SRD=0.2
    CDC=0.8
    w_cat=0.5
    
    pop_size=10
    D=4
    SMP=int(0.25*pop_size)
   
    search_space=[-10,10]
    
    CSO(MR,SRD,CDC,w_cat, pop_size, D, SMP, search_space)
    #print gb, fgb
    
    
main()
    