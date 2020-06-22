import numpy as np
import math
import random

def create_population(pop_size,D):
    x_min=0.1
    x_max=0.9
    
    par=np.zeros((pop_size,D))
    vel=np.zeros((pop_size,D))
    
    
    for i in xrange(pop_size):
        for j in xrange(D):
            par[i,j]=random.uniform(x_min,x_max)
            vel[i,j]=random.uniform(x_min/3, x_max/3)
    
    return par,vel        
    
def objective_fucntion(w):
    
    x=w[0]
    y=w[1]
    #Ackley's function
    return -20.0*math.exp(-0.2*math.sqrt(0.5*(x**2+y**2)))- math.exp(0.5*(math.cos(2*math.pi*x)+math.cos(2*math.pi*x)))+math.e+20 
    '''
    s=0
    for i in xrange(1,len(w)):
        s=s+ 100*(w[i]-w[i-1])**2 + (w[i-1]-1)**2
    return s'''

def evaluate_fitness(X):
    fit=np.zeros(len(X))    
    
    for i in xrange(len(X)):
        fit[i]=objective_fucntion(X[i,:])
        
    return fit
    
def PSO(pop_size,D):
    
    par,vel=create_population(pop_size,D)
    
    fit=evaluate_fitness(par)
    
    global_fit=np.min(fit)
       
    
    local_par = par
    local_fit = fit
    
    indx= np.argmin(fit)
    global_par=par[indx,:]   #best srtring of w
    
       
    C1 = 1 #cognitive parameter
    C2 = 4-C1 # social parameter
    #C=1  #constriction factor
    
       
    maxit=10
    iter = 0
    while iter < maxit:
        iter = iter + 1
        #w=(maxit-iter)/maxit
        r1 = np.random.rand(pop_size,D)
        r2 = np.random.rand(pop_size,D)
        # update particle positions
        vel = vel + C1*r1*(local_par-par) + C2*r2*((np.ones((pop_size,1))*global_par)-par)
        par = par + vel # updates particle position
    
        fit=evaluate_fitness(par)
        
        better_fit = fit < local_fit
        local_fit = local_fit*np.invert(better_fit) +  fit*better_fit
        
        local_par[np.where(better_fit==True),:]= par[np.where(better_fit==True),:]
        
        temp = np.min(local_fit)
        t = np.argmin(local_fit)
        if temp<global_fit:
            global_par=par[t,:]
            indx=t
            global_fit=temp
    
    #print cost,localcost
    return global_par,global_fit,            

def main():
    pop_size=20
    D=2
    
    p,f= PSO(pop_size,D)
    print p,f

main()    
