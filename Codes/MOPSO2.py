import numpy as np
import matplotlib.pyplot as plt
import random


def fit1(w):
    return sum(x**2 for x in w)
    #return w[0]
    
def fit2(w):
    #g=1+9*sum(w[1:])/(len(w)-1)

    #return g*(1-(w[0]/g)**2) 
    return sum((x-2.0)**2 for x in w) 
def GetCosts(pop):
    
    cost=np.zeros((len(pop),2))
    
    for i in xrange(len(pop)):
        cost[i]=[fit1(pop[i]), fit2(pop[i])]
        
    
    #print "COST",cost
    return cost
    
def Cost(a,b):
    x=[fit1(a), fit2(a)]
    y=[fit1(b), fit2(b)]
    
    return x,y     
    


def Dominates(a,b):

    x,y=Cost(a,b)
    
    
    dom= x[0]<y[0] and x[1]<y[1] 
    #print "dom",dom
    return dom

def DetermineDomination(pop):

    npop=len(pop)
    Dominated=np.zeros(npop)
    for i in xrange(npop):
        
        for j in xrange(npop):
            if not Dominated[i]:
                if Dominates(pop[i],pop[j]):
                    Dominated[j]=1
                           
    return Dominated            

def GetNonDominatedParticles(pop,Dominated):

    
    nd_pop=pop[np.where(Dominated==0)]
    
    return nd_pop
def CreateHypercubes(rep_costs,nGrid,alpha):
    nobj=len(rep_costs[0])
    
    G=np.zeros((nobj,nGrid))    
    
    
    for j in xrange(nobj):
        
        min_cj=min(rep_costs[:,j])
        max_cj=max(rep_costs[:,j])
        
        dcj=alpha*(max_cj-min_cj)
        
        min_cj=min_cj-dcj
        max_cj=max_cj+dcj
        #print max_cj, min_cj
        g=np.linspace(min_cj,max_cj,num=nGrid-1)
        
        G[j,:]=np.concatenate(([-np.inf], g))
        #G_Lower[j,:]=np.concatenate(([-np.inf], gx))
        #G_Upper[j,:]=np.concatenate((gx, [np.inf]))
        
    return G
def findCandidateSolution(rep,rep_costs,nGrid,alpha):
    G=CreateHypercubes(rep_costs,nGrid,alpha)
    #G[0]--> x-axis
    #G[1]--> y-axis
    cell=np.zeros((nGrid-1,nGrid-1))
    cost_index=np.zeros((len(rep_costs), 2))
    
    for i in xrange(len(rep_costs)):
        for j in xrange(nGrid-2):
            for k in xrange(nGrid-2):
                if (rep_costs[i,0]>G[0,j] and rep_costs[i,0]<G[0,j+1]) and (rep_costs[i,0]>G[1,k] and rep_costs[i,0]<G[1,k+1]):
                    cell[j,k]=cell[j,k]+1
                    cost_index[i]=[j,k]
                    
    ind=np.where(cell>0) 
    m=np.random.randint(len(ind[0])-1) #roulette-wheel selection
    
    
    
    n=np.where(cost_index[:,0]==ind[0][m])
    
    
    h=random.choice(n[0])
    return h
      
    
def updateRep(rep,rep_cost, best_pos, best_cst, rep_size, nGrid, alpha):
    
    
    dm=False
    for i in range(len(rep)):
        if Dominates(rep[i],best_pos):
            dm=True
    if dm==False:
        while len(rep)>=rep_size:
            #print "rep shape b4 deletion",rep.shape
            h=findCandidateSolution(rep,rep_costs,nGrid,alpha)
            rep=np.vstack((rep[:h],rep[h+1:]))
            #print "rep shape af deletion",rep.shape
        rep=np.vstack((rep,best_pos))        
    
    return rep
    
def create_population(search_space,nVar,pop_size):
    
    min_n=search_space[0]
    max_n=search_space[1]    
    
    position=np.zeros((pop_size,nVar))
    velocity=np.zeros((pop_size,nVar))
    
    for i in xrange(pop_size):
        for j in xrange(nVar):
            position[i,j]=min_n +np.random.random()*(max_n-min_n)
            velocity[i,j]=0
    
    return position,velocity        

def MOPSO(search_space,nVar,pop_size,rep_size,maxgen,w,wdamp,c1,c2,nGrid, alpha):
    
    position,velocity = create_population(search_space,nVar,pop_size)
    costs=GetCosts(position)

    Dominated=DetermineDomination(position)

    rep=GetNonDominatedParticles(position,Dominated)
    
    rep_costs=GetCosts(rep)
    
    best_position=position
    best_costs=costs
    
    
    
    for epoch in xrange(100):
        #print "rep shape b4",rep.shape
        h=findCandidateSolution(rep,rep_costs,nGrid,alpha)
        
        #print "h",h
        for i in xrange(pop_size):
            
            #print "rep shape",rep.shape
            #print "rep",rep[h]
            velocity[i]=w*velocity[i] + np.random.random()*(best_position[i]-position[i])\
                           + np.random.random()*( rep[h]-position[i])            
            
            position[i]=np.add(position[i] ,velocity[i])
            
            g1=np.where(position < search_space[0])
            g2=np.where(position > search_space[1])
            
            #position[g]=
            velocity[g1]=-velocity[g1]
            position[g1]= search_space[0]
            
            velocity[g2]=-velocity[g2]
            position[g2]= search_space[1]
                        
            #print position
            if Dominates(position[i], best_position[i]):
                best_position[i]=position[i]
                best_costs[i]=costs[i]
            
            rep=updateRep(rep,rep_costs, best_position[i], best_costs[i], rep_size,nGrid,alpha)
            rep_costs=GetCosts(rep)
            print "rep shape", rep.shape
    best_costs=GetCosts(best_position)
    rep_costs=GetCosts(rep)
    
    return best_costs,rep_costs
   
    
    
if __name__=="__main__":
    
    search_space=[-10,10]    
    
    nVar=1
    
    pop_size=100
    rep_size=100
    
    maxgen=100
    
        
    phi1=2.05
    phi2=2.05
    phi=phi1+phi2
    chi=2.0/(phi-2+np.sqrt(phi**2-4*phi))
    
    w=chi              #Inertia Weight
    wdamp=1            # Inertia Weight Damping Ratio
    c1=0.2#chi*phi1        # Personal Learning Coefficient
    c2=0.3#chi*phi2        # Global Learning Coefficient
    nGrid=10
    alpha=0.1
         
    best_costs,rep_costs=MOPSO(search_space,nVar,pop_size,rep_size,maxgen,w,wdamp,c1,c2,nGrid,alpha)
       
    plt.figure()    
    plt.scatter(best_costs[:,0],best_costs[:,1])
    
    plt.show()
    plt.figure()
    plt.scatter(rep_costs[:,0],rep_costs[:,1]) 
    plt.show()
    