import numpy as np
import random
import matplotlib.pyplot as plt

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
      
    
def updateRep(rep,rep_cost, best_pos, rep_size, nGrid, alpha):
    
    
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
    

def seeking_mode(w,f,SMP, SRD, CDC, nVar,rep):
    
    Y= w*np.ones((SMP,nVar))
    
    for i in xrange(SMP):
        if np.random.random() < CDC:
            Y[i,:]=Y[i,:] + (np.round(np.random.uniform(low=-1, high=1))*SRD*Y[i,:])
                
       
    for i in xrange(SMP):
        dm=False
        for j in xrange(len(rep)):
            if Dominates(rep[j], Y[i]):
                dm=True
                break 
        if dm==False:
            rep=np.vstack((rep,Y[i]))
    
    
    return rep , Y[np.random.randint(SMP-1)]

def tracing_mode(pos,vel,costs, best_position, C, w_cat, rep, rep_size, nGrid, alpha):
  
    
    
    vel=w_cat*vel + np.random.random()*(best_position-pos)
                                
        
    pos=np.add(pos,vel)
    g1=np.where(pos < search_space[0])
    g2=np.where(pos > search_space[1])
    
    #position[g]=
    vel[g1]=-vel[g1]
    pos[g1]= search_space[0]
    
    vel[g2]=-vel[g2]
    pos[g2]= search_space[1]
    
    if Dominates(pos, best_position):
            best_position=pos
            
        
    rep=updateRep(rep,rep_costs, best_position, rep_size,nGrid,alpha)
    
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
    
    
def MOCSO(search_space,nVar,pop_size,rep_size,maxgen,w_car,C,nGrid,alpha, MR,SRD,CDC,SMP):
 
    position,velocity = create_population(search_space,nVar,pop_size)
    costs=GetCosts(position)

    Dominated=DetermineDomination(position)

    rep=GetNonDominatedParticles(position,Dominated)
    
    rep_costs=GetCosts(rep)
     
      
    mode=np.zeros(pop_size)
    
   

    for epoch in xrange(maxgen):
        
        mode=np.round(np.random.rand(pop_size))
                
        for i in xrange(pop_size):
            if mode[i]==0:
                
                rep, position[i]=seeking_mode(position[i],costs[i],SMP, SRD, CDC, nVar,rep)
            else:
                
                best_position=rep[np.random.randint(len(rep)-1)]
                rep=tracing_mode(position[i],velocity[i],costs[i], best_position, C, w_cat, rep,rep_size, nGrid, alpha)
        
    rep_costs=GetCosts(rep)
            
           
    return rep_costs
            
        
if __name__=="__main__":
    
    MR=0.5
    SRD=0.2
    CDC=0.8
    w_cat=0.5
    C=2
    SMP=5
   
    search_space=[-10,10]    
    
    D=nVar=1
    pop_size=100
    rep_size=100
    maxgen=10
    
    nGrid=10
    alpha=0.1
       
    
    rep_costs=MOCSO(search_space,nVar,pop_size,rep_size,maxgen,w_cat,C,nGrid,alpha, MR,SRD,CDC,SMP)
       
   
    plt.figure()
    plt.scatter(rep_costs[:,0],rep_costs[:,1]) 
    plt.show()