import numpy as np
import matplotlib.pyplot as plt

def fit1(w):
    #print w
    return sum(x**2 for x in w)
    
def fit2(w):
    return sum((x-2.0)**2 for x in w)  
    
def GetCosts(pop):
    
    cost=np.zeros((len(pop),2))
    
    for i in xrange(len(pop)):
        cost[i,0]=fit1(pop[i])
        cost[i,1]=fit2(pop[i])
    
    #print "COST",cost
    return cost
    
def Cost(a,b):
    x=[fit1(a), fit2(a)]
    y=[fit1(b), fit2(b)]
    
    return x,y     
    

def Dominates(a,b):

    x,y=Cost(a,b)
    
    
    dom= np.all(x<=y) and np.any(x<y)
    
    return dom

def DetermineDomination(pop):

    npop=len(pop)
    Dominated=np.zeros(npop)
    for i in xrange(npop):
        Dominated[i]=False
        for j in xrange(i-1):
            if not Dominated[i]:
                if Dominates(pop[i],pop[j]):
                    Dominated[j]=True
                elif Dominates(pop[j],pop[i]):
                    Dominated[j]=True
                    break
    return Dominated            

def GetNonDominatedParticles(pop,Dominated):

    
    nd_pop=pop[np.where(Dominated==1)]
    #print nd_pop
    return nd_pop
    
def GetGridIndex(rep, G_Lower, G_Upper,rep_cost): 
    c=rep_cost
    nobj=len(c)
    ngrid=len(G_Upper[0])

    Index=12#np.ones((1,nobj))*ngrid
    SubIndex=np.zeros(nobj)
    #print SubIndex.shape
    for j in xrange(nobj):
        U=G_Upper[j]
        
        i=np.where(c[j]<U)[0][0]
        print i
        SubIndex[j]=i
      
    #print Index, SubIndex    
    return Index, SubIndex      
    
    
def RouletteWheelSelection(p):
    r=np.random.random()
    c=np.cumsum(p)
    i=np.where(r<=c)
    
    return i
    
def GetOccupiedCells(rep):
    GridIndices=GetGridIndex(rep)
    
    occ_cell_index=np.unique(GridIndices)
    
    occ_cell_member_count=np.zeros(occ_cell_index.shape)

    m=len(occ_cell_index)
    for k in xrange(m):
        occ_cell_member_count[k]=sum(np.where(GridIndices==occ_cell_index[k]))
    return occ_cell_index, occ_cell_member_count
    
def DeleteFromRep(rep,EXTRA,gamma, GridIndex, GridSub):
    
    '''
    for k in xrange(EXTRA):
        [occ_cell_index, occ_cell_member_count]=GetOccupiedCells(rep)

        p=occ_cell_member_count**gamma
        p=p/sum(p)

        selected_cell_index=occ_cell_index(RouletteWheelSelection(p))

        GridIndices=GridIndex

        selected_cell_members=np.where(GridIndices==selected_cell_index)

        n=len(selected_cell_members)

        selected_memebr_index=np.random.random_integer(n)

        j=selected_cell_members(selected_memebr_index)
        
        rep=np.append((rep[:j-1], rep[j+1:]))
    '''
    return rep    
def CreateHypercubes(costs,nGrid,alpha):
    nobj=len(costs[0])
    
    G_Upper=G_Lower=np.zeros((nobj,nGrid))    
    
    
    for j in xrange(nobj):
        
        min_cj=min(costs[j,:])
        max_cj=max(costs[j,:])
        
        dcj=alpha*(max_cj-min_cj)
        
        min_cj=min_cj-dcj
        max_cj=max_cj+dcj
        
        gx=np.linspace(min_cj,max_cj,num=nGrid-1)
        
        G_Lower[j,:]=np.concatenate(([-np.inf], gx))
        G_Upper[j,:]=np.concatenate((gx, [np.inf]))
        
    return G_Lower, G_Upper
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
    #print "rep", rep.shape
    rep_costs=GetCosts(rep)
    G_Lower, G_Upper=CreateHypercubes(rep_costs,nGrid,alpha)
    
    rep_GridIndex= rep_GridSubIndex= np.zeros((len(rep),2))
    #for i in xrange(len(rep)):
    #    rep_GridIndex[i], rep_GridSubIndex[i]=GetGridIndex(rep[i],G_Lower, G_Upper,rep_costs[i])

    best_position=position
    best_costs=costs
    
    rep_h=np.zeros((2,nVar))
    
    for epoch in xrange(maxgen):
        for i in xrange(pop_size):
            
            #Cst=GetCosts(rep)
            #s=np.argmin(Cst[:,0])
            s=np.random.randint(0,len(rep))
            rep_h[0]=rep[s]
            #s=np.argmin(Cst[:,1])
            rep_h[1]=rep[s]
                
            #velocity[i]=w*velocity[i]  +c1*np.random.random()*(best_position[i] - position[i]) \
            #                 +c2*np.random.random()*(rep_h[0] -  position[i]) \
            #                 +c2*np.random.random()*(rep_h[1] -  position[i])
            h=np.random.random_integers(0,len(rep)-1)
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
            elif Dominates(best_position[i], position[i]):
                if np.random.random()<0.3:
                    best_position[i]=position[i]
                    best_costs[i]=costs[i]
                    
        #print best_cost
        Dominated=DetermineDomination(best_position)
        nd_position=GetNonDominatedParticles(best_position,Dominated)
        rep=nd_position
        #rep=np.vstack((rep,nd_position))
        #rep_GridIndex= rep_GridSubIndex= np.zeros((len(rep),2))
        #for i in xrange(len(rep)):
        #    rep_GridIndex[i], rep_GridSubIndex[i]=GetGridIndex(rep[i],G_Lower, G_Upper, rep_costs)

        
        if len(rep)>rep_size:
                       
            EXTRA=len(rep)-rep_size
            rep=DeleteFromRep(rep,EXTRA,0.5, rep_GridIndex, rep_GridSubIndex)
            
            rep_costs=GetCosts(rep)
     #       G_Lower, G_Upper=CreateHypercubes(rep_costs,nGrid,alpha)
        print rep.shape
        w=w*wdamp  
        #print "size of REP", len(rep)
    
    best_costs=GetCosts(best_position)
    rep_costs=GetCosts(rep)
    
    return best_costs,rep_costs
    #print rep_costs    
       
    
if __name__=="__main__":
    
    search_space=[-10,10]    
    
    nVar=1
    
    pop_size=100
    rep_size=20
    
    maxgen=20
    
        
    phi1=2.05
    phi2=2.05
    phi=phi1+phi2
    chi=2.0/(phi-2+np.sqrt(phi**2-4*phi))
    
    w=chi              #Inertia Weight
    wdamp=1            # Inertia Weight Damping Ratio
    c1=chi*phi1        # Personal Learning Coefficient
    c2=chi*phi2        # Global Learning Coefficient
    nGrid=10
    alpha=0.1
         
    best_costs,rep_costs=MOPSO(search_space,nVar,pop_size,rep_size,maxgen,w,wdamp,c1,c2,nGrid,alpha)
    #plt.scatter(best_costs[:,0],best_costs[:,1])
    #plt.scatter(rep_costs[:,0],rep_costs[:,1]) 

    