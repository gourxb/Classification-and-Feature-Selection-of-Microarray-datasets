import numpy as np
import matplotlib.pyplot as plt

def fit1(w):
    #return sum(x**2 for x in w)
    return w[0]
    
def fit2(w):
    g=1+9*sum(w[1:])/(len(w)-1)

    return g*(1-(w[0]/g)**2) 
    #return sum((x-2.0)**2 for x in w) 
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
        Dominated[i]=False
        for j in xrange(i-1):
            if not Dominated[i]:
                if Dominates(pop[i],pop[j]):
                    Dominated[j]=True
                #elif Dominates(pop[j],pop[i]):
                #    Dominated[j]=True
                #    break
    #print Dominated            
    return Dominated            

def GetNonDominatedParticles(pop,Dominated):

    
    nd_pop=pop[np.where(Dominated==0)]
    #print "ND pop",nd_pop
    return nd_pop
    
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
    #print "rep", rep
    
      
    #G_Lower, G_Upper=CreateHypercubes(rep_costs,nGrid,alpha)
    
    #rep_GridIndex= rep_GridSubIndex= np.zeros((len(rep),2))
    #for i in xrange(len(rep)):
    #    rep_GridIndex[i], rep_GridSubIndex[i]=GetGridIndex(rep[i],G_Lower, G_Upper,rep_costs[i])

    best_position=position
    best_cost=costs
    
    rep_h=np.zeros((2,nVar))
    
    for epoch in xrange(maxgen):
        for i in xrange(pop_size):
            
            #Cst=GetCosts(rep)
            s=np.argmin(rep_costs[:,0])
            rep_h[0]=rep[s]
            s=np.argmin(rep_costs[:,1])
            rep_h[1]=rep[s]
                
            velocity[i]=w*velocity[i]  +c1*np.random.random()*(best_position[i] - position[i]) \
                             +c2*np.random.random()*(rep_h[0] -  position[i]) \
                             +c2*np.random.random()*(rep_h[1] -  position[i])
            #h=np.random.random_integers(0,len(rep)-1)
            #velocity[i]=w*velocity[i] + np.random.random()*(best_position[i]-position[i])\
            #               + np.random.random()*( rep[h]-position[i])            
            
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
                best_cost[i]=costs[i]
            
            for j in xrange(len(rep)):
                if Dominates(best_position[i], rep[j]):
                    rep=np.vstack((rep,best_position[i]))
                    break
        #print best_cost
        #Dominated=DetermineDomination(best_position)
        #nd_position=GetNonDominatedParticles(best_position,Dominated)
        #rep=nd_position
        #rep=np.vstack((rep,nd_position))
        #Dominated=DetermineDomination(rep)
        #rep=GetNonDominatedParticles(rep,Dominated)
        #rep_costs=GetCosts(rep)
        rep=rep[0:20]
        #
        #
    
    best_costs=GetCosts(best_position)
    rep_costs=GetCosts(rep)
    
    return best_costs,rep_costs
    #print rep_costs    
    
    
if __name__=="__main__":
    
    search_space=[0,1]    
    
    nVar=10
    
    pop_size=100
    rep_size=20
    
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
    