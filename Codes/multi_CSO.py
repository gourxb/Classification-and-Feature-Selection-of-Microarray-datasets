import numpy as np
import math
import matplotlib.pyplot as plt



def create_cats(pop_size, D, search_space):
    
    X=search_space[0]*np.ones((pop_size,D))+ np.random.rand(pop_size,D)*(np.ones((pop_size,D))*(search_space[1]-search_space[0]))
    
    
    V=np.random.rand(pop_size,D)
    #print X.shape
    return X,V
    
  
    
def dominates(p1,p2):
    
    if (p1 < p2).all():
        return True
    else:
        return False

def calculate_crowding_distance(I,objectives):
    
    m=2
    l=len(I)
    distance=np.zeros((l))
    
    for p in range(len(I)):
        distance[p] = 0
        
    for obj_index in range(m):
        #I=sort_objective(I,objectives[:,obj_index])
        distance[0]=distance[l-1]=float('inf')
        
        f_max=max(objectives[obj_index])
        f_min=min(objectives[obj_index])        
        
        for i in xrange(1,l-1):
            distance[i]= distance[i]+ (objectives[i+1,obj_index]-objectives[i-1,obj_index])/(f_max-f_min)
      
    return distance        

def crowded_comparison_operator(rank, distance, x,y):
    
    if rank[x] < rank[y]:
        return 1
        
    elif rank[x] > rank[y]:
        return -1
        
    elif distance[x] > distance[y]:
        return 1
        
    elif distance[x] < distance[y]:
        return -1
        
    else:
        return 0    

def sort_crowding(P,r,d):
    for i in range(len(P) - 1, -1, -1):
        for j in range(1, i + 1):
            s1 = j - 1
            s2 = j
            
            if crowded_comparison_operator(r,d , s1, s2) < 0:
                P[j - 1] = P[s2]
                P[j] = P[s1]
                
    return P
            
def select_parents(fronts,fronts_v,rank, pop_size,D, objectives):
    
    dist={}
    for each in fronts:
        dist[each]=calculate_crowding_distance(fronts[each],objectives)
    
    pop_next=np.empty((0,D))
    vel_next=np.empty((0,D))
    i=1
    
        
    while len(pop_next)+len(fronts[i])<pop_size:
        #print i,  len(pop_next)+len(fronts[i]), pop_size  
        g=np.array(fronts[i])
        g_v=np.array(fronts_v[i])
        #print pop_next.shape, g.shape
        pop_next=np.concatenate((pop_next, g))
        vel_next=np.concatenate((vel_next, g_v))
        #print "FRONTS",fronts[i]
        #print i,pop_next        
        i=i+1
        
    
    #print "gi"
    P=sort_crowding(fronts[i], rank, dist[i])
    
    pop_next=np.vstack((pop_next, P[:(pop_size-len(pop_next))]))
    vel_next=np.vstack((vel_next, P[:(pop_size-len(vel_next))]))
    
    
    return pop_next,vel_next

        
def non_dominated_archive(pop,V,fit,pop_size,D):
    fronts = {}
    fronts_v={}
    #print "pop", pop
    fronts[1] = np.empty((0,len(pop[0])))
    fronts_v[1] = np.empty((0,len(pop[0])))
   
    S={}
    n={} 
    rank={}
    
    for p in xrange(len(pop)):
        S[p] = []
        n[p] = 0
        for q in xrange(len(pop)):
                          
            if dominates(fit[p,:],fit[q,:]):
                S[p].append(pop[q,:])
            
            elif dominates(fit[q,:],fit[p,:]):
                n[p] += 1
        
        if n[p] == 0:
            rank[p]=0
            
            fronts[1]=np.vstack((fronts[1],pop[p,:]))
            fronts_v[1]=np.vstack((fronts[1],V[p,:]))
            #print fronts[1]
    
       
    i = 1
    
    while len(fronts[i]) != 0:
        next_front = np.empty((0,len(pop[0])))
        next_front_v= np.empty((0,len(V[0])))
        
        for r in xrange(len(fronts[i])):
            for s in S:
                n[s] -= 1
               
                if n[s] == 0:
                    rank[s]=i+1
                    next_front=np.vstack((next_front,pop[p,:]))
                    next_front_v=np.vstack((next_front_v,pop[p,:]))
        i += 1
        
        if len(next_front)!=0:
            fronts[i] = next_front
            fronts_v[i] = next_front_v
        else:
            break
    
          
    X,V=select_parents(fronts,fronts_v,rank, pop_size,D, fit)
    #print "X",X
    #print "XXXXX",X
    
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
    
  
    
def seeking_mode(w,f,SMP, SRD, CDC, D):
   
    T= w*np.ones((SMP,D))
    
    for i in xrange(SMP):
        if np.random.random() < CDC:
            T[i,:]=T[i,:] + (np.round(np.random.uniform(low=-1, high=1))*SRD*T[i,:])
                
    fit_T=evaluate_fitness(T)
    
    for i in xrange(SMP):

        if dominates(fit_T[i], f):
            return T[i,:], fit_T[0]
    
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
    
    X,V=non_dominated_archive(X,V,fit,pop_size,D)
    
    return X,V
   
def CSO(MR,SRD,CDC,w_cat, pop_size, D, SMP, search_space):


    X, V =create_cats(pop_size, D, search_space)
    #print X
    fit=evaluate_fitness(X)
    X,V=non_dominated_archive(X,V,fit,pop_size,D) 
    
    
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
                X[i,:],fit[i]=seeking_mode(X[i,:],fit[i],SMP, SRD, CDC, D)
                #print X[i,:],fit[i]
            else:
                global_best_cat=X[np.random.random_integers(0,len(X)-1),:]
                X,V=tracing_mode(X,V,global_best_cat, w_cat,fit,pop_size,D, search_space)
                #print X
        epoch=epoch+1 
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
    