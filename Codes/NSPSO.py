import numpy as np
import matplotlib.pyplot as plt

def create_population(pop_size,no_var,search_space):
    x_min=search_space[0]
    x_max=search_space[1]
    
    par=np.zeros((pop_size,no_var))
    vel=np.zeros((pop_size,no_var))
    
    
    for i in xrange(pop_size):
        for j in xrange(no_var):
            par[i,j]=np.random.uniform(x_min,x_max)
            vel[i,j]=np.random.uniform(x_min/3, x_max/3)
    
    return par,vel

def fit1(vector):
    return sum(x**2 for x in vector)


def fit2(vector):
    return sum((x-2.0)**2 for x in vector)
    
    
def evaluate_fitness(pop,pop_size,no_var, search_space):

    
    fitness=np.zeros((pop_size, 2))
    for i in xrange(pop_size):
        
        fitness[i,:]= [fit1(pop[i,:]), fit2(pop[i,:])]

    return fitness 
    
def dominates(p1,p2):
    
    if (p1 < p2).all():
        return True
    else:
        return False
    
def non_dominated_sort(X,V,fitness):
    fronts_X = {}
    fronts_V = {}
    
    fronts_X[1] = np.empty((0,len(X[0])))
    fronts_V[1] = np.empty((0,len(X[0])))
    #rank=np.zeros(len(pop))
    #print fronts[1].shape
    #print pop.shape
    S={}
    n={} 
    rank={}
    
    for p in xrange(len(X)):
        S[p] = []
        n[p] = 0
        for q in xrange(len(X)):
                          
            if dominates(fitness[p,:],fitness[q,:]):
                S[p].append(X[q,:])
            
            elif dominates(fitness[q,:],fitness[p,:]):
                n[p] += 1
        
        if n[p] == 0:
            rank[p]=0
            #print fronts[1], pop[p,:]
            fronts_X[1]=np.vstack((fronts_X[1],X[p,:]))
            fronts_V[1]=np.vstack((fronts_V[1],V[p,:]))
            #print fronts[1]
    
    i = 1
    
    while len(fronts_X[i]) != 0:
        next_front_X = np.empty((0,len(X[0])))
        next_front_V = np.empty((0,len(V[0])))
        
        for r in xrange(len(fronts_X[i])):
            for s in S:
                n[s] -= 1
               
                if n[s] == 0:
                    rank[s]=i+1
                    next_front_X=np.vstack((next_front_X, X[s,:]))
                    next_front_V=np.vstack((next_front_V, V[s,:]))
        i += 1
        
        if len(next_front_X)!=0:
            fronts_X[i] = next_front_X
            fronts_V[i] = next_front_V
        else:
            break
    
               
    return fronts_X, fronts_V, rank

def calculate_crowding_distance(I,fitness):
    
    m=2
    l=len(I)
    distance=np.zeros((l))
    
    for p in range(len(I)):
        distance[p] = 0
        
    for obj_index in range(m):
        #I=sort_objective(I,fitness[:,obj_index])
        distance[0]=distance[l-1]=float('inf')
        
        f_max=max(fitness[obj_index])
        f_min=min(fitness[obj_index])        
        
        for i in xrange(1,l-1):
            distance[i]= distance[i]+ (fitness[i+1,obj_index]-fitness[i-1,obj_index])/(f_max-f_min)
      
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
            
def select_particles(fronts_X,fronts_V,rank, pop_size,no_var, fitness):
    
    dist={}
    for each in fronts_X:
        dist[each]=calculate_crowding_distance(fronts_X[each],fitness)
    
    X_next=np.empty((0,no_var))
    V_next=np.empty((0,no_var))
    i=1
    
      
    while len(X_next)+len(fronts_X[i])<=pop_size:
                
        #g=np.array(fronts[i])
        #print pop_next.shape, g.shape
        X_next=np.concatenate((X_next, fronts_X[i]))
        V_next=np.concatenate((V_next, fronts_V[i]))
        i=i+1
    
    P=sort_crowding(fronts_X[i], rank, dist[i])
    
    X_next=np.vstack((X_next, P[:(pop_size-len(X_next))]))
    V_next=np.vstack((V_next, P[:(pop_size-len(V_next))]))
    
    return X_next, V_next
    

    
   
   
def NSPSO(max_gen,pop_size,no_var,w,C1,C2,search_space):
    
    X,V=create_population(pop_size,no_var,search_space)  
    fitness=evaluate_fitness(X,pop_size,no_var, search_space)
    fronts_X,fronts_V,rank=non_dominated_sort(X,V,fitness)
    
    #print fitness
    gen=0  
    while gen< max_gen:
        #print gen
        gen=gen+1
        
        V_next=np.zeros(V.shape)
        X_next=np.zeros(X.shape)        
        
        X_gbest= fronts_X[1][0]
        X_lbest= X[np.argmin(fitness.all())]
        
        for i in xrange(pop_size):
            
            
            
            r1=np.random.random()
            r2=np.random.random()            
            
            g1=C1*r1*(X_lbest-X[i,:])
            g2=C2*r2*(X_gbest-X[i,:])             
            
            #print i,g1,g2
            V_next[i,:]=w*V[i,:] + g1 + g2         
            X_next[i,:]=X[i,:]+V[i,:]
            
            #print V_next,X_next
            
        #print V_next
        fitness_next=evaluate_fitness(X_next,pop_size,no_var, search_space) 
            
        X=np.vstack((X,X_next))
        V=np.vstack((V,V_next))
        fitness=np.vstack((fitness,fitness_next))
        
        #print fitness
        fronts_X,fronts_V,rank=non_dominated_sort(X,V,fitness)
        #print fronts_X
        X,V=select_particles(fronts_X,fronts_V,rank, pop_size,no_var, fitness)        
        fitness=evaluate_fitness(X,pop_size,no_var, search_space)
        #print fitness    
    return X,V,fitness        
        
        

if __name__=="__main__":
    
    max_gen=20
    pop_size=100
    
    
    no_var=1
    w=0.5
    C1=1
    C2=3
    
    search_space=[-10,10]
    
    X,V,fitness=NSPSO(max_gen,pop_size,no_var,w,C1,C2,search_space)
    colors=np.random.rand(pop_size)
    
    plt.scatter(fitness[:,0],fitness[:,1], c=colors) 
   
    d=np.zeros(pop_size)    
    for i in xrange(pop_size):
        min_d=np.inf
        s=0
        for k in xrange(pop_size):
            if i!=k:
                for m in xrange(2):
                    s=s+np.abs(fitness[i,m]- fitness[k,m])
                if s<min_d:
                    min_d=s
        d[i]=min_d        
    
    d_mean=np.mean(d)
    N=pop_size
    GD=np.sqrt(np.sum(d**2)/N)
    
    S=np.sqrt((1.0/N)* np.sum((d-d_mean)**2))
    
    g1=(max(fitness[:,0])-min(fitness[:,0]))
    g2=(max(fitness[:,1])-min(fitness[:,1]))
    g=g1+g2
    
    Del=(g+np.sum(d-d_mean))/(g+N*d_mean)
    
    
    print "Generational distance GD:", GD
    print "Spread S:", S
    print "Spread Del", Del