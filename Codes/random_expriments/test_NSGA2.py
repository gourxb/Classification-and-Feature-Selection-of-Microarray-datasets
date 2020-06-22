import numpy as np
import matplotlib.pyplot as plt

def objective1(vector):
    return sum(x**2 for x in vector)


def objective2(vector):
    return sum((x-2.0)**2 for x in vector)



def decode(bitstring, no_var, search_space, bits_per_param):
    
    v=[]
    count=0
    for i in range(no_var):
        min_val=search_space[0]
        max_val=search_space[1]
        
        
        param=bitstring[i*bits_per_param: i+1*bits_per_param][::-1]
        sum_s=0
        for j in xrange(len(param)):
            sum_s+= param[j]*(2**j)
        v.append(min_val + ((max_val-min_val)/((2.0**bits_per_param)-1.0)) * sum_s )
        count+=1
    return v

def calculate_objectives(pop,pop_size,no_var, search_space, bits_per_param):

    vector=np.zeros((pop_size,no_var))
    objectives=np.zeros((pop_size, 2))
    for i in xrange(pop_size):
        vector[i,:]=decode(pop[i,:],no_var, search_space, bits_per_param)
        objectives[i,:]= [objective1(vector[i,:]), objective2(vector[i,:])]

    return vector,objectives    
    
def create_population(pop_size,no_var,bits_per_param):

    return np.round(np.random.rand(pop_size,no_var*bits_per_param))
    
def dominates(p1,p2):
    
    if (p1 < p2).all():
        return True
    else:
        return False
    
def fast_nondominated_sort(pop,objectives):
    fronts = {}
    
    fronts[1] = np.empty((0,len(pop[0])))
    #rank=np.zeros(len(pop))
    #print fronts[1].shape
    #print pop.shape
    S={}
    n={} 
    rank={}
    
    for p in xrange(len(pop)):
        S[p] = []
        n[p] = 0
        for q in xrange(len(pop)):
                          
            if dominates(objectives[p,:],objectives[q,:]):
                S[p].append(pop[q,:])
            
            elif dominates(objectives[q,:],objectives[p,:]):
                n[p] += 1
        
        if n[p] == 0:
            rank[p]=0
            #print fronts[1], pop[p,:]
            fronts[1]=np.vstack((fronts[1],pop[p,:]))
            #print fronts[1]
    
    i = 1
    
    while len(fronts[i]) != 0:
        next_front = np.empty((0,len(pop[0])))
        
        for r in xrange(len(fronts[i])):
            for s in S:
                n[s] -= 1
               
                if n[s] == 0:
                    rank[s]=i+1
                    next_front=np.vstack((next_front,pop[s,:]))
        
        i += 1
        
        if len(next_front)!=0:
            fronts[i] = next_front
        else:
            break
    
               
    return fronts, rank

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
                       
   
def better(pop,distance,rank,r1,r2):
  if rank[r1] < rank[r2] :
      return pop[r1,:]
  else:
      return pop[r2,:]
  
  if rank[r1]==rank[r2]:    
      if distance[r1] > distance[r2]:
          return pop[r1,:]
      else:
          return pop[r2,:]
          
          

def point_mutation(bitstring):
     #print len(bitstring)
     if np.random.random() < (1.0/len(bitstring)):
         k=np.random.random_integers(0,len(bitstring)-1)
         #print bitstring
         bitstring[k]= int(bitstring[k]-1)  
         
     return bitstring    
         

def crossover(parent1, parent2, rate):
    
    if np.random.random()>rate:
        return parent1
    else:
        point = 1 + np.random.random_integers(len(parent1)-2)
        parent1=np.append(parent1[0:point],parent2[point:len(parent1)])
        return parent1
    
def reproduce(selected, pop_size, no_var, bits_per_param, p_cross):
    
    children = np.empty((0,no_var*bits_per_param))  
    
    while len(children)<pop_size:
        p1 = selected[np.random.random_integers(0,pop_size-1),:]
        p2 = selected[np.random.random_integers(0,pop_size-1),:]
        
        child=np.zeros((no_var*bits_per_param))
        for i in range(no_var):
            child[i*bits_per_param: i+1*bits_per_param] = crossover(p1[i*bits_per_param: i+1*bits_per_param], p2[i*bits_per_param: i+1*bits_per_param], p_cross)
            child[i*bits_per_param: i+1*bits_per_param] = point_mutation(child[i*bits_per_param: i+1*bits_per_param])
            
            
        children=np.vstack((children, child))
    
    return children    

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
            
def select_parents(fronts,rank, pop_size, objectives):
    
    dist={}
    for each in fronts:
        dist[each]=calculate_crowding_distance(fronts[each],objectives)
    
    pop_next=np.empty((0,no_var*bits_per_param))
    i=1
    
      
    while len(pop_next)+len(fronts[i])<=pop_size:
                
        g=np.array(fronts[i])
        #print pop_next.shape, g.shape
        pop_next=np.concatenate((pop_next, g))
        i=i+1
    
    P=sort_crowding(fronts[i], rank, dist[i])
    
    pop_next=np.vstack((pop_next, P[:(pop_size-len(pop_next))]))
    
    return pop_next
    


def NSGA(search_space, max_gens,no_var, pop_size, p_cross, bits_per_param):
    pop=create_population(pop_size,no_var,bits_per_param)

    vector,objectives=calculate_objectives(pop,pop_size,no_var, search_space, bits_per_param)
    fronts,rank=fast_nondominated_sort(pop, objectives)
    
    
    distance=calculate_crowding_distance(pop,objectives)
    #print distance,rank
    selected=np.zeros(pop.shape)
    for i in xrange(pop_size):
        r1=np.random.randint(0,pop_size-1)
        r2=np.random.randint(0,pop_size-1)
        selected[i,:]=better(pop,distance,rank,r1,r2)
    
    #print selected    
    children=reproduce(selected, pop_size, no_var, bits_per_param, p_cross) 
    vector_child,objectives_child=calculate_objectives(children,pop_size,no_var, search_space, bits_per_param)
    #print selected
    #print children
    
    for q in xrange(max_gens):
        #print pop.shape, children.shape
        R=np.vstack((pop,children))
        vector=np.vstack((vector,vector_child))
        objectives=np.vstack((objectives,objectives_child))
        
        #print R.shape, objectives.shape
        
        fronts,rank=fast_nondominated_sort(R, objectives)
               
        pop = select_parents(fronts,rank, pop_size, objectives)
        vector,objectives=calculate_objectives(pop,pop_size,no_var, search_space, bits_per_param)
        
        selected=np.zeros(pop.shape)
        for i in xrange(pop_size):
            r1=np.random.randint(0,pop_size-1)
            r2=np.random.randint(0,pop_size-1)
            selected[i,:]=better(pop,distance,rank,r1,r2)
        
        children = reproduce(selected, pop_size, no_var, bits_per_param, p_cross)
        vector_child, objectives_child=calculate_objectives(children,pop_size,no_var, search_space, bits_per_param)
    
    return vector, objectives
     
    
if __name__=="__main__":
    
    no_var=1
    search_space=np.array([10,-10])
    max_gens = 10
    pop_size = 100
    p_cross = 0.80
    bits_per_param=16
    
    colors=np.random.rand(pop_size)    
    
    vector, objectives = NSGA(search_space, max_gens,no_var, pop_size, p_cross, bits_per_param)
    #print len(objectives[:,0]), len(objectives[:,1])
    plt.scatter(objectives[:,0],objectives[:,1], c=colors) 
