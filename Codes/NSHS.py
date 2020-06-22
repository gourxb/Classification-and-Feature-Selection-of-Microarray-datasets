import numpy as np
import matplotlib.pyplot as plt

def objective1(x):
    '''
    return x[0]
    
    f1=0
    for i in xrange(len(x)-1):
        f1=f1+(-10*np.exp(-0.2*np.sqrt(x[i]**2 + x[i+1]**2)))
    
    return f1'''
    return x**2 
    

def objective2(x):
    '''
    g=1+9*sum(x[1:])/(len(x)-1)

    return g*(1-(x[0]/g)**2)        
    
    f2=0
    for i in xrange(len(x)):
        f2=f2+(np.abs(x[i])**0.8 + 5*np.sin(x[i]**3))
        
    return f2'''
    return (x-2.0)**2

    


def calculate_objectives(hm,hms,no_var, search_space):

    objectives=np.zeros((hms, 2))
    for i in xrange(hms):
        objectives[i,:]= [objective1(hm[i,:]), objective2(hm[i,:])]

    return objectives    
    
def create_harmony(hms,no_var,search_space):
    
    hm=np.zeros((hms,no_var))
    min_val=search_space[0]
    max_val=search_space[1]
    
    for i in xrange(hms):
        for j in xrange(no_var):
            hm[i,j]=min_val+ np.random.random()*(max_val- min_val)
            
    return hm
    
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
    #print len(S[2]),len(S[2][0])
    while len(fronts[i]) != 0:
        next_front = np.empty((0,len(pop[0])))
        
        for r in fronts[i]:
            for q in S:
                n[q] -= 1
               
                if n[q] == 0:
                    rank[q]=i+1
                    next_front=np.vstack((next_front,pop[q,:]))
        
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
            
def select_harmonies(fronts,rank, hms, objectives,no_var):
    
    dist={}
    for each in fronts:
        dist[each]=calculate_crowding_distance(fronts[each],objectives)
    
    hm_next=np.empty((0,no_var))
    
    i=1
    while len(hm_next)+len(fronts[i])<=hms:
                
        #g=np.array(fronts[i])
        #print hm_next.shape, g.shape
        hm_next=np.vstack((hm_next, fronts[i]))
        i=i+1
        #print hm_next.shape
    P=sort_crowding(fronts[i], rank, dist[i])
    
    hm_next=np.vstack((hm_next, P[:(hms-len(hm_next))]))
    
    return hm_next
    
def improvise(hm,k,hms,hmcr,par,bw,search_space,no_var):
        
    
    w_new=np.zeros(no_var)
    
    min_val=search_space[0]
    max_val=search_space[1]
    
    r1=np.random.random()
    r2=np.random.random()
    #r=random.random()
    if(r1<hmcr):
        for col in xrange(no_var):
            a=np.random.randint(0,hms-1)
            #print hm[a,col]
            #print a,col
            w_new[col]=hm[a,col]
          
            
        if(r2<par):
            r3=np.random.random()
            for i in range(no_var):
                if(r3<0.5):
                    w_new[i]=hm[k,i]+ r3*bw
                
                else:
                    w_new[i]=hm[k,i]- r3*bw
            
             
    else:
        for i in xrange(no_var):
            w_new[i]=min_val+ np.random.random()*(max_val- min_val)
        
                
    return w_new

def NSHS(search_space, max_gen,no_var, hms,  hmcr, par ,bw):
    hm=create_harmony(hms,no_var,search_space)
    objectives=calculate_objectives(hm,hms,no_var, search_space)
    
    for j in xrange(1):
        hm_new=np.zeros(hm.shape)
        for k in xrange(hms):
            hm_new[k,:]= improvise(hm,k,hms,hmcr,par,bw,search_space,no_var)
        
        objectives_new=calculate_objectives(hm_new,hms,no_var, search_space)                 
        
        hm=np.vstack((hm,hm_new))
        objectives=np.vstack((objectives,objectives_new))
        
        
        fronts,rank=fast_nondominated_sort(hm, objectives)
        
        
        hm=select_harmonies(fronts,rank, hms, objectives, no_var)
        objectives=calculate_objectives(hm,hms,no_var, search_space)
        
    return hm,objectives

    
if __name__=="__main__":
    
    no_var=1
    search_space=np.array([-100,100])
    max_gen = 20
    pop_size=hms = 50
    hmcr=0.9
    par=0.5
    bw=0.02
    
    
    colors=np.random.rand(hms)    
    
    hm, objectives = NSHS(search_space, max_gen,no_var, hms, hmcr, par,bw)
    #print hm, objectives
    plt.scatter(objectives[:,0],objectives[:,1], c=colors) 

    d=np.zeros(pop_size)    
    for i in xrange(pop_size):
        min_d=np.inf
        s=0
        for k in xrange(pop_size):
            if i!=k:
                for m in xrange(2):
                    s=s+np.abs(objectives[i,m]- objectives[k,m])
                if s<min_d:
                    min_d=s
        d[i]=min_d        
    
    d_mean=np.mean(d)
    N=pop_size
    GD=np.sqrt(np.sum(d**2)/N)
    
    S=np.sqrt((1.0/N)* np.sum((d-d_mean)**2))
    
    g1=(max(objectives[:,0])-min(objectives[:,0]))
    g2=(max(objectives[:,0])-min(objectives[:,0]))
    g=g1+g2
    
    Del=(g+np.sum(d-d_mean))/(g+N*d_mean)
    
    
    print "Generational distance GD:", GD
    print "Spread S:", S
    print "Spread Del", Del