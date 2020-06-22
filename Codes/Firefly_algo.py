import numpy as np

def create_fireflies(pop_size,D,search_space):
    return np.ones((pop_size,D))*search_space[0]+ np.random.rand(pop_size,D)*(np.ones((pop_size,D))*(search_space[1]-search_space[0]))
    
def evaluate_light(ff,pop_size):
    light=np.zeros(pop_size)
    for i in xrange(pop_size):
        x=ff[i,0]
        y=ff[i,1]
        light[i]= (x-1)**2 + 100.0*(y-x**2)**2 #(x+2*y-7)**2 + (2*x+y -5)**2
    
    return light    

def sort_light(ff,light):    

    ind=np.argsort(light)
    
    return ff[ind], light[ind]

def ff_move(a,b,r,alpha,gamma,search_space):
    
    beta0=1
    beta=beta0*np.exp(-gamma*(r**2))
    
    for i in xrange(len(a)):
        a[i]=a[i]+beta*(a[i]-b[i])+alpha*np.random.uniform(low=0,high=0.5)

    return a

def distance(a,b):
    s=0
    for i in range(len(a)):
        s=s+(a[i]-b[i])**2
    
    return np.sqrt(s)
    

    
def firefly(pop_size,D,alpha,gamma,delta,search_space):
    
    ff=create_fireflies(pop_size,D,search_space)
    light=evaluate_light(ff,pop_size)
    
    ff,light=sort_light(ff,light)
    
    light_o=light[0]
    
    for q in range(100):
        for i in xrange(pop_size):
            for j in xrange(pop_size):
                if light[i]<light[j]:
                    r=distance(ff[i],ff[j])
                    ff_move(ff[i],ff[j],r,alpha,gamma,search_space)
                    light[i]=light[i]*np.exp(-gamma*(r**2))
                    
                    light=evaluate_light(ff,pop_size)
                    ff,light=sort_light(ff,light)
                    
                alpha=alpha*delta
                light_o=light[0]
                
    print ff[0,:],light[0]

def main():
    alpha=0.2
    gamma=0.9
    delta=0.8
    
    pop_size=10
    D=2
    search_space=[-5,5]
    
    firefly(pop_size,D,alpha,gamma,delta,search_space)


main()    