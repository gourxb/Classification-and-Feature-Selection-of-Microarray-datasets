import numpy as np
import random
from sklearn import metrics,neighbors
from sklearn import cross_validation as cv


def check_functions(w,data,data_label):
    
    f=np.where(w==1)
    sample_data=data[:,f[0]]
    
    train, test, train_label, test_label = cv.train_test_split(sample_data,data_label, test_size=0.25, random_state=545)
    
    clf=neighbors.KNeighborsClassifier(algorithm='ball_tree',weights='uniform')    
       
    clf.fit(train,train_label)
    pr=clf.predict(test)
    acc=metrics.accuracy_score(test_label,pr)  
    
    return [len(f[0]),acc]
   

def non_dominated_sort(hm,cardinality, accuracy):
        
    fronts = {}
    
    S = {}
    n = {}
    for s in xrange(len(hm)):
        S[s] = []
        n[s] = 0
        
    fronts[1] = []
    
    for p in xrange(len(hm)):
        for q in xrange(len(hm)):
            if p == q:
                continue
            
            #if p dominates
            if cardinality[p]<cardinality[q] and  accuracy[p]>accuracy[q]:
            #Add q to set of solution dominateed by p            
                S[p].append(hm[q,:])
            
            elif cardinality[p]>cardinality[q] and  accuracy[p]<accuracy[q]:
                n[p] += 1
        
        if n[p] == 0:
            fronts[1].append(hm[p,:])
    
    i = 1
    
    while len(fronts[i]) != 0:
        next_front = []
        
        for r in fronts[i]:
            for s in S:
                n[s] -= 1
                if n[s] == 0:
                    next_front.append(s)
        
        i += 1
        fronts[i] = next_front
                
    return hm[fronts[1],:], cardinality[fronts[1]], accuracy[fronts[1]]

   
    
    
def create_harmony(hms,tf):
     
    hm=np.round(np.random.rand(hms,tf))
    ''''    
    for i in xrange(hms):
        for j in xrange(tf):
            if (random.random()<0.005):
                #print i,j
                hm[i,j]=1
    '''
    return hm

def improvise(hm,k):
    
    
    hmcr=0.5
    par=0.7
    
    w_new=np.zeros((1,len(hm[0])))
    
    r1=random.random()
    r2=random.random()
    #r=random.random()
    if(r1<hmcr):
        for col in xrange(len(hm[0])):
            a=random.randint(0,len(hm)-1)
            #print hm[a,col]
            #print a,col
            w_new[0,col]=hm[a,col]
          
            
        if(r2<par):
            #print hm.shape,k
            f=np.where(hm[k,:]==1)
            for each in f[0]:
                    if each!=len(hm[0])-1:
                        w_new[0,each]=0
                        w_new[0,each+1]=1
                    elif each!=0:
                        w_new[0,each]=0
                        w_new[0,each-1]=1
             
    else:
        '''
        for i in xrange(len(hm[0])):
            if (random.random()<0.005):                  
                w_new[0,i]=1
        '''        
    return w_new

def prediction(w,data,data_label):
    
    
    f=np.where(w==1)
    sample_data=data[:,f[0]]
    
    train, test, train_label, test_label = cv.train_test_split(sample_data,data_label, test_size=0.25, random_state=545)
    
    clf=neighbors.KNeighborsClassifier(algorithm='ball_tree',weights='uniform')    
       
    clf.fit(train,train_label)
    prediction=clf.predict(test)
    accuracy=metrics.accuracy_score(test_label,prediction) 
    
    print "testing accuracy",accuracy 
       
    print data.shape, sample_data.shape
    
    data_label=data_label.reshape((len(data_label), 1))    
    sample_data=np.hstack((sample_data, data_label)) 
    np.savetxt("HSFS_mulitobj.csv", sample_data, delimiter=",")
    
def NSHS():
    
    f="C:\\Users\\USER\\Desktop\\Research codes\\micro_ array_datasets\\ProstateCancer\\prostate\\prostate_TumorVSNormal.csv"
    data=np.genfromtxt(open(f,'r'), delimiter=',')
    data[np.isnan(data)]=0
    
    data_label=data[:,-1]
    data=data[:,:-1]    
    
    hms=5    
    
    tf=len(data[0])
    hm=create_harmony(hms,tf)
    x_new=np.zeros((1,len(hm)))
    
    accuracy=np.zeros(hms)
    cardinality=np.zeros(hms)   
    
    for i in xrange(hms):
       cardinality[i], accuracy[i]= check_functions(hm[i],data,data_label)
    
    #print hm.shape,cardinality.shape, accuracy.shape
    for j in xrange(10):
        hm_new=np.empty((0,tf))
        
        for k in xrange(len(hm)):
            x_new=improvise(hm,k)
            c,a=check_functions(x_new,data,data_label)
            hm_new=np.vstack((hm_new, x_new))
            cardinality=np.append(cardinality,[c])
            accuracy=np.append(accuracy,[a])
            
        hm=np.vstack((hm, hm_new))
        #print hm.shape,cardinality.shape, accuracy.shape
        hm,cardinality, accuracy=non_dominated_sort(hm,cardinality, accuracy)
        
    #print hm
    print accuracy
    print cardinality
    prediction(hm[0,:],data,data_label)
NSHS()   
        
