import random
from sklearn import metrics,preprocessing,svm,neighbors,grid_search
import numpy as np
from sklearn import cross_validation as cv


def check_fitness(w,data,data_label):
    
    f=np.where(w==1)
    sample_data=data[:,f[0]]
    
    train, test, train_label, test_label = cv.train_test_split(sample_data,data_label, test_size=0.25, random_state=545)
    
    clf=neighbors.KNeighborsClassifier(algorithm='ball_tree',weights='uniform')    
       
    clf.fit(train,train_label)
    pr=clf.predict(test)
    acc=metrics.accuracy_score(test_label,pr)  
    #print 1.0/len(f[0])
    fit=10*(1.0/len(f[0]))+acc 
    return fit
    
           
    
def create_harmony_memory(hm_len,tf):
    #hm=np.round(np.abs(np.random.rand(hm_len,tf)-np.random.rand(hm_len,tf)))
    hm=np.zeros((hm_len,tf))    
    for i in xrange(hm_len):
        for j in xrange(tf):
            if (random.random()<0.01):
                #print i,j
                hm[i,j]=1
    
    return hm
    
def sort_harmony(fitness,hm):
    
    ind=np.argsort(fitness)[::-1]
    fitness=fitness[ind]
    hm=hm[ind]   
    
    
    return fitness,hm
    
def update_harmony(fitness,hm,data, data_label):
    
    hmcr=0.9
    par=0.5
    
      
    
    hm_len=len(hm) 
    
   
    
    r1=random.random()
    r2=random.random()
    
    for q in range(20):
    
        for j in range(hm_len):
            w_new=np.zeros((len(hm[0])))
            if(r1<hmcr):
                for col in xrange(len(hm[0])):
                    a=random.randint(0,len(hm)-1)
                    
                    w_new[col]=hm[a,col]
                  
                    
                if(r2<par):
                    f=np.where(hm[j]==1)
                    for each in f[0]:
                            if each!=len(hm[0])-1:
                                w_new[each]=0
                                w_new[each+1]=1
                            elif each!=0:
                                w_new[each]=0
                                w_new[each-1]=1
                     
            else:
               #w_new[:]= np.round(np.abs(np.random.rand(1,len(hm[0]))-np.random.rand(1,len(hm[0]))))
                                
                for i in xrange(len(hm[0])):
                    if (random.random()<0.01):                  
                        w_new[i]=1
               
            fit=check_fitness(w_new,data, data_label)
            
            if(fit>fitness[-1]):
                hm[-1,:]=w_new
                fitness[-1]=fit
                fitness,hm=sort_harmony(fitness,hm)                       
            
           
    return fitness,hm       

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
    
def main(): 
    #f="C:\\Users\\USER\\Desktop\\Research codes\\micro_ array_datasets\\ColonTumor\\colonTumor.csv"
    #f="C:\\Users\\USER\\Desktop\\Research codes\\micro_ array_datasets\\BreastCancer\\BreastCancer\\breastCancer.csv"
    #f="C:\\Users\\USER\\Desktop\\Research codes\\micro_ array_datasets\\ALL-AML_Leukemia_2\\ALL-AML_Leukemia\\AMLALL.csv"
    f="C:\\Users\\USER\\Desktop\\Research codes\\micro_ array_datasets\\ProstateCancer\\prostate\\prostate_TumorVSNormal.csv"
    data=np.genfromtxt(open(f,'r'), delimiter=',')
    data[np.isnan(data)]=0
    
    data_label=data[:,-1]
    data=data[:,:-1]
    
    
    
    

    hm_len=10
    
    tf=len(data[0])
    fitness=np.zeros(hm_len)
    
    hm=create_harmony_memory(hm_len,tf) 
    
    for i in range(hm_len):
       fitness[i]=check_fitness(hm[i],data,data_label)
    
    fitness,hm=sort_harmony(fitness,hm)
    fitness,hm=update_harmony(fitness,hm,data, data_label)
    print "Training fitness",fitness[0]
     
    #print fitness
    prediction(hm[0,:],data,data_label)
            
main()