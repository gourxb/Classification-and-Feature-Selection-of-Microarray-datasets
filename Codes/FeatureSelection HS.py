import random
from sklearn import metrics,preprocessing,svm,neighbors,grid_search
import numpy as np
from sklearn import cross_validation as cv


def check_accuracy(w,nf,clf,tr,ts,tr_label,ts_label):
    
    
      
    
    sample_tr=np.zeros((len(tr),nf)) 
    sample_ts=np.zeros((len(ts),nf))
    
    for j in range(nf):
        for i in range(len(tr)):
            sample_tr[i,j]=tr[i,w[j]]
        for i in range(len(ts)):    
            sample_ts[i,j]=ts[i,w[j]]
    
       
    clf.fit(sample_tr,tr_label)
    pr=clf.predict(sample_ts)
    acc=metrics.accuracy_score(ts_label,pr)  
    
    return acc
           
    
def create_harmony_memory(hm_len,tf):
    hm=np.np.round(np.abs(np.random.rand(hm_len,tf)-np.random.rand(hm_len,tf)))
    
    
    return hm
    
def non_dominated_sort_harmony(accuracy,hm):
    #print rmse.shape
    ind=np.argsort(accuracy)[::-1]
    accuracy=accuracy[ind]
    hm=hm[ind]   
    
    #print "sort=",type(hm)
    return accuracy,hm
    
def update_harmony(accuracy,nf,clf,hm,train,test, train_label, test_label):
    
    hmcr=0.9
    par=0.5
    
    total_feature=len(train[0])    
    
    
    hm_len=len(hm) 
    
   
    w_new=np.zeros(nf)
    
    
    for q in range(20):
    
        for j in range(hm_len):
            
            r1=random.random()
            r2=random.random()
        
            if(r1<hmcr):
                a=random.randint(0,hm_len-1)
                w_new[:]=hm[a,:]
                  
                    
                if(r2<par and not((total_feature-1 or 0) in hm[j,:-1])):
                    if(random.random()<0.5):
                        w_new[:]=hm[j,:]+ 1
                        
                    else:
                        w_new[:]=hm[j,:]- 1
                     
            else:
               w_new[:]= random.sample(xrange(total_feature),nf)
               
            acc=check_accuracy(w_new,nf, clf,train,test, train_label, test_label)
            
            if(acc>hm[0,-1]):
                hm[0,:]=w_new
                accuracy[0]=acc
                                                     
            else : sort_harmony(accuracy,hm)
            
    return accuracy,hm       

def prediction(w,nf,train,train_label,test,test_label,t):
    
    sample_data=np.zeros((len(train)+len(test),nf))
    sample_train=np.zeros((len(train),nf)) 
    sample_test=np.zeros((len(test),nf))
    
    for j in range(nf):
        for i in range(len(train)):
            sample_train[i,j]=train[i, w[j]]
        for i in range(len(test)):    
            sample_test[i,j]=test[i, w[j]]
    
    train_label=train_label.reshape((len(train_label), 1))    
    test_label=test_label.reshape((len(test_label), 1))
    
    
    sample_train=np.hstack((sample_train, train_label)) 
    sample_test=np.hstack((sample_test, test_label))       
    sample_data=np.vstack((sample_train, sample_test)) 
    np.savetxt("reduced_datasets\\"+t+"_HSFS_"+ str(nf)+".csv", sample_data, delimiter=",")
    #print metrics.classfication_report(test)      
            
    '''    
    #parameters={'kernel':('rbf','poly'),'C':[10],'gamma':[0.2],'degree':[4,5]}
    #svr=svm.SVC()
    clf=neighbors.KNeighborsClassifier(algorithm='ball_tree',weights='uniform')
    sample_train=np.zeros((len(train),nf)) 
    sample_test=np.zeros((len(test),nf))
        
    for j in range(nf):
        for i in range(len(train)):
            sample_train[i,j]=train[i, w[j]]
        for i in range(len(test)):    
            sample_test[i,j]=test[i, w[j]]
    
    #clf=grid_search.GridSearchCV(svr,parameters)
    clf.fit(sample_train,train_label)
    
    prediction=clf.predict(sample_test)
    
    print metrics.accuracy_score(test_label,prediction)
    #print metrics.classification_report(test_label,prediction)
       
    print "Confusion Matrix"
    print metrics.confusion_matrix(test_label,prediction)
    fpr, tpr, thresholds = metrics.roc_curve(test_label,prediction)
    roc_auc = metrics.auc(fpr, tpr)
    print "Area under the ROC curve : %f" % roc_auc
    
    # Plot ROC curve
    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    '''
    

    
def main(nf,t,f): 
    #f="C:\\Users\\USER\\Desktop\\Research codes\\micro_ array_datasets\\ColonTumor\\colonTumor.csv"
    #f="C:\\Users\\USER\\Desktop\\Research codes\\micro_ array_datasets\\BreastCancer\\BreastCancer\\breastCancer.csv"
    #f="C:\\Users\\USER\\Desktop\\Research codes\\micro_ array_datasets\\ALL-AML_Leukemia_2\\ALL-AML_Leukemia\\AMLALL.csv"
    #f="C:\\Users\\USER\\Desktop\\Research codes\\micro_ array_datasets\\ProstateCancer\\prostate\\prostate_TumorVSNormal.csv"
    data=np.genfromtxt(open(f,'r'), delimiter=',')
    data[np.isnan(data)]=0
    
    data_label=data[:,-1]
    data=data[:,:-1]
    
    
    
    train, test, train_label, test_label = cv.train_test_split(data,data_label, test_size=0.25, random_state=545)

    hm_len=10
    
    tf=len(data[0])
    accuracy=np.zeros(hm_len)
    clf=neighbors.KNeighborsClassifier(algorithm='ball_tree',weights='uniform')

    
    hm=create_harmony_memory(hm_len,nf,tf) 
    
    for i in range(hm_len):
       accuracy[i]=check_accuracy(hm[i],nf,clf, train,test, train_label, test_label)
    
    accuracy,hm=sort_harmony(accuracy,hm)
    accuracy,hm=update_harmony(accuracy,nf,clf,hm,train,test, train_label, test_label)
    print "Training accuracy",accuracy[0]
    best_features=hm[0]
    fg.write("%s,%d" %(t,nf))

    for each in best_features:
        fg.write(",%d" %each)
    
    #print best_features
    prediction(best_features,nf,train,train_label,test,test_label,t)
            
