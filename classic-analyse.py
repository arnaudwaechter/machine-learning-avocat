#!/usr/local/bin/python3
#
import pandas as pa
import csv as csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
from scipy.stats import pearsonr
from sklearn import svm, datasets
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.covariance import empirical_covariance
from sklearn.feature_extraction.text import CountVectorizer # bag of words
import re
import pdb # debugger

#pdb.set_trace()


##################
# Function Stacker
#
def stacker( csv_file ):
	XRAW=csv_file.loc[:,'description':'nb']
	X=[]
	y=[]
	name=[]

	name.append("taille_description")

	for i, line in XRAW.iterrows():

		nb		=	line["nb"]
		taille	=	len(str(line["description"]))
		divorce = line["description"].find("divorce")
		euro = line["description"].find("euro")
		prix = line["description"].find("prix")
		amiable = line["description"].find("amiable")

		xline=[taille, divorce, euro,prix,amiable]
		X.append(xline)
		y.append(nb)

	X=np.array(X, dtype=('i4' ))
	y=np.array(y, dtype=('i4' ))

	# on normalise les features
	#X = normalize(X);
	return name,X,y

#########################################################################################

# import some data to play with
csv_file = pa.read_csv('./q4a_ai.csv', index_col='id') 

# call the stacker
name,X,y = stacker( csv_file )

# set train, crossvalid and test data
X_train, X_cv, y_train, y_cv = train_test_split( X, y, test_size=0.3, random_state=1972)

print("Train : "+str(len(X_train))+"   CrossValid : "+str(len(X_cv)))

##### MINORITY REPORT
####################################################
# draw learning curve
if (1==1):
	C = 0.8  # SVM regularization parameter
	Jtrain=np.zeros(len(X_train))
	Jcv=np.zeros(len(X_train))

	print("training")
	clf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1).fit(X_train, y_train)
	print("testing")
	Jcv = mean_squared_error(y_cv, clf.predict(X_cv))
	print(Jcv)
	exit()

	for n in range(10, len(X_train),10):
		clf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1).fit(X_train[:n], y_train[:n])
	#	clf = RandomForestClassifier(n_estimators=400).fit(X_train[:n], y_train[:n])
	#	clf = ExtraTreesClassifier(n_estimators=400).fit(X_train[:n], y_train[:n])
		Jtrain[n] = mean_squared_error(y_train, clf.predict(X_train))
		Jcv[n] = mean_squared_error(y_cv, clf.predict(X_cv))

	# create a mesh to plot in
	x_min=0
	x_max=len(X_train)
	y_min=0
	y_max=Jcv[:].max()+1

	x=range(10, len(X_train), 10)
	plt.plot(x, Jtrain[x], "g" )
	plt.plot(x, Jcv[x], "r" )
	# plt.plot(x, Precision[x], "b--" )
	#plt.plot(x, Recall[x], "g--" )
	#plt.plot(x, Precision[x], "r--" )

	plt.show()

#### GO FOR GLORY
#####################################################
if (1==0):
	csv_file = pa.read_csv('./test.csv', index_col='PassengerId') 

	# call the stacker
	name,Xtest = stacker( csv_file )
	Z = clf.predict(Xtest)
	print("The final result")
	print(Z)

	# write the submission file
	c=open('submit.csv', 'w')
	c.write( 'PassengerId,Survived\n')
	i=0
	for y in Z:
		c.write( str(csv_file.index[i])+","+str(y)+"\n" )
		i=i+1
	c.close()

	print("File submit.csv done")

# feature min, max, mean, mod, survival rate per mode
###################################################
if (1==0):
	for m in range(0, len(name)+1 ):
		mean=np.mean(X_train[:,m])
		max=np.amax(X_train[:,m])
		min=np.amin(X_train[:,m])
		uniq=np.unique(X_train[:,m], return_counts=True)
		nonzero=np.count_nonzero(X_train[:,m])
		x=np.zeros(max)
		tot=0
		print("["+str(m)+"] "+name[m]+" --> min:"+str(min)+\
			" max:"+str(max)+" moy:"+str(mean)+" uniq:"+str(uniq)+\
			" nonzero:"+str(nonzero/len(X_train)*100)+" %")
		plt.title(name[m]+" vs target")
		good=np.zeros(max+1)
		bad=np.zeros(max+1)
		ly=len(X_train)
		for cur in range(0, ly):
			if (y[cur]>0 ):
				good[ X_train[cur,m] ] +=1
			if (y[cur]==0 ):
				bad[ X_train[cur,m] ] +=1
		x=range(0, max+1)
		plt.plot(x, (100*good[x]/(good[x]+bad[x])), "g" )
		plt.show()
			
	exit()


# matrice de convergence
#############################################################
if (1==0):
	mco= empirical_covariance( X_cv )
	print("Matrice de covariance entre les features")
	mcopa = pa.DataFrame(mco)
	mcopa.columns = name
	print(mcopa)
	exit

# matrice de confusion (target prévu vs target reelle)
########################################################
if (1==0):
	confu = confusion_matrix( y_cv, clf.predict(X_cv) )
	print("Confusion matrice (00=vrai negatif / 11=vrai positif / 01=faux positif...)")
	print(confu)
	exit

# Feature importance of the forest
#####################################
if (1==0):
	forest = ExtraTreesClassifier(n_estimators=25, random_state=0)

	forest.fit(X_cv, y_cv)
	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_],
       	      axis=0)
	indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	print("Feature ranking:")

	for f in range(X_cv.shape[1]):
	    print("%d. feature %d (%f) %s " % \
	    	(f + 1, indices[f], importances[indices[f]], name[ indices[f] ]))

	# Plot the feature importances of the forest
	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(X_cv.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
	plt.xticks(range(X_cv.shape[1]), indices)
	plt.xlim([-1, X.shape[1]])
	plt.show()
	

# COLD CASE
# all the case where the prediction was wrong
############################################
if (1==0) :
	print("Cold Case : all case wrong...")
	print("PassengerId Pclass Name Sex Age SibSp Parch Fare Cabin Embarked")
	clf = RandomForestClassifier(n_estimators=40).fit(X_train, y_train)
	y=y_cv.reshape(1,-1)
	for idx in range(0, len(X_cv)):
		if ( y[0,idx] != clf.predict(X_cv[idx])):
			print( "No:"+str(idx)+" y:"+str(y[0,idx])+" X:"+str(X_cv[idx]) )
			
	exit()


#########################################################################################



			



