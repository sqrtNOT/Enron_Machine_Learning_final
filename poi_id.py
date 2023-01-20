#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select features
features_list=['poi','expenses','to_poi_fraction','from_poi_fraction','shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL')

### Task 3: Create new features
for i in data_dict:
	if(data_dict[i]['from_poi_to_this_person']=="NaN" or data_dict[i]['to_messages']=="NaN"):
		data_dict[i]['from_poi_fraction']="NaN"
	else:
		data_dict[i]['from_poi_fraction']=float(data_dict[i]['from_poi_to_this_person'])/float(data_dict[i]['to_messages'])
        if(data_dict[i]['from_this_person_to_poi']=="NaN" or data_dict[i]['from_messages']=="NaN"):
                data_dict[i]['to_poi_fraction']="NaN"
        else:
                data_dict[i]['to_poi_fraction']=float(data_dict[i]['from_this_person_to_poi'])/float(data_dict[i]['from_messages'])

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
#from sklearn.neural_network import MLPClassifier
#clf = MLPClassifier(random_state=1234,hidden_layer_sizes=(100,100),solver='adam',alpha=1,max_iter=1000)

#from sklearn.svm import SVC
#clf=SVC(random_state=1234,kernel="poly",degree=4,C=0.1,max_iter=1000)

### Task 5: Tune the chosen classifier
#from sklearn.tree import DecisionTreeClassifier
#clf=DecisionTreeClassifier(random_state=1234,criterion="entropy",splitter="best",max_depth=3,min_samples_leaf=3)

#generate testing and training sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train,labels_train)
#print(clf.feature_importances_)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

