import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.cluster import AffinityPropagation
from statistics import mean
import datetime
from dateutil.parser import parse
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from subprocess import call
from sklearn.tree._export import plot_tree
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


df = pd.read_csv('NFL play by play.csv')

main_df = df[['qtr', 'down', 'TimeSecs', 'yrdline100', 'ydstogo', 'ydsnet', 'GoalToGo', 'FirstDown', 'Touchdown', 'PlayType', 'ScoreDiff', 'AbsScoreDiff', 'EPA', 'WPA', 'Season']]



#edit df to only include 4th downs
four_down_df = main_df[main_df['down'] == 4].reset_index()

#edit df to calculate specific field positions, playtypes, and ydstogo
four_down_df = four_down_df[four_down_df['yrdline100'] > 34]
four_down_df = four_down_df[four_down_df['yrdline100'] < 51]
four_down_df = four_down_df[(four_down_df['PlayType'] != 'Punt') & (four_down_df['PlayType'] != 'Field Goal')]

print(four_down_df['EPA'].median())
print(four_down_df['EPA'].mean())

mean_punt_epa_list =[]
mean_go_epa_list = []
x_range = range(35, 81)


for i in range(35, 81):
	new_df = four_down_df[four_down_df['yrdline100'] == i]
	punt_df = new_df[new_df['PlayType'] == 'Punt']
	punt_mean_epa = punt_df['EPA'].mean()
	mean_punt_epa_list.append(punt_mean_epa)

	go_df = new_df[(new_df['PlayType'] != 'Punt') & (new_df['PlayType'] != 'Field Goal')]
	go_mean_epa = go_df['EPA'].mean()
	mean_go_epa_list.append(go_mean_epa)



#exclude QB Kneels, TOs, and No Plays

four_down_df = four_down_df[(four_down_df['PlayType'] != 'QB Kneel') & (four_down_df['PlayType'] != 'No Play') & (four_down_df['PlayType'] != 'Timeout') & (four_down_df['PlayType'] != 'Field Goal')]


# making pivot table to examine playtype EPA
counts = four_down_df['PlayType'].value_counts().to_dict()

print(counts)

epa_count = four_down_df.groupby(['qtr', 'PlayType']).EPA.mean().reset_index()

fourth_pivot = epa_count.pivot(columns = 'PlayType', index = 'qtr', values = 'EPA')

# adding columns to convert playtypes into numbers for training labels

def twotype(x):
	if x == 'Punt':
		return 1
	elif x == 'Field Goal':
		return 1
	else:
		return 2

four_down_df['PlayType2'] = four_down_df.PlayType.apply(lambda x: twotype(x))

def threetype(x):
	if x == 'Punt':
		return 1
	elif x == 'Field Goal':
		return 2
	else:
		return 3

four_down_df['PlayType3'] = four_down_df.PlayType.apply(lambda x: threetype(x))

# examing WPA for future project
# decisions & plays that have the biggest impact on games?

def wpa_pos_neg(x):
	if x >= 0:
		return 1
	else:
		return 0

four_down_df['added_wpa'] = four_down_df.WPA.apply(lambda x: wpa_pos_neg(x))


# dataset is too big to use entire play library of  ~30,000 fourth down plays
# slice dataset into specific quarters so that CPU can handle it


four_down_four_q_df = four_down_df[four_down_df['qtr'] == 4]

# filtering to only allow goal line situations

# labels and features for 2 vs 3 label decision trees

labels_three = four_down_four_q_df['PlayType3'].values

labels_two = four_down_four_q_df['PlayType2'].values

# labels to determine if a play helped or hurt a team's winning percentage

labels_wpa = four_down_four_q_df['added_wpa'].values

features = four_down_four_q_df[['TimeSecs', 'yrdline100', 'ydstogo', 'ScoreDiff']].values

# two label decision tree

model_two = tree.DecisionTreeClassifier()

x_two_train, x_two_test, y_two_train, y_two_test = train_test_split(features, labels_two, train_size = 0.8, test_size = 0.2, random_state = 6)

model_two.fit(x_two_train, y_two_train)

y_two_guesses = model_two.predict(x_two_test)

acc = accuracy_score(y_two_test, y_two_guesses)


# two label random forest between punt and go for it 

two_classifier = RandomForestClassifier(n_estimators = 1000, random_state = 0)

two_classifier.fit(x_two_train, y_two_train)

forest_two_predict = two_classifier.predict(x_two_test)

acc_forest_two = accuracy_score(y_two_test, forest_two_predict)

print(acc_forest_two)

## visualizing the random forest

feature_names = ['Time left in game', 'Distance to GL', 'Yards to first down', 'Score Differential']

label_three_names = ['Punt', 'Field Goal', 'Go For It']

label_two_names = ['Punt', 'Go For It']

feature_imp = pd.Series(two_classifier.feature_importances_, index= feature_names).sort_values(ascending = False)

sns.barplot(x = feature_imp, y = feature_imp.index)

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title('Deciding Factors on Deadzone 4th Downs')

plt.show()


# three label decision tree

model_three = tree.DecisionTreeClassifier()

x_three_train, x_three_test, y_three_train, y_three_test = train_test_split(features, labels_three, train_size = 0.8, test_size = 0.2, random_state = 6)

model_three.fit(x_three_train, y_three_train)

y_two_guesses = model_three.predict(x_three_test)

acc_three = accuracy_score(y_three_test, y_two_guesses)



# vizualizing the decision tree

feature_names = ['Time left in game', 'Distance to GL', 'Yards to first down', 'Goal to Go', 'Score Differential']

label_three_names = ['Punt', 'Field Goal', 'Go For It']


## building a random forest to see how it compares to the decision tree accuracy 

classifier = RandomForestClassifier(n_estimators = 1000, random_state = 0)

classifier.fit(x_three_train, y_three_train)

forest_three_predict = classifier.predict(x_three_test)

acc_forest = accuracy_score(y_three_test, forest_three_predict)



## building a random forest to predicth whether a given 4th down situation will increase of decrease a team's chance to win the game

wpa_classifier = RandomForestClassifier(n_estimators = 1000, random_state = 4)

training_features, test_features, training_labels, test_labels = train_test_split(features, labels_wpa, train_size = 0.8, test_size = 0.2, random_state = 5)

wpa_classifier.fit(training_features, training_labels)

predicted_wpa_labels = wpa_classifier.predict(test_features)

wpa_accuracy = accuracy_score(test_labels, predicted_wpa_labels)





