#!/usr/bin/env python
# coding: utf-8

# In[94]:


# Importing the Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import neattext.functions as nfx
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn import preprocessing 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
from textblob import TextBlob
from collections import defaultdict
from wordcloud.wordcloud import WordCloud,STOPWORDS
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')


# In[95]:


# Loading the dataset
data = pd.read_csv('reviews.csv')
data


# In[97]:


data.info


# In[3]:


print("The shape of Data is (row,column) : "+str(data.shape))


# # EDA

# In[4]:


# Handling Nan Values
data.isna().sum()


# In[5]:


data = data[data.Text.notnull()]
print(data)


# In[6]:


print("The shape of Data is (row,column) : "+str(data.shape))


# In[7]:


# Handling Nan Values
data.isna().sum()


# In[8]:


# Polarity
data['Polarity'] = data['Text'].map(lambda text: TextBlob(text).sentiment.polarity)
data


# In[9]:


def polarity(row):
    '''This function returns sentiment value based on the overall ratings from the users'''
    if row['Polarity'] > 0.0:
        val = 'Positive'
    elif row['Polarity'] < 0.0:
        val = 'Negative'
    elif row['Polarity'] == 0.0:
        val = 'Neutral'
    else:
        val = -1
    return val


# In[10]:


# Applying the function in our new column
data['Sentiment'] = data.apply(polarity,axis=1)
data


# In[11]:


data.drop(['Polarity'], axis=1,inplace=True)
data


# In[12]:


data['Sentiment'].value_counts()


# In[13]:


df = pd.DataFrame(data)
df.to_csv('amazon.csv', index=False)


# In[14]:


# again reading the csv file
df = pd.read_csv('amazon.csv')
df


# In[15]:


# Plot
sns.countplot(x='Sentiment',data=df)


# In[16]:


df['Clean_text'] = df['Text'].apply(nfx.remove_userhandles)
df['Clean_text'] = df['Clean_text'].apply(nfx.remove_stopwords)
df['Clean_text'] = df['Clean_text'].apply(nfx.remove_special_characters)


# In[17]:


df


# In[18]:


# Filtering data
review_pos = df[df['Sentiment']=='Positive'].dropna()
review_neu = df[df['Sentiment']=='Neutral'].dropna()
review_neg = df[df['Sentiment']=='Negative'].dropna()


# In[19]:


# Positive word cloud
text = review_pos["Text"]
wordcloud = WordCloud(width=3000,height=2000, background_color='black',stopwords=STOPWORDS).generate(str(text))

fig = plt.figure(figsize=(40,30),facecolor='k',edgecolor='k')
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[20]:


# Neutral Word Cloud
text = review_neu["Text"]
wordcloud = WordCloud(width=3000,height=2000, background_color='black',stopwords=STOPWORDS).generate(str(text))

fig = plt.figure(figsize=(40,30),facecolor='k',edgecolor='k')
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[21]:


#negative word cloud
text = review_neg["Text"]
wordcloud = WordCloud(width=3000,height=2000, background_color='black',stopwords=STOPWORDS).generate(str(text))

fig = plt.figure(figsize=(40,30),facecolor='k',edgecolor='k')
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[22]:


# Features and Labels split
Xfeatures = df['Clean_text']
ylabels = df['Sentiment']


# In[23]:


# Train and test split
# Split Data
x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=0.3,random_state=42)


# In[24]:


x_train.shape


# In[25]:


x_test.shape


# In[26]:


y_train.shape


# In[27]:


y_test.shape


# # Model Building

# ### 1. Logistics regression

# In[28]:


pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression(max_iter=200,C=1,multi_class='ovr'))])
pipe_lr.fit(x_train,y_train)


# In[29]:


pipe_lr_train = pipe_lr.predict(x_train)
print(pd.crosstab(y_train,pipe_lr_train))


# In[30]:


pipe_lr_test=pipe_lr.predict(x_test)
print(pd.crosstab(y_test,pipe_lr_test))


# In[31]:


accuracy_log_train=accuracy_score(y_train,pipe_lr_train)
accuracy_log_train


# In[32]:


accuracy_log_test=accuracy_score(y_test,pipe_lr_test)
accuracy_log_test


# In[33]:


print(classification_report(y_test,pipe_lr_test))


# In[34]:


f1_score_lg = f1_score(y_true=y_test,y_pred = pipe_lr_test,average='macro')
f1_score_lg


# In[35]:


recall_lg=recall_score(y_true=y_test,y_pred= pipe_lr_test,average='macro')
print(f"Recall score using Logistic Regression is {recall_lg}")


# In[36]:


precision_lg=precision_score(y_true=y_test,y_pred=pipe_lr_test,average='macro')
print(f"Precision Score using Logistic Regression is {precision_lg}")


# ### 2. SVM

# In[37]:


pipe_svm = Pipeline(steps=[('cv',CountVectorizer()),('svm',SVC(gamma=1,C=1,kernel='rbf',decision_function_shape='ovo'))])
pipe_svm.fit(x_train,y_train)


# In[38]:


pipe_svm_train = pipe_svm.predict(x_train)
print(pd.crosstab(y_train,pipe_svm_train))


# In[39]:


pipe_svm_test=pipe_svm.predict(x_test)
print(pd.crosstab(y_test,pipe_svm_test))


# In[40]:


accuracy_svm_train=accuracy_score(y_train,pipe_svm_train)
accuracy_svm_train


# In[41]:


accuracy_svm_test=accuracy_score(y_test,pipe_svm_test)
accuracy_svm_test


# In[42]:


print(classification_report(y_test,pipe_svm_test))


# In[43]:


f1_score_svm = f1_score(y_true=y_test,y_pred = pipe_svm_test,average='macro')
f1_score_svm


# In[44]:


recall_svm=recall_score(y_true=y_test,y_pred= pipe_svm_test,average='macro')
print(f"Recall score using Support Vectore Machine is {recall_svm}")


# In[45]:


precision_svm=precision_score(y_true=y_test,y_pred=pipe_svm_test,average='macro')
print(f"Precision Score using Support Vector Machine is {precision_svm}")


# ### 3. Naive Bayes

# In[46]:


pipe_nb = Pipeline(steps=[('cv',CountVectorizer()),('mb',MultinomialNB())])
pipe_nb.fit(x_train,y_train)


# In[47]:


pipe_nb_train = pipe_nb.predict(x_train)
print(pd.crosstab(y_train,pipe_nb_train))


# In[48]:


pipe_nb_test=pipe_nb.predict(x_test)
print(pd.crosstab(y_test,pipe_nb_test))


# In[49]:


accuracy_nb_train=accuracy_score(y_train,pipe_nb_train)
accuracy_nb_train


# In[50]:


accuracy_nb_test=accuracy_score(y_test,pipe_nb_test)
accuracy_nb_test


# In[51]:


print(classification_report(y_test,pipe_nb_test))


# In[52]:


f1_score_nb = f1_score(y_true=y_test,y_pred = pipe_nb_test,average='macro')
f1_score_nb


# In[53]:


recall_nb=recall_score(y_true=y_test,y_pred= pipe_nb_test,average='macro')
print(f"Recall score using Naive Bayes is {recall_nb}")


# In[54]:


precision_nb=precision_score(y_true=y_test,y_pred=pipe_nb_test,average='macro')
print(f"Precision Score using Naive Bayes is {precision_nb}")


# ### 4. Decision Tree

# In[55]:


pipe_dt = Pipeline(steps=[('cv',CountVectorizer()),('dt',DecisionTreeClassifier(class_weight='balanced',
                                criterion='entropy',
                                max_depth=3,
                                min_samples_split=6))])
pipe_dt.fit(x_train,y_train)


# In[56]:


pipe_dt_train = pipe_dt.predict(x_train)
print(pd.crosstab(y_train,pipe_dt_train))


# In[57]:


pipe_dt_test=pipe_dt.predict(x_test)
print(pd.crosstab(y_test,pipe_dt_test))


# In[58]:


accuracy_dt_train=accuracy_score(y_train,pipe_dt_train)
accuracy_dt_train


# In[59]:


accuracy_dt_test=accuracy_score(y_test,pipe_dt_test)
accuracy_dt_test


# In[60]:


print(classification_report(y_test,pipe_dt_test))


# In[61]:


f1_score_dt = f1_score(y_true=y_test,y_pred = pipe_dt_test,average='macro')
f1_score_dt


# In[62]:


recall_dt=recall_score(y_true=y_test,y_pred= pipe_dt_test,average='macro')
print(f"Recall score using Decision Tree is {recall_dt}")


# In[63]:


precision_dt=precision_score(y_true=y_test,y_pred=pipe_dt_test,average='macro')
print(f"Precision Score using Decision Tree is {precision_dt}")


# ### 5. Random Forest

# In[64]:


pipe_rf = Pipeline(steps=[('cv',CountVectorizer()),('rf',RandomForestClassifier(n_estimators=100,
                                 class_weight='balanced',
                                 criterion='entropy',
                                 max_depth=3,
                                 max_samples=0.7,
                                 min_samples_split=6))])
pipe_rf.fit(x_train,y_train)


# In[65]:


pipe_rf_train = pipe_rf.predict(x_train)
print(pd.crosstab(y_train,pipe_rf_train))


# In[66]:


pipe_rf_test=pipe_rf.predict(x_test)
print(pd.crosstab(y_test,pipe_rf_test))


# In[67]:


accuracy_rf_train=accuracy_score(y_train,pipe_rf_train)
accuracy_rf_train


# In[68]:


accuracy_rf_test=accuracy_score(y_test,pipe_rf_test)
accuracy_rf_test


# In[69]:


print(classification_report(y_test,pipe_rf_test))


# In[70]:


f1_score_rf = f1_score(y_true=y_test,y_pred = pipe_rf_test,average='macro')
f1_score_rf


# In[71]:


recall_rf=recall_score(y_true=y_test,y_pred= pipe_rf_test,average='macro')
print(f"Recall score using Random Forest is {recall_rf}")


# In[72]:


precision_rf=precision_score(y_true=y_test,y_pred=pipe_rf_test,average='macro')
print(f"Precision Score using Random Forest is {precision_rf}")


# ### 6. KNN

# In[73]:


pipe_knn = Pipeline(steps=[('cv',CountVectorizer()),('knn',KNeighborsClassifier(n_neighbors = 1))])
pipe_knn.fit(x_train,y_train)


# In[74]:


pipe_knn_train = pipe_knn.predict(x_train)
print(pd.crosstab(y_train,pipe_knn_train))


# In[75]:


pipe_knn_test=pipe_knn.predict(x_test)
print(pd.crosstab(y_test,pipe_knn_test))


# In[76]:


accuracy_knn_train=accuracy_score(y_train,pipe_knn_train)
accuracy_knn_train


# In[77]:


accuracy_knn_test=accuracy_score(y_test,pipe_knn_test)
accuracy_knn_test


# In[78]:


print(classification_report(y_test,pipe_knn_test))


# In[79]:


f1_score_knn = f1_score(y_true=y_test,y_pred = pipe_knn_test,average='macro')
f1_score_knn


# In[80]:


recall_knn=recall_score(y_true=y_test,y_pred= pipe_knn_test,average='macro')
print(f"Recall score using KNN is {recall_knn}")


# In[81]:


precision_knn=precision_score(y_true=y_test,y_pred=pipe_knn_test,average='macro')
print(f"Precision Score using KNN is {precision_knn}")


# ### 7. Bagging

# In[82]:


pipe_bag = Pipeline(steps=[('cv',CountVectorizer()),('bag',BaggingClassifier(n_estimators = 100))])
pipe_bag.fit(x_train,y_train)


# In[83]:


pipe_bag_train = pipe_bag.predict(x_train)
print(pd.crosstab(y_train,pipe_bag_train))


# In[84]:


pipe_bag_test=pipe_bag.predict(x_test)
print(pd.crosstab(y_test,pipe_bag_test))


# In[85]:


accuracy_bag_train=accuracy_score(y_train,pipe_bag_train)
accuracy_bag_train


# In[86]:


accuracy_bag_test=accuracy_score(y_test,pipe_bag_test)
accuracy_bag_test


# In[87]:


print(classification_report(y_test,pipe_bag_test))


# In[88]:


f1_score_bag = f1_score(y_true=y_test,y_pred = pipe_bag_test,average='macro')
f1_score_bag


# In[89]:


recall_bag=recall_score(y_true=y_test,y_pred= pipe_bag_test,average='macro')
print(f"Recall score using Bagging is {recall_bag}")


# In[90]:


precision_bag=precision_score(y_true=y_test,y_pred=pipe_bag_test,average='macro')
print(f"Precision Score using Bagging is {precision_bag}")


# # Model Comparison

# In[91]:


model_accuracy={'Model':pd.Series(['Logistic Regression','SVM','Naive Bayes','Decision Tree','Random Forest','KNN','Bagging']),
                'Train Accuracy':pd.Series([accuracy_log_train,accuracy_svm_train,accuracy_nb_train,accuracy_dt_train,accuracy_rf_train,accuracy_knn_train,accuracy_bag_train]),
               'Test Accuracy':pd.Series([accuracy_log_test,accuracy_svm_test,accuracy_nb_test,accuracy_dt_test,accuracy_rf_test,accuracy_knn_test,accuracy_bag_test]),
               'F1 Score':pd.Series([f1_score_lg,f1_score_svm,f1_score_nb,f1_score_dt,f1_score_rf,f1_score_knn,f1_score_bag]),
               'Precision':pd.Series([precision_lg,precision_svm,precision_nb,precision_dt,precision_rf,precision_knn,precision_bag]),
               'Recall Score':pd.Series([recall_lg,recall_svm,recall_nb,recall_dt,recall_rf,recall_knn,recall_bag])}

metrics_table=pd.DataFrame(model_accuracy)
metrics_table.sort_values(['F1 Score'],ascending=False)


# # Deployment

# In[92]:


# Save Model and Pipepline
import joblib
pipeline_file = open("app.pkl","wb")
joblib.dump(pipe_bag,pipeline_file)
pipeline_file.close()


# In[ ]:


get_ipython().system('streamlit run model.py')


# In[ ]:




