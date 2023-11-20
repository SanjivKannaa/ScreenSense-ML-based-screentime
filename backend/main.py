import pandas as pd
import numpy as np
from scipy.stats import randint
import seaborn as sns # used for plot interactive graph.
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

dataset=pd.read_csv('website_classification.csv')
dataset.shape


# dataset.head()

df = dataset[['website_url','cleaned_website_text','Category']].copy()
# df.head()

# pd.DataFrame(df.Category.unique()).values

"""Now we need to represent each category as a number, so as our predictive model can better understand the different categories."""

# Create a new column 'category_id' with encoded categories
df['category_id'] = df['Category'].factorize()[0]
category_id_df = df[['Category', 'category_id']].drop_duplicates()


# Dictionaries for future use
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Category']].values)

# New dataframe
# df.head()

# category_id_df

from wordcloud import WordCloud,STOPWORDS
plt.figure(figsize=(40,25))
subset = df[df['Category']=='Travel']
text = subset.cleaned_website_text.values
cloud1=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800
                       ).generate(" ".join(text))
plt.subplot(4,4,1)
plt.axis('off')
plt.title("Travel",fontsize=40)
plt.imshow(cloud1)
subset = df[df['Category']=='Social Networking and Messaging']
text = subset.cleaned_website_text.values
cloud2=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800
                       ).generate(" ".join(text))
plt.subplot(4,4,2)
plt.axis('off')
plt.title("Social Networking and Messaging",fontsize=40)
plt.imshow(cloud2)
subset = df[df['Category']=='News']
text = subset.cleaned_website_text.values
cloud3=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800
                       ).generate(" ".join(text))
plt.subplot(4,4,3)
plt.axis('off')
plt.title("News",fontsize=40)
plt.imshow(cloud3)

subset = df[df['Category']=='Streaming Services']
text = subset.cleaned_website_text.values
cloud4=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800
                       ).generate(" ".join(text))
plt.subplot(4,4,4)
plt.axis('off')
plt.title("Streaming Services",fontsize=40)
plt.imshow(cloud4)

subset = df[df['Category']=='Sports']
text = subset.cleaned_website_text.values
cloud5=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800
                       ).generate(" ".join(text))
plt.subplot(4,4,5)
plt.axis('off')
plt.title('Sports',fontsize=40)
plt.imshow(cloud5)

subset = df[df['Category']=='Photography']
text = subset.cleaned_website_text.values
cloud6=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800
                       ).generate(" ".join(text))
plt.subplot(4,4,6)
plt.axis('off')
plt.title("Photography",fontsize=40)
plt.imshow(cloud6)

subset = df[df['Category']=='Law and Government']
text = subset.cleaned_website_text.values
cloud7=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800
                       ).generate(" ".join(text))
plt.subplot(4,4,7)
plt.axis('off')
plt.title("Law and Government",fontsize=40)
plt.imshow(cloud7)

subset = df[df['Category']=='Health and Fitness']
text = subset.cleaned_website_text.values
cloud8=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800
                       ).generate(" ".join(text))
plt.subplot(4,4,8)
plt.axis('off')
plt.title("Health and Fitness",fontsize=40)
plt.imshow(cloud8)

subset = df[df['Category']=='Games']
text = subset.cleaned_website_text.values
cloud9=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800
                       ).generate(" ".join(text))
plt.subplot(4,4,9)
plt.axis('off')
plt.title("Games",fontsize=40)
plt.imshow(cloud9)

subset = df[df['Category']=='E-Commerce']
text = subset.cleaned_website_text.values
cloud10=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800
                       ).generate(" ".join(text))
plt.subplot(4,4,10)
plt.axis('off')
plt.title("E-Commerce",fontsize=40)
plt.imshow(cloud10)

subset = df[df['Category']=='Forums']
text = subset.cleaned_website_text.values
cloud11=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800
                       ).generate(" ".join(text))
plt.subplot(4,4,11)
plt.axis('off')
plt.title("Forums",fontsize=40)
plt.imshow(cloud11)

subset = df[df['Category']=='Food']
text = subset.cleaned_website_text.values
cloud12=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800
                       ).generate(" ".join(text))
plt.subplot(4,4,12)
plt.axis('off')
plt.title("Food",fontsize=40)
plt.imshow(cloud12)

subset = df[df['Category']=='Education']
text = subset.cleaned_website_text.values
cloud13=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800
                       ).generate(" ".join(text))
plt.subplot(4,4,13)
plt.axis('off')
plt.title("Education",fontsize=40)
plt.imshow(cloud13)

subset =df[df['Category']=='Computers and Technology']
text = subset.cleaned_website_text.values
cloud14=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800
                       ).generate(" ".join(text))
plt.subplot(4,4,14)
plt.axis('off')
plt.title("Computers and Technology",fontsize=40)
plt.imshow(cloud14)

subset = df[df['Category']=='Business/Corporate']
text = subset.cleaned_website_text.values
cloud15=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800
                       ).generate(" ".join(text))
plt.subplot(4,4,15)
plt.axis('off')
plt.title("Business/Corporate",fontsize=40)
plt.imshow(cloud15)

subset = df[df['Category']=='Adult']
text = subset.cleaned_website_text.values
cloud16=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800
                       ).generate(" ".join(text))
plt.subplot(4,4,16)
plt.axis('off')
plt.title("Adult",fontsize=40)
plt.imshow(cloud16)
# plt.show()

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2),
                        stop_words='english')

# We transform each cleaned_text into a vector
features = tfidf.fit_transform(df.cleaned_website_text).toarray()

labels = df.category_id

# print("Each of the %d text is represented by %d features (TF-IDF score of unigrams and bigrams)" %(features.shape))

# Finding the three most correlated terms with each of the categories
N = 3
for Category, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names_out())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  # print("\n==> %s:" %(Category))
  # print("  * Most Correlated Unigrams are: %s" %(', '.join(unigrams[-N:])))
  # print("  * Most Correlated Bigrams are: %s" %(', '.join(bigrams[-N:])))

"""Spliting the data into train and test sets
The original data was divided into features (X) and target (y), which were then splitted into train (75%) and test (25%) sets. Thus, the algorithms would be trained on one set of data and tested out on a completely different set of data (not seen before by the algorithm).
"""

X = df['cleaned_website_text'] # Collection of text
y = df['Category'] # Target or the labels we want to predict

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state = 0)

# y_train.value_counts()

# y_test.value_counts()

models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    GaussianNB()
]

# 5 Cross-validation
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
# cv_df

mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1,
          ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']
# acc

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.25, random_state=1)

model = LinearSVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

calibrated_svc = CalibratedClassifierCV(estimator=model, cv="prefit")  # Change 'base_estimator' to 'estimator'

calibrated_svc.fit(X_train, y_train)
predicted = calibrated_svc.predict(X_test)
# print(metrics.accuracy_score(y_test, predicted))

# Classification report
# print('\t\t\t\tCLASSIFICATIION METRICS\n')
# print(metrics.classification_report(y_test,predicted,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],target_names= df['Category'].unique()))

conf_mat = confusion_matrix(y_test, predicted,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap="OrRd", fmt='d',
            xticklabels=category_id_df.Category.values,
            yticklabels=category_id_df.Category.values)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.title("CONFUSION MATRIX - LinearSVC\n", size=16)

for predicted in category_id_df.category_id:
    for actual in category_id_df.category_id:
        if predicted != actual and conf_mat[actual, predicted] >0:
            # print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual],id_to_category[predicted],
                                                                #    conf_mat[actual, predicted]))
            display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Category',
                                                                'cleaned_website_text']])

model.fit(features, labels)

N = 4
for Category, category_id in sorted(category_to_id.items()):
  indices = np.argsort(model.coef_[category_id])
  feature_names = np.array(tfidf.get_feature_names_out())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  # print("\n==> '{}':".format(Category))
  # print("  * Top unigrams: %s" %(', '.join(unigrams)))
  # print("  * Top bigrams: %s" %(', '.join(bigrams)))

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, df['category_id'], test_size=0.25, random_state=0)

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words='english')

fitted_vectorizer = tfidf.fit(X_train)
tfidf_vectorizer_vectors = fitted_vectorizer.transform(X_train)

m = LinearSVC().fit(tfidf_vectorizer_vectors, y_train)
m1 = CalibratedClassifierCV(estimator=m, cv="prefit").fit(tfidf_vectorizer_vectors, y_train)  # Change 'base_estimator' to 'estimator

# Now you can use m1 for prediction
y_pred = m1.predict(fitted_vectorizer.transform(X_test))

accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy:", accuracy)


from bs4 import BeautifulSoup
import bs4 as bs4
from urllib.parse import urlparse
import requests
from collections import Counter
import pandas as pd
import os
class ScrapTool:
    def visit_url(self, website_url):
        '''
        Visit URL. Download the Content. Initialize the beautifulsoup object. Call parsing methods. Return Series object.
        '''
        #headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36'}
        content = requests.get(website_url,timeout=60).content

        #lxml is apparently faster than other settings.
        soup = BeautifulSoup(content, "lxml")
        result = {
            "website_url": website_url,
            "website_name": self.get_website_name(website_url),
            "website_text": self.get_html_title_tag(soup)+self.get_html_meta_tags(soup)+self.get_html_heading_tags(soup)+
                                                               self.get_text_content(soup)
        }

        #Convert to Series object and return
        return pd.Series(result)

    def get_website_name(self,website_url):
        '''
        Example: returns "google" from "www.google.com"
        '''
        return "".join(urlparse(website_url).netloc.split(".")[-2])

    def get_html_title_tag(self,soup):
        '''Return the text content of <title> tag from a webpage'''
        return '. '.join(soup.title.contents)

    def get_html_meta_tags(self,soup):
        '''Returns the text content of <meta> tags related to keywords and description from a webpage'''
        tags = soup.find_all(lambda tag: (tag.name=="meta") & (tag.has_attr('name') & (tag.has_attr('content'))))
        content = [str(tag["content"]) for tag in tags if tag["name"] in ['keywords','description']]
        return ' '.join(content)

    def get_html_heading_tags(self,soup):
        '''returns the text content of heading tags. The assumption is that headings might contain relatively important text.'''
        tags = soup.find_all(["h1","h2","h3","h4","h5","h6"])
        content = [" ".join(tag.stripped_strings) for tag in tags]
        return ' '.join(content)

    def get_text_content(self,soup):
        '''returns the text content of the whole page with some exception to tags. See tags_to_ignore.'''
        tags_to_ignore = ['style', 'script', 'head', 'title', 'meta', '[document]',"h1","h2","h3","h4","h5","h6","noscript"]
        tags = soup.find_all(text=True)
        result = []
        for tag in tags:
            stripped_tag = tag.strip()
            if tag.parent.name not in tags_to_ignore\
                and isinstance(tag, bs4.element.Comment)==False\
                and not stripped_tag.isnumeric()\
                and len(stripped_tag)>0:
                result.append(stripped_tag)
        return ' '.join(result)

import spacy as sp
from collections import Counter
sp.prefer_gpu()
import en_core_web_sm
#anconda prompt ko run as adminstrator and copy paste this:python -m spacy download en
nlp = en_core_web_sm.load()
import re
def clean_text(doc):
    '''
    Clean the document. Remove pronouns, stopwords, lemmatize the words and lowercase them
    '''
    doc = nlp(doc)
    tokens = []
    exclusion_list = ["nan"]
    for token in doc:
        if token.is_stop or token.is_punct or token.text.isnumeric() or (token.text.isalnum()==False) or token.text in exclusion_list :
            continue
        token = str(token.lemma_.lower().strip())
        tokens.append(token)
    return " ".join(tokens)

# from flask import Flask, request, jsonify
# import sqlite3

# connection = sqlite3.connect("database.db")
# cursor = connection.cursor()
# cursor.execute('''CREATE TABLE IF NOT EXISTS logs( 
# time timestamp, 
# category INTEGER);''')
# connection.commit()
# del connection
# del cursor


# app = Flask(__name__)

# @app.route('/get_stats', methods=['GET'])
# def get_data():
#     connection = sqlite3.connect("database.db")
#     cursor = connection.cursor()
#     cursor.execute("SELECT * FROM logs")
#     data = connection.fetchall()
#     result = {
#         "progress1": 0,
#         "progress2": 0,
#         "progress3": 0,
#         "progress4": 0,
#         "progress5": 0,
#         "progress6": 0,
#         "progress7": 0,
#         "progress8": 0,
#         "progress9": 0,
#         "progress10": 0,
#         "progress11": 0,
#         "progress12": 0,
#         "progress13": 0,
#         "progress14": 0,
#         "progress15": 0,
#         "progress16": 0
#     }
#     for i in data:
#         if int(i[1]) == 1:
#             result["progress1"] += 1
#         if int(i[2]) == 2:
#             result["progress2"] += 1
#         if int(i[3]) == 3:
#             result["progress3"] += 1
#         if int(i[4]) == 4:
#             result["progress4"] += 1
#         if int(i[5]) == 5:
#             result["progress5"] += 1
#         if int(i[6]) == 6:
#             result["progress6"] += 1
#         if int(i[7]) == 7:
#             result["progress7"] += 1
#         if int(i[8]) == 8:
#             result["progress8"] += 1
#         if int(i[9]) == 9:
#             result["progress9"] += 1
#         if int(i[10]) == 10:
#             result["progress10"] += 1
#         if int(i[11]) == 11:
#             result["progress11"] += 1
#         if int(i[12]) == 12:
#             result["progress12"] += 1
#         if int(i[13]) == 13:
#             result["progress13"] += 1
#         if int(i[14]) == 14:
#             result["progress14"] += 1
#         if int(i[15]) == 15:
#             result["progress15"] += 1
#         if int(i[16]) == 16:
#             result["progress16"] += 1
#     return jsonify(result)


# @app.route('/add_to_logs', methods=['POST'])
# def api():
#     website = str(request.form['url'])
#     # website= input("enter website:")
#     scrapTool = ScrapTool()
#     connection = sqlite3.connect("database.db")
#     cursor = connection.cursor()
#     try:
#         web=dict(scrapTool.visit_url(website))
#         text=(clean_text(web['website_text']))
#         t=fitted_vectorizer.transform([text])
#         # print(id_to_category[m1.predict(t)[0]])
#         category = (id_to_category[m1.predict(t)[0]]).split()[-1]
#         if category == "Travel":
#             query = "INSERT INTO logs values (" + datetime.datetime.now() + ", 1)"
#             cursor.execute(query)
#         elif category == "Social Networking and Messaging":
#             query = "INSERT INTO logs values (" + datetime.datetime.now() + ", 2)"
#             cursor.execute(query)
#         elif category == "News":
#             query = "INSERT INTO logs values (" + datetime.datetime.now() + ", 3)"
#             cursor.execute(query)
#         elif category == "Streaming Services":
#             query = "INSERT INTO logs values (" + datetime.datetime.now() + ", 4)"
#             cursor.execute(query)
#         elif category == "Sports":
#             query = "INSERT INTO logs values (" + datetime.datetime.now() + ", 5)"
#             cursor.execute(query)
#         elif category == "Photography":
#             query = "INSERT INTO logs values (" + datetime.datetime.now() + ", 6)"
#             cursor.execute(query)
#         elif category == "Law and Government":
#             query = "INSERT INTO logs values (" + datetime.datetime.now() + ", 7)"
#             cursor.execute(query)
#         elif category == "Health and Fitness":
#             query = "INSERT INTO logs values (" + datetime.datetime.now() + ", 8)"
#             cursor.execute(query)
#         elif category == "Games":
#             query = "INSERT INTO logs values (" + datetime.datetime.now() + ", 9)"
#             cursor.execute(query)
#         elif category == "E-Commerce":
#             query = "INSERT INTO logs values (" + datetime.datetime.now() + ", 10)"
#             cursor.execute(query)
#         elif category == "Forums":
#             query = "INSERT INTO logs values (" + datetime.datetime.now() + ", 11)"
#             cursor.execute(query)
#         elif category == "Food":
#             query = "INSERT INTO logs values (" + datetime.datetime.now() + ", 12)"
#             cursor.execute(query)
#         elif category == "Education":
#             query = "INSERT INTO logs values (" + datetime.datetime.now() + ", 13)"
#             cursor.execute(query)
#         elif category == "Computers and Technology":
#             query = "INSERT INTO logs values (" + datetime.datetime.now() + ", 14)"
#             cursor.execute(query)
#         elif category == "Business/Corporate":
#             query = "INSERT INTO logs values (" + datetime.datetime.now() + ", 15)"
#             cursor.execute(query)
#         elif category == "Adult":
#             query = "INSERT INTO logs values (" + datetime.datetime.now() + ", 16)"
#             cursor.execute(query)
#         else:
#             print("CATEGORY NOT IDENTIFIED (line 502)")
#             return jsonify({"response": "0"})
#         return jsonify({"response": "1"})
#         # data=pd.DataFrame(m1.predict_proba(t)*100,columns=df['Category'].unique())
#         # data=data.T
#         # data.columns=['Probability']
#         # data.index.name='Category'
#         # a=data.sort_values(['Probability'],ascending=False)
#         # a['Probability']=a['Probability'].apply(lambda x:round(x,2))
#     except:
#         # print("Connection Timedout!")
#         return jsonify({"response": "Connection Timedout!"})

# # if __name__==__main__:
# # app.run(host="0.0.0.0", port=5000, debug=True)
# app.run(port=5000, debug=True)

# a

# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.set(font_scale=1.5)
# plt.figure(figsize=(10, 5))
# i = list(a.index)

# # Use 'i' for the x-axis and 'a['Probability']' for the y-axis
# ax = sns.barplot(x=i, y=a['Probability'])

# plt.title("Probability Prediction for each Category of the URL", fontsize=18)
# plt.ylabel('Probability', fontsize=16)
# plt.xlabel('Category', fontsize=16)

# # Adding the text labels
# rects = ax.patches
# labels = a['Probability']
# for rect, label in zip(rects, labels):
#     height = rect.get_height()
#     ax.text(rect.get_x() + rect.get_width() / 2, height + 2, label, ha='center', va='bottom', fontsize=14)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='center')
# plt.show()