import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
raw_df = pd.read_excel('usecase_4_.xlsx')
completed = raw_df[raw_df['Study Status']=="COMPLETED"]
def calculate_months_difference(row):
    start_date = row['Start Date']
    end_date = row['Completion Date']
    if pd.isnull(start_date) or pd.isnull(end_date):
        return None
    delta = relativedelta(end_date, start_date)
    return delta.years * 12 + delta.months
completed['Time Interval'] = completed.apply(lambda row: calculate_months_difference(row), axis=1)
completed.dropna(subset=['Time Interval'],inplace=True)
clean_comp = completed.drop(columns=['Study Title','Study URL','Study Status','Study Results','Interventions','NCT Number','Study Design','Completion Date','Start Date','Primary Completion Date','First Posted','Results First Posted','Last Update Posted','Other IDs'])
clean_comp['Target'] = clean_comp['Study Recruitment Rate'] * clean_comp['Time Interval']
new = clean_comp
new.head()
encoder = LabelEncoder()
enc_col = ['Phases','Sex','Age','Funder Type','Study Type']
for i in enc_col:
    new[i] = encoder.fit_transform(new[i])
new.head()
new = new.fillna(' ')
new['Main'] = new['Brief Summary'] + ' ' + new['Conditions'] + ' ' + new['Primary Outcome Measures'] + ' ' + new['Secondary Outcome Measures'] + ' ' + new['Other Outcome Measures'] + ' ' + new['Sponsor'] + ' ' + new['Collaborators']
new['Main'].head()
new_clean = new.drop(columns=['Brief Summary','Conditions','Primary Outcome Measures','Secondary Outcome Measures','Other Outcome Measures','Sponsor','Collaborators'])
new_clean['Location List'] = new_clean['Locations'].str.split('|')
new_clean['Location Count'] = new_clean['Location List'].apply(len)
new_clean.head()
new_clean['Final Target'] = new_clean['Target'] * new_clean['Location Count']
new_clean['Final Target'] = new_clean['Final Target'].round().astype(int)
df = new_clean.drop(columns=['Locations','Location List','Final Target'])
df.head()
df.shape
df.reset_index(inplace=True)
def remove_non_alpha(text):
    return re.sub(r'[^a-zA-Z]', ' ', text)
df['Main'] = df['Main'].apply(remove_non_alpha)
df['Main'].head()
lematizer = WordNetLemmatizer()
corpus = []
for i in range(0,len(df)):
        review = df['Main'][i]
        review = review.lower()
        review = review.split()
        review = [lematizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
with open('corpus.txt', 'w') as f:
    for item in corpus:
        f.write(f"{item}\n")
loaded_text_array = []
with open('corpus.txt', 'r') as f:
    for line in f:
        loaded_text_array.append(line.strip())
X = df.drop(columns=['Main','Study Recruitment Rate','Target','index'])
X['Text'] = loaded_text_array
Y = df['Study Recruitment Rate']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
nlp_pipeline = Pipeline([('Vectorizer', TfidfVectorizer())])
preprocessor = ColumnTransformer(transformers=[('Text',nlp_pipeline,'Text')])
final_pipeline1 = Pipeline([('preprocessor', preprocessor),('Regressor',XGBRegressor())])
final_pipeline1.fit(x_train,y_train)
y_pred1 = final_pipeline1.predict(x_train)
y_test_pred1 = final_pipeline1.predict(x_test)
mean_absolute_error(y_train,y_pred1)
mean_absolute_error(y_test,y_test_pred1)