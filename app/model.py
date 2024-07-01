import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from difflib import SequenceMatcher

def ip_to_features(ip):
   if not ip or not isinstance(ip, str):
       return [0, 0, 0, 0]  # Default value for missing or invalid IPs
   parts = ip.split('.')
   if len(parts) != 4:
       return [0, 0, 0, 0]  # Default value for improperly formatted IPs
   return [int(part) if part.isdigit() else 0 for part in parts]

def similar(a, b):
   return SequenceMatcher(None, a, b).ratio()

def calculate_string_similarity(record, data, columns):
   similarities = {col: [] for col in columns}
   for column in columns:
       if column in record and record[column] != '':
           column_similarities = data[column].apply(lambda x: similar(record[column], x))
           similarities[column] = column_similarities
       else:
           similarities[column] = [0] * len(data)
   return similarities

def handle_unseen_values(column, encoder, value):
   try:
       return encoder.transform([value])[0]
   except ValueError:
       return -1  # Default value for unseen labels

def extract_features(data, label_encoders=None, vectorizer=None):
   if label_encoders is None:
       label_encoders = {
           'location': LabelEncoder(),
           'operating_system': LabelEncoder(),
           'application_owner': LabelEncoder(),
           'system_owner': LabelEncoder()
       }
       for column in label_encoders:
           data[column] = label_encoders[column].fit_transform(data[column].fillna(''))
   else:
       for column in label_encoders:
           if column in data:
               data[column] = data[column].apply(lambda x: handle_unseen_values(column, label_encoders[column], x))

   if vectorizer is None:
       vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4))
       hostname_ngrams = vectorizer.fit_transform(data['hostname'])
   else:
       hostname_ngrams = vectorizer.transform(data['hostname'])

   hostname_df = pd.DataFrame(hostname_ngrams.toarray(), columns=vectorizer.get_feature_names_out())

   ip_features = data['ip'].apply(ip_to_features).tolist()
   ip_df = pd.DataFrame(ip_features, columns=[f'ip_part_{i}' for i in range(4)])

   X = data.drop(['system_owner', 'application_owner', 'hostname', 'ip'], axis=1, errors='ignore')
   X = pd.concat([X.reset_index(drop=True), hostname_df.reset_index(drop=True), ip_df.reset_index(drop=True)], axis=1)

   return X, label_encoders, vectorizer

def train_models(data):
   X, label_encoders, vectorizer = extract_features(data)
   y_system_owner = data['system_owner']
   y_application_owner = data['application_owner']

   X_train_sys, X_test_sys, y_train_sys, y_test_sys = train_test_split(X, y_system_owner, test_size=0.2, random_state=42)
   X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(X, y_application_owner, test_size=0.2, random_state=42)

   system_owner_model = RandomForestClassifier(n_estimators=100, random_state=42)
   system_owner_model.fit(X_train_sys, y_train_sys)

   application_owner_model = RandomForestClassifier(n_estimators=100, random_state=42)
   application_owner_model.fit(X_train_app, y_train_app)

   return system_owner_model, application_owner_model, label_encoders, vectorizer, list(X.columns)

def predict_owners(new_record, system_owner_model, application_owner_model, label_encoders, vectorizer, feature_columns):
   new_data = pd.DataFrame([new_record])
   X, _, _ = extract_features(new_data, label_encoders, vectorizer)

   X = X.reindex(columns=feature_columns, fill_value=0)

   system_owner_prediction = system_owner_model.predict(X)
   application_owner_prediction = application_owner_model.predict(X)

   predicted_system_owner = label_encoders['system_owner'].inverse_transform(system_owner_prediction)[0]
   predicted_application_owner = label_encoders['application_owner'].inverse_transform(application_owner_prediction)[0]

   return predicted_system_owner, predicted_application_owner