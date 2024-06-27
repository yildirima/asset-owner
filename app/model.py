import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def train_models(data):
   label_encoders = {
       'location': LabelEncoder(),
       'operating_system': LabelEncoder(),
       'application_owner': LabelEncoder(),
       'system_owner': LabelEncoder()
   }

   for column in label_encoders:
       data[column] = label_encoders[column].fit_transform(data[column])

   X = data.drop(['system_owner', 'application_owner', 'hostname', 'ip'], axis=1)
   y_system_owner = data['system_owner']
   y_application_owner = data['application_owner']

   X_train_sys, X_test_sys, y_train_sys, y_test_sys = train_test_split(X, y_system_owner, test_size=0.2, random_state=42)
   X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(X, y_application_owner, test_size=0.2, random_state=42)

   system_owner_model = RandomForestClassifier(n_estimators=100, random_state=42)
   system_owner_model.fit(X_train_sys, y_train_sys)

   application_owner_model = RandomForestClassifier(n_estimators=100, random_state=42)
   application_owner_model.fit(X_train_app, y_train_app)

   return system_owner_model, application_owner_model, label_encoders

def predict_owners(new_record, system_owner_model, application_owner_model, label_encoders):
   new_record_transformed = new_record.copy()
   for column in label_encoders:
       if column in new_record_transformed:
           new_record_transformed[column] = label_encoders[column].transform([new_record_transformed[column]])[0]

   new_record_transformed = pd.DataFrame([new_record_transformed]).drop(['hostname', 'ip'], axis=1)

   system_owner_prediction = system_owner_model.predict(new_record_transformed)
   application_owner_prediction = application_owner_model.predict(new_record_transformed)

   predicted_system_owner = label_encoders['system_owner'].inverse_transform(system_owner_prediction)[0]
   predicted_application_owner = label_encoders['application_owner'].inverse_transform(application_owner_prediction)[0]

   return predicted_system_owner, predicted_application_owner