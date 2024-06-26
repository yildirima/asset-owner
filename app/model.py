import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def train_model(data):
   label_encoder = LabelEncoder()
   data['location'] = label_encoder.fit_transform(data['location'])
   data['operating_system'] = label_encoder.fit_transform(data['operating_system'])
   data['system_owner'] = label_encoder.fit_transform(data['system_owner'])

   X = data.drop(['application_owner', 'hostname', 'ip'], axis=1)
   y = data['application_owner']
   y = label_encoder.fit_transform(y)

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)

   return model, label_encoder

def predict_application_owner(new_record, model, label_encoder):
   new_record_transformed = new_record.copy()
   new_record_transformed['location'] = label_encoder.transform([new_record['location']])[0]
   new_record_transformed['operating_system'] = label_encoder.transform([new_record['operating_system']])[0]
   new_record_transformed['system_owner'] = label_encoder.transform([new_record['system_owner']])[0]

   new_record_transformed = pd.DataFrame([new_record_transformed]).drop(['hostname', 'ip'], axis=1)

   prediction = model.predict(new_record_transformed)
   predicted_owner = label_encoder.inverse_transform(prediction)
   return predicted_owner[0]