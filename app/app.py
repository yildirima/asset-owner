from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
from model import train_models, predict_owners
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ensure sample_data.csv exists before proceeding
if not os.path.exists('sample_data.csv'):
   raise FileNotFoundError("sample_data.csv not found in /app directory")

# Load and train the models initially
data = pd.read_csv('sample_data.csv')
system_owner_model, application_owner_model, label_encoders = train_models(data)

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
   if 'file' not in request.files:
       return redirect(request.url)
   file = request.files['file']
   if file.filename == '':
       return redirect(request.url)
   if file:
       filename = secure_filename(file.filename)
       file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
       file.save(file_path)
       global data, system_owner_model, application_owner_model, label_encoders
       data = pd.read_csv(file_path)
       system_owner_model, application_owner_model, label_encoders = train_models(data)
       return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
   if request.method == 'POST':
       new_record = {
           'hostname': request.form['hostname'],
           'ip': request.form['ip'],
           'location': request.form['location'],
           'operating_system': request.form['operating_system']
       }
       predicted_system_owner, predicted_application_owner = predict_owners(new_record, system_owner_model, application_owner_model, label_encoders)
       return render_template('predict.html', system_owner=predicted_system_owner, application_owner=predicted_application_owner)
   return render_template('predict.html')

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)