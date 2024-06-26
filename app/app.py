from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
from model import train_model, predict_application_owner
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load and train the model initially
data = pd.read_csv('sample_data.csv')
model, label_encoder = train_model(data)

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
       global data, model, label_encoder
       data = pd.read_csv(file_path)
       model, label_encoder = train_model(data)
       return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
   if request.method == 'POST':
       new_record = {
           'hostname': request.form['hostname'],
           'ip': request.form['ip'],
           'location': request.form['location'],
           'operating_system': request.form['operating_system'],
           'system_owner': request.form['system_owner']
       }
       predicted_owner = predict_application_owner(new_record, model, label_encoder)
       return render_template('predict.html', prediction=predicted_owner)
   return render_template('predict.html')

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)