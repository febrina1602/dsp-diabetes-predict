from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app) #  HTML menembak API ini

model = joblib.load('rf_diabetes_14features.pkl')
scaler = joblib.load('scaler_14features.pkl')
fitur_penting = joblib.load('top_14_features_list.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data_input = request.json 
        if 'Homaβ' in data_input:
            data_input['Homa¦Â'] = data_input.pop('Homaβ')
            
        df_input = pd.DataFrame([data_input])
        df_input = df_input[fitur_penting]
        data_scaled = scaler.transform(df_input)
        
        prediksi = model.predict(data_scaled)[0]
        probabilitas = model.predict_proba(data_scaled).tolist()[0]
        
        labels = ["Normal", "Prediabetes", "Diabetes"]
        
        return jsonify({
            'status': 'success',
            'prediksi_kelas': int(prediksi),
            'prediksi_label': labels[prediksi],
            'probabilitas': probabilitas
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)