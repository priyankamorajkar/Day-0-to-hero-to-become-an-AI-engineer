import pandas as pd
import numpy as np
import os
import glob
import joblib
import kagglehub
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__, template_folder='.', static_folder='.')

def setup_model():
    path = kagglehub.dataset_download("dravidvaishnav/mumbai-house-prices")
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    df = pd.read_csv(csv_files[0])

    def convert_to_numeric(row):
        val = row['price']
        unit = str(row['price_unit']).strip()
        if unit == 'Cr': return val * 10000000
        elif unit == 'L': return val * 100000
        return val

    df['Total_Price'] = df.apply(convert_to_numeric, axis=1)
    

    df = df[['Total_Price', 'area', 'locality', 'bhk']].dropna()


    localities = sorted(df['locality'].unique())
    loc_mapping = {loc: i for i, loc in enumerate(localities)}
    df['Loc_Encoded'] = df['locality'].map(loc_mapping)

    X = df[['area', 'bhk', 'Loc_Encoded']]
    y = df['Total_Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Accuracy (R2 Score): {r2:.2f}")
    print(f"Mean Absolute Error (Avg deviation): ₹{mae:,.2f}")

    joblib.dump(model, 'mumbai_model.pkl')
    joblib.dump(loc_mapping, 'loc_mapping.pkl')
    print("--- Task 6: API Logic Initialized ---")
    return localities

try:
    LOCALITIES = setup_model()
except Exception as e:
    print(f"Setup Error: {e}")
    LOCALITIES = []

@app.route('/')
def home():
    return render_template('index.html', localities=LOCALITIES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        area_val = request.form.get('area')
        bhk_val = request.form.get('bhk')
        loc_name = request.form.get('locality')
        
        model = joblib.load('mumbai_model.pkl')
        mapping = joblib.load('loc_mapping.pkl')
        
        loc_encoded = mapping[loc_name]
        prediction = model.predict([[float(area_val), int(bhk_val), loc_encoded]])[0]
 
        if prediction >= 10000000:
            formatted = f"₹{prediction/10000000:.2f} Cr"
        else:
            formatted = f"₹{prediction/100000:.2f} Lakh"

        return render_template('index.html', 
                               prediction_text=f'Predicted Price: {formatted}', 
                               localities=LOCALITIES,
                               area=area_val,
                               bhk=bhk_val,
                               selected_loc=loc_name)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", localities=LOCALITIES)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)