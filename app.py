from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble._loss import SomeLossFunction  # Adjust as needed

# Load saved model
model = pickle.load(open('models/water (1).pkl','rb'))

# def predict_water_potability(model):
#     # Collect user input
#     user_input = {
#         'ph': float(input('Enter PH level: ')),
#         'Hardness': float(input('Enter Hardness: ')),
#         'Solids': float(input('Enter Solids: ')),
#         'Chloramines': float(input('Enter Chloramines: ')),
#         'Sulfate': float(input('Enter Sulfate: ')),
#         'Conductivity': float(input('Enter Conductivity: ')),
#         'Organic_carbon': float(input('Enter Organic Carbon: ')),
#         'Trihalomethanes': float(input('Enter Trihalomethanes: ')),
#         'Turbidity': float(input('Enter Turbidity: '))
#     }
#     # Convert the user input into a DataFrame
#     user_df = pd.DataFrame([user_input])
# #     # Make a prediction using the trained model
#     user_prediction = model.predict(user_df)
# #     # Display the result
#     if user_prediction[0] == 0:
#         print('The Water is not Potable and Safe to drink')
#     else:
#         print('The Water is Potable and Safe to drink')
#     # except ValueError as e:
#     #     print(f"Invalid input: {e}")
#     # except Exception as e:
#     #     print(f"An error occurred: {e}")
#     return user_df         
# Initialize the Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the URL Classification API"})

@app.route("/predict", methods=["POST"])
def pred():
    """
    Endpoint for predicting water potability based on chemical parameters.
    """
    data = request.json
    required_fields = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

    # Check for missing fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

    try:
        # Convert input to DataFrame
        user_df = pd.DataFrame([data])

        # Make prediction
        user_prediction = model.predict(user_df)

        # Prepare response
        # prediction = "Potable" 
        if user_prediction[0] == 0 :
            prediction = "Potable" 
        else :
            prediction = "Not Potable"
        return jsonify({"prediction": prediction, "input_data": data})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Run the app
# If running locally, use: python app.py
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=5000)
