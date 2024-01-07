import streamlit as st
import pandas as pd
import pickle

# Load the trained model
loaded_model = pickle.load(open('C:/Users/Sowmya Acharya/Documents/Major_Project/trained_model.sav', 'rb'))

# Function to preprocess the input data
def preprocess_data(data):
    # Ensure the input data has the same structure as the training data
    # Binary encoding for categorical features
    binary_mapping = {'No': 0, 'Yes': 1, 'Normal': 0, 'Abnormal': 1, 'Not Present': 0, 'Present': 1, 'Good':0, 'Poor':1}
    binary_features = ['red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria', 'hypertension',
                       'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'pedal_edema', 'anemia']
    data[binary_features] = data[binary_features].replace(binary_mapping)

    # Return the preprocessed data as a numpy array
    return data.values.reshape(1, -1)

# Function to make predictions
def predict_output(data):
    # Preprocess the input data
    processed_data = preprocess_data(data)

    # Make predictions using the trained model
    predictions = loaded_model.predict(processed_data)
    label_map = {0: 'CKD', 1: 'Non-CKD'}
    predictions = [label_map[prediction] for prediction in predictions]

    # Print the message based on the prediction
    if predictions[0] == 'CKD':
        message = "The predicted value is CKD (Chronic Kidney Disease)."
    else:
        message = "The predicted value is Non-CKD (Non-Chronic Kidney Disease)."

    return predictions, message

# Create the Streamlit web app
def main():
    # Set the title and sidebar options
    st.title('Chronic Kidney Disease Prediction')
    st.sidebar.title('Input Features')

    # Add input fields for each feature
    age = st.sidebar.number_input('Age', value=0, min_value=0, max_value=100, step=1)
    blood_pressure = st.sidebar.number_input('Blood Pressure', value=0.0, min_value=0.0, max_value=200.0, step=1.0)
    specific_gravity = st.sidebar.number_input('Specific Gravity', value=1.0, min_value=1.0, max_value=1.05, step=0.01)
    albumin = st.sidebar.number_input('Albumin', value=0, min_value=0, max_value=5, step=1)
    sugar = st.sidebar.number_input('Sugar', value=0, min_value=0, max_value=5, step=1)
    red_blood_cells = st.sidebar.selectbox('Red Blood Cells', ['Normal', 'Abnormal'])
    pus_cell = st.sidebar.selectbox('Pus Cell', ['Normal', 'Abnormal'])
    pus_cell_clumps = st.sidebar.selectbox('Pus Cell Clumps', ['Present', 'Not Present'])
    bacteria = st.sidebar.selectbox('Bacteria', ['Present', 'Not Present'])
    blood_glucose_random = st.sidebar.number_input('Blood Glucose Random', value=0, min_value=0, max_value=500, step=1)
    blood_urea = st.sidebar.number_input('Blood Urea', value=0, min_value=0, max_value=300, step=1)
    serum_creatinine = st.sidebar.number_input('Serum Creatinine', value=0.0, min_value=0.0, max_value=20.0, step=0.1)
    sodium = st.sidebar.number_input('Sodium', value=0, min_value=0, max_value=200, step=1)
    potassium = st.sidebar.number_input('Potassium', value=0.0, min_value=0.0, max_value=20.0, step=0.1)
    hemoglobin = st.sidebar.number_input('Hemoglobin', value=0.0, min_value=0.0, max_value=25.0, step=0.1)
    packed_cell_volume = st.sidebar.number_input('Packed Cell Volume', value=0, min_value=0, max_value=100, step=1)
    white_blood_cell_count = st.sidebar.number_input('White Blood Cell Count', value=0, min_value=0, max_value=40000, step=1)
    red_blood_cell_count = st.sidebar.number_input('Red Blood Cell Count', value=0.0, min_value=0.0, max_value=10.0, step=0.1)
    hypertension = st.sidebar.selectbox('Hypertension', ['Yes', 'No'])
    diabetes_mellitus = st.sidebar.selectbox('Diabetes Mellitus', ['Yes', 'No'])
    coronary_artery_disease = st.sidebar.selectbox('Coronary Artery Disease', ['Yes', 'No'])
    appetite = st.sidebar.selectbox('Appetite', ['Good', 'Poor'])
    pedal_edema = st.sidebar.selectbox('Pedal Edema', ['Yes', 'No'])
    anemia = st.sidebar.selectbox('Anemia', ['Yes', 'No'])

    # Create a dictionary with the input data
    data = {
        'age': age,
        'blood_pressure': blood_pressure,
        'specific_gravity': specific_gravity,
        'albumin': albumin,
        'sugar': sugar,
        'red_blood_cells': red_blood_cells,
        'pus_cell': pus_cell,
        'pus_cell_clumps': pus_cell_clumps,
        'bacteria': bacteria,
        'blood_glucose_random': blood_glucose_random,
        'blood_urea': blood_urea,
        'serum_creatinine': serum_creatinine,
        'sodium': sodium,
        'potassium': potassium,
        'hemoglobin': hemoglobin,
        'packed_cell_volume': packed_cell_volume,
        'white_blood_cell_count': white_blood_cell_count,
        'red_blood_cell_count': red_blood_cell_count,
        'hypertension': hypertension,
        'diabetes_mellitus': diabetes_mellitus,
        'coronary_artery_disease': coronary_artery_disease,
        'appetite': appetite,
        'pedal_edema': pedal_edema,
        'anemia': anemia
    }

    # Convert the dictionary to a pandas DataFrame
    input_data = pd.DataFrame(data, index=[0])

    # Display the input data
    st.write('Input Data:')
    st.dataframe(input_data)

    # Make predictions when the 'Predict' button is clicked
    if st.button('Predict'):
        # Call the predict_output function to get the predictions
        predictions, message = predict_output(input_data)

        # Display the predictions
        st.write('Predicted Output:')
        st.write(predictions[0])

        # Display the message
        st.write('Prediction Message:')
        st.write(message)

# Run the web app
if __name__ == '__main__':
    main()
