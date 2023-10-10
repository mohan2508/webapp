import numpy as np
import pickle
import streamlit as st
    

load_model = pickle.load(open('diabetes_model.sav', 'rb'))   

def diabetes_prediction(input_data):
    input_data=( 8,183,	64	,0,	0	,23.3,	0.672,	32) #input data to array
    
    input_data_as_numpy_array = np.asarray(input_data)  #reshape the array for prediction
    
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = load_model.predict(input_data_reshaped)
    
    print( prediction)
    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return'The person is diabetic'
    
def main():
    
    st.title('Diabetic Prediction App')
    
    column1,column2,column3 = st.columns(3)
    with column1:
        Pregnancies = st.text_input('Number of pregnancies')
    with column2:
        Glucose = st.text_input('Blood Glucose Level')
    with column3:
        BloodPressure = st.text_input('BloodPressure Value')
    with column1:
        SkinThickness= st.text_input('SkinThickness Value')
    with column2:
        Insulin= st.text_input('Insulin Level')
    with column3:
        BMI = st.text_input('BMI Value')
    with column1:
        DiabetesPedigreeFunction = st.text_input(' DiabetesPedigreeFunction')
    with column2:
        Age = st.text_input('Present Age of the person')
    
    diagnosis = ''
    
    if st.button('Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis) 
    
if __name__ == '__main__':
    main()      
    
        
     