import numpy as np 
import pickle


load_model = pickle.load(open('diabetes_model.sav', 'rb'))

input_data=( 8,183,	64	,0,	0	,23.3,	0.672,	32)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = load_model.predict(input_data_reshaped)
print(prediction)
if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
