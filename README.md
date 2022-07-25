# Classification of Heart Disease Patients

# Summary 
A project implementing simple neural network on Keras to classify patients of heart disease from healthy persons.

# Dataset
This project is based on a Kaggle dataset by David Lapp. The link is as follows: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset.  This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.  The attributes are age, sex, chest pain type (4 values), resting blood pressure, serum cholestoral in mg/dl, fasting blood sugar > 120 mg/dl, resting electrocardiographic results (values 0,1,2), maximum heart rate achieved, exercise induced angina, oldpeak = ST depression induced by exercise relative to rest, the slope of the peak exercise ST segment, number of major vessels (0-3) colored by flourosopy, thal: 0 = normal; 1 = fixed defect; 2 = reversable defect.  Lastly, the names and social security numbers of the patients were recently removed from the database, replaced with dummy values.

# Methodology
For preprocessing, we scale the data with Standard Scaler. We then implement a NN with 4 hidden layers with relu activation and l1 regularisation. We also include Dropout layers to avoid overfitting

# Result
With 50 epochs and a batch size of 32, we reach a training accuracy of 95.12% and validation accuracy of 96.59%
