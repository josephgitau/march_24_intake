# importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns


## ml libaries
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# Create a title for our app
st.title('Prediction and Evaluation of MPG dataset')
st.header("This is Joseph's web app")


# Load the dataset
df = sns.load_dataset('mpg')


# create app header
st.header("sample mpg dataset")
st.dataframe(df.head())


# create an input where user can show more data sample
st.sidebar.subheader('Show more data')
if st.sidebar.checkbox('Show more data'):
    # specify the number of rows to show
    st.subheader('Showing more data')
    num_rows = st.sidebar.number_input('Number of rows to show', 1, 100)
    st.table(df.head(num_rows))


## create sample visual with two columns


col1, col2 = st.columns(2)


with col1:
    # visual number one (Avegare mpg per origin)
    st.subheader('Average mpg per origin')
    avg_mpg = df.groupby('origin')['mpg'].mean()
    st.bar_chart(avg_mpg)


with col2:
    # visual number two (scatter plot of mpg vs horsepower)
    st.subheader('Scatter plot of mpg vs horsepower')
    st.scatter_chart(data=df, x='mpg', y='horsepower')


## machine learning
# create a subheader
st.subheader('Machine Learning Model')
st.write('In this section, we will train a machine learning model to predict the mpg of a car based on its features')


## preprocess the data
# print and drop any missing values
st.write('Checking for missing values')
st.write(df.isnull().sum())
df.dropna(inplace=True)


# create a feature and target variable
X = df.drop(['mpg', 'name'], axis=1)
y = df['mpg']


# encode the categorical variables
# instantiate the encoder
encoder = OrdinalEncoder()


# encode the categorical variables
X['origin'] = encoder.fit_transform(X[['origin']])


## display the processed data
st.write('Processed data')
st.dataframe(X.head())


## create our training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## fit the model
rf = RandomForestRegressor()


# fit model to the training data
st.subheader('Training the model')
with st.spinner('Training the model...'):
    rf.fit(X_train, y_train)


st.success('Model trained successfully')


## evaluate the model
st.subheader('Model Evaluation')
st.write('In this section, we will evaluate the model on the test data')


## make predictions
y_pred = rf.predict(X_test)


# calculate the mean squared error
mse = np.mean((y_test - y_pred)**2)
st.write(f'Mean Squared Error: {mse}')


# calculate the root mean squared error
rmse = np.sqrt(mse)
st.write(f'Root Mean Squared Error: {rmse}')


# r2 score of the model
r2 = rf.score(X_test, y_test)
st.write(f'R2 Score: {r2}')


## Create a siderbar for user inputs
st.header('Make a prediction')
st.sidebar.subheader('option 1: file upload')
st.sidebar.write('Upload a csv file with your car features')


##
uploaded_file = st.sidebar.file_uploader('Upload your file', type=['csv'])


if uploaded_file is not None:
    # read the file
    input_data = pd.read_csv(uploaded_file)
    input_data = input_data.drop(['mpg', 'name'], axis=1)
    # confirm all columns match
    if set(input_data.columns) == set(X.columns):
        st.write('Columns match')
    else:
        st.write('Columns do not match')
    # encode the origin
    input_data['origin'] = encoder.transform(input_data[['origin']])
    # make prediction
    prediction = rf.predict(input_data)
    input_data['predicted_mpg'] = prediction


    # display the prediction
    st.write('Bulk Predictions table')
    st.dataframe(input_data)


st.sidebar.subheader('option 2: manual input')
st.sidebar.write('Enter the car features manually')


cylinders = st.sidebar.number_input('cylinders')
displacement = st.sidebar.number_input('displacement')
horsepower = st.sidebar.number_input('horsepower')
weight = st.sidebar.number_input('weight')
acceleration = st.sidebar.number_input('acceleration')
model_year = st.sidebar.number_input('model_year')


options = df['origin'].unique()
origin = st.sidebar.selectbox('origin', options)


user_input_dictionary = {'cylinders': cylinders, 'displacement': displacement, 'horsepower': horsepower,
                         'weight': weight, 'acceleration': acceleration, 'model_year': model_year, 'origin': origin}


# make prediction
user_input = pd.DataFrame(user_input_dictionary, index=[0])
st.write(user_input)
st.dataframe(user_input)
user_input['origin'] = encoder.transform(user_input[['origin']])
prediction = rf.predict(user_input)
user_input['predicted_mpg'] = prediction


st.write(f'The predicted mpg for the car is: {prediction[0]}')
st.dataframe(user_input)
st.write('End of the app')
