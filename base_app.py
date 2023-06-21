"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
from PIL import Image

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

#load training data


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	image = Image.open('logo.jpg')

	col1, col2 = st.columns([1, 3])
	with col1:
		st.image(image, use_column_width=True)
	with col2:
		st.title("Twitter Sentiment Classifier App")
	#add more text
	st.write('Predict the sentiment of each twitter using various models with each tweet falling into one of 4 categories: anti-man made climate change, neutral, pro-man made climate change and lastly, whether a tweet represent factual news!')
	st.write('To access the codebase for this application, please visit the following GitHub repository: insert link to our git repository')

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Model Explainations","Explore the data"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Model Explaination" page
	if selection == "Model Explainations":
		options = ['Logistic Regression','Support Vector Classifier','Random Forest Classifier','XGBoost Classifier','CatBoost Classifer', 'Neural Networks Classifier','Naives Bayes Classifier','KNN Classifier']
		selection = st.selectbox('Which model would you like to learn more about?',options)

		if selection == "Logistic Regression":
			st.info('Explain the inner workings of Logistic Regression model')
			#st.markdown('Explain the inner workings of this model')
		if selection == "Support Vector Classifier":
			st.info('Explain the inner workings of Support Vector Machines model')
			#st.markdown('Explain the inner workings of this model')
		if selection == "Random Forest Classifier":
			st.info('Explain the inner workings of Random Forest model')
			#st.markdown('Explain the inner workings of this model')
		if selection == "XGBoost Classifier":
			st.info('Explain the inner workings of XGBoost model')
			#st.markdown('Explain the inner workings of this model')
		if selection == "CatBoost Classifier":
			st.info('Explain the inner workings of CatBoost model')
			#st.markdown('Explain the inner workings of this model')
		if selection == "Neural Networks Classifier":
			st.info('Explain the inner workings of the Neural Networks model')
			#st.markdown('Explain the inner workings of this model')
		if selection == "Naives Bayes Classifier":
			st.info('Explain the inner workings of Naives Bayes model')
			#st.markdown('Explain the inner workings of this model')
		if selection == "KNN Classifier":
			st.info('Explain the inner workings of KNN model')
			#st.markdown('Explain the inner workings of this model')

		
	if selection == "Explore the Data":
		option = ['What would like to explore']


	# Building out the predication page
	if selection == 'Prediction':
		pred_type = st.sidebar.selectbox("Predict sentiment of a single tweet or submit a csv for multiple tweets", ('Single Tweet', 'Multiple Tweets'))

		if pred_type == "Single Tweet":
			#st.info("Prediction with ML Models")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter tweet here","Type Here")


			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

		if pred_type == "Multiple Tweets":
			upload = st.file_uploader('Upload a CSV file here', type='csv', accept_multiple_files=False, key=None, help='Only CSV files are accepted', on_change=None, args=None, kwargs=None)
		
			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
