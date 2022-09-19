import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pickle
import data_creation_v3 as d
from tensorflow.keras.models import load_model
import streamlit as st
import time
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)
import matplotlib.pyplot as plt
l = os.getcwd()

order = ['bodyLength', 'bscr', 'dse', 'dsr', 'entropy', 'hasHttp', 'hasHttps',
       'has_ip', 'numDigits', 'numImages', 'numLinks', 'numParams',
       'numTitles', 'num_%20', 'num_@', 'sbr', 'scriptLength', 'specialChars',
       'sscr', 'urlIsLive', 'urlLength']

class TimerError(Exception):
     """A custom exception used to report errors in use of Timer class"""
 
class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        return (f"Elapsed time: {elapsed_time:0.4f} seconds")

if __name__ == '__main__':
	st.title("")
	
	st.text("A Neural Network based trained model for extracting features and predicting the type of URL.")
	st.write("_______________")
	st.text(""" 
        """)
	st.sidebar.text("Credits:")
	st.sidebar.text(""" 
        """)
	t = Timer()
	user_input = st.text_input("Enter URL:")
	t.start()
	a = d.UrlFeaturizer(user_input).run()
	test = []
	for i in order:
	    test.append(a[i])
	s1=t.stop()
	encoder = LabelEncoder()
	encoder.classes_ = np.load(l+'/GUI/lblenc_v1.npy',allow_pickle=True)
	scalerfile = l+'/GUI/scaler.sav'
	scaler = pickle.load(open(scalerfile, 'rb'))

	st.sidebar.write("_______________")
	genre = st.sidebar.radio(
	     "Please Select Your Model:",
	     ('TF', 'TF-Lite'))

	if (genre=='TF'):
		model = load_model(l+'/GUI/Model_v2.h5')	
		test = pd.DataFrame(test).replace(True,1).replace(False,0).to_numpy().reshape(1,-1)
		t.start()
		predicted = np.argmax(model.predict(scaler.transform(test)),axis=1)
		s = t.stop()
	else:
		interpreter = tf.lite.Interpreter(model_path=l+"/GUI/tflite_quant_model.tflite")
		interpreter.allocate_tensors()
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()
		test = pd.DataFrame(test).replace(True,1).replace(False,0).to_numpy(dtype="float32").reshape(1,-1)
		interpreter.set_tensor(input_details[0]['index'], scaler.transform(test))
		t.start()
		interpreter.invoke()
		output_data = interpreter.get_tensor(output_details[0]['index'])
		s = t.stop()
		predicted = np.argmax(output_data,axis=1)

	st.sidebar.write("_______________")
	st.sidebar.text("Feature Extraction:")
	st.sidebar.text(str(s1))
	st.sidebar.text("Prediction :")
	st.sidebar.text(str(s))

	ben = [1.        , 1.        , 1.        , 1.        , 0.56158211,
       0.        , 1.        , 0.        , 0.58866722, 1.        ,
       1.        , 0.16708727, 1.        , 0.16454762, 1.        ,
       1.        , 1.        , 0.95        , 1.        , 0.        ,
       0.70961575]

	submit = st.button('Predict')
	if (user_input==""):
		st.write("Enter Valid URL")

	if submit and user_input!="":
		pred = encoder.inverse_transform(predicted)[0]
		st.header("Type of URL : "+pred)
		st.subheader("What is a "+pred+" URL?")
		if (pred!="Benign"):
			image = Image.open(l+'/GUI/danger.jpeg')
			st.sidebar.image(image)
		else:
			image = Image.open(l+'/GUI/safe.png')
			st.sidebar.image(image)


		st.text("""
		.	URL String Characteristics: Features derived from the URL string itself.
		.	URL Domain Characteristics: Domain characteristics of the URLs domain. These include whois information and shodan information.
		.	Page Content Characteristics: Features extracted from the URL’s page (if any)
			""")

		st.header("Extracted Features vs Safe URL")
		st.text("Given below are the features extracted from the URL and the values of these features are plotted along x-axis with the features on the y-axis.")
		plt.figure(figsize=(12,12))
		plt.plot(scaler.transform(test)[0],order,color='red', marker='>',linewidth=0.65,linestyle=":",alpha=0.5)
		plt.plot(ben,order,marker='o',linewidth=0.65,linestyle="--",alpha=0.5)
		plt.legend(["Extracted Features","Avg Safe URL"])
		plt.title("Variation of features for different types of URLs")
		plt.ylabel("Features")
		plt.xlabel("Normalised Mean Values")
		plt.plot()
		st.pyplot()
	
