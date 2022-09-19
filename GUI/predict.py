import os
from textwrap import wrap
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pickle
import data_creation_v3 as d
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import streamlit as st
import time
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)
import matplotlib.pyplot as plt
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

order = ['bodyLength', 'bscr', 'dse', 'dsr', 'entropy', 'hasHttp', 'hasHttps',
       'has_ip', 'numDigits', 'numImages', 'numLinks', 'numParams',
       'numTitles', 'num_%20', 'num_@', 'sbr', 'scriptLength', 'specialChars',
       'sscr', 'urlIsLive', 'urlLength']



def convert_to_ascii(sentence):
    sentence_ascii=[]
    for i in sentence:
        if(ord(i)<8222):      # ” has ASCII of 8221
            
            if(ord(i)==8217): # ’  :  8217
                sentence_ascii.append(134)
            if(ord(i)==8221): # ”  :  8221
                sentence_ascii.append(129)
            if(ord(i)==8220): # “  :  8220
                sentence_ascii.append(130)
            if(ord(i)==8216): # ‘  :  8216
                sentence_ascii.append(131)
            if(ord(i)==8217): # ’  :  8217
                sentence_ascii.append(132)
            if(ord(i)==8211): # –  :  8211
                sentence_ascii.append(133)
            if (ord(i)<=128):
                    sentence_ascii.append(ord(i))
            else:
                    pass
    zer=np.zeros((10000))
    for i in range(len(sentence_ascii)):
        zer[i]=sentence_ascii[i]

    zer.shape=(100, 100)
    return zer

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
	st.title("BrowseEnsure: harmful URL and XSS detection system")
	
	st.text("")
	st.write("_______________")
	st.text(""" 
        """)
	st.sidebar.text("By:")
	st.sidebar.text(""" Yugandhar Alla
        """)

	user_input = st.text_input("Enter URL:")
	a = d.UrlFeaturizer(user_input).run()
	test = []
	for i in order:
	    test.append(a[i])

	encoder = LabelEncoder()
	encoder.classes_ = np.load('lblenc_v1.npy',allow_pickle=True)
	scalerfile = 'scaler.sav'
	scaler = pickle.load(open(scalerfile, 'rb'))
	model = load_model("Model_v2.h5")#, custom_objects={'f1_m':f1_m,"precision_m":precision_m, "recall_m":recall_m})
	test = pd.DataFrame(test).replace(True,1).replace(False,0).to_numpy().reshape(1,-1)
	t = Timer()
	t.start()
	predicted = np.argmax(model.predict(scaler.transform(test)),axis=1)
	s = t.stop()
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
			image = Image.open('danger.jpeg')
			st.sidebar.image(image)
		else:
			image = Image.open('safe.png')
			st.sidebar.image(image)

		if (pred=="Benign"):
			st.text("These URLs are generally harmless and non-malicious.")
		elif(pred=="Spam"):
			st.write("Spam refers to a broad range of unwanted pop-ups, links, data and emails that we face in our daily interactions on the web. Spam’s namesake is, (now unpopular) luncheon meat that was often unwanted but ever present. Spam can be simply unwanted, but it can also be harmful, misleading and problematic for your website in a number of ways.")
			st.write("Read More: [https://www.goup.co.uk/guides/spam/](https://www.goup.co.uk/guides/spam/)")
		elif(pred=="Defacement"):
			st.write("Web defacement is an attack in which malicious parties penetrate a website and replace content on the site with their own messages. The messages can convey a political or religious message, profanity or other inappropriate content that would embarrass website owners, or a notice that the website has been hacked by a specific hacker group.")
			st.write("Read More: [https://www.imperva.com/learn/application-security/website-defacement-attack/](https://www.imperva.com/learn/application-security/website-defacement-attack/)")	
		elif(pred=="Malware"):
			st.write("The majority of website malware contains features which allow attackers to evade detection or gain and maintain unauthorized access to a compromised environment. Some common types of website malware include credit card stealers, injected spam content, malicious redirects, or even website defacements.")
			st.write("Read More: [https://sucuri.net/guides/website-malware/](https://sucuri.net/guides/website-malware/)")	
		else:
			st.write("A phishing website (sometimes called a 'spoofed' site) tries to steal your account password or other confidential information by tricking you into believing you're on a legitimate website. You could even land on a phishing site by mistyping a URL (web address).")
			st.write("Read More: [https://safety.yahoo.com/Security/PHISHING-SITE.html#:~:text=A%20phishing%20website%20(sometimes%20called,a%20URL%20(web%20address).](https://safety.yahoo.com/Security/PHISHING-SITE.html#:~:text=A%20phishing%20website%20(sometimes%20called,a%20URL%20(web%20address).)")	


		st.write("")
		st.header("Extracted Features vs Safe URL")
		st.subheader("Given below are the features extracted from the URL and the values of these features are plotted along x-axis with the features on the y-axis.")
		plt.figure(figsize=(12,12))
		plt.plot(scaler.transform(test)[0],order,color='red', marker='>',linewidth=0.65,linestyle=":",alpha=0.5)
		plt.plot(ben,order,marker='o',linewidth=0.65,linestyle="--",alpha=0.5)
		plt.legend(["Extracted Features","Avg Safe URL"])
		plt.title("Variation of features for different types of URLs")
		plt.ylabel("Features")
		plt.xlabel("Normalised Mean Values")
		plt.plot()
		st.pyplot()

	a = st.text_input("Enter XSS::")
	model = load_model("yes.h5")
	if (a == ""):
		st.write("Enter Valid URL:")
	s1 = st.button('Predict XSS:')

	if s1 and a != "":
		a = np.reshape(a, (-1, 1))
		a = pd.DataFrame(a)
		a.columns = ["a"]
		a = a["a"]
		arr = np.zeros((len(a),100,100))

		for i in range(len(a)):
			image=convert_to_ascii(a[i])
			x=np.asarray(image,dtype='float')
			import cv2
			image =  cv2.resize(x, dsize=(100,100), interpolation=cv2.INTER_CUBIC)
			image/=128
			arr[i]=image
		data = arr.reshape(arr.shape[0], 100, 100, 1)
		A = model.predict(data)
		A = np.around(A, decimals=0)
		if A == 1:
			st.write("Safe")
		if A == 0:
			st.write("Unsafe")
