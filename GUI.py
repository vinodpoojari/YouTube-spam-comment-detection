import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import seaborn as sns
from PIL import Image
import random
import time


class Application:
    def __init__(self):
        st.title('YouTube Spam Detection')

        self.add_selectbox = st.sidebar.selectbox(
            "How would you like to be contacted?",
            ("Train", "Test")
        )
        if self.add_selectbox == 'Test':
            self.test_model()
        

    def load_model(self):
        print('[INFO] Model will loading')
        vect_filename = "Vect_Model.pkl"
        with open(vect_filename, 'rb') as f:
            self.vectorizer = pickle.load(f)

        filename = "Model.pkl"
        with open(filename, 'rb') as f:
            self.Models = pickle.load(f)
         
        print('[INFO] Model will loaded')
        return self.vectorizer,self.Models
    def get_output(self,vectorizer,Models):
        self.input_vectorized = vectorizer.transform([self.user_input])
        self.prediction = Models.predict(self.input_vectorized)
        if int(self.prediction[0])==0:
            self.prediction_is='Normal'
        elif int(self.prediction[0])==1:
            self.prediction_is='Spam'
        return self.prediction_is
            
    def test_model(self):
        self.vectorizer,self.Models = self.load_model()
        st.header('Test YouTube Spam detection')
        self.user_input = st.text_input("Type Here ...")
        
        
        if(st.button('Submit')):
            output = self.user_input.title()
            st.success(f'You : {output}')
            res = self.get_output(self.vectorizer,self.Models)
            st.info(f"Model Say : {res}")


    def train_model(self):

        psy = pd.read_csv('./Youtube01-Psy.csv', usecols= ['CONTENT', 'CLASS'])
        katyperry = pd.read_csv('./Youtube02-KatyPerry.csv', usecols= ['CONTENT', 'CLASS'])
        lmfao = pd.read_csv('./Youtube03-LMFAO.csv', usecols= ['CONTENT', 'CLASS'])
        eminem = pd.read_csv('./Youtube04-Eminem.csv', usecols= ['CONTENT', 'CLASS'])
        shakira = pd.read_csv('./Youtube05-Shakira.csv', usecols= ['CONTENT', 'CLASS'])
        print(psy.shape)
        print(katyperry.shape)
        print(lmfao.shape)
        print(eminem.shape)
        print(shakira.shape)
        self.data = pd.concat([psy, katyperry, lmfao, eminem, shakira], ignore_index=True)
        print(self.data.shape)
        print(self.data.sample(13))
        self.X = np.array(self.data['CONTENT'])
        self.y = np.array(self.data['CLASS'])        
        

        graphs =st.sidebar.radio('Display Graph : ',['Dispaly data','Bar graph', 'Pie Chart'])
        
        if graphs == 'Dispaly data':
            st.header('Loading dataset..')
            progress=st.progress(0)
            for i in range(1,100):
                time.sleep(0.1)
                progress.progress(i+1)
            if i==99:
                st.dataframe(self.data,width=1000, height=800)
                
        elif graphs == 'Bar graph':
            
            fig1, ax1 = plt.subplots()
            colors = np.array(["red",'#4CAF50'])
            label=np.array(["Not Spam", "Spam"])
            data = np.array([sum(self.y == 0),sum(self.y == 1)])
            ax1.bar(label,data)
            st.pyplot(fig1)

        elif graphs == 'Pie Chart':
            categories = ["Not Spam", "Spam"]
            values = np.array([sum(self.y == 0),sum(self.y == 1)])
            explodes = [0.2, 0]
            fig1, ax1 = plt.subplots()
            ax1.pie(values, labels = categories,explode = explodes, shadow = True,radius = 0.4)
            st.pyplot(fig1)

        train_check = st.sidebar.checkbox('Train manually',value=False)
        if train_check ==True:
              
            X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,test_size=0.20,random_state=42)
            vect=TfidfVectorizer() 
            X_train=vect.fit_transform(X_train)
            X_test=vect.transform(X_test)
            print("TfidfVectorizer model will saving")
            pkl_filename = "Vect_Model.pkl"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(vect, file)
            model=SVC()
            model.fit(X_train,y_train)
            pred=model.predict(X_test)
            acc=accuracy_score(y_test,pred)
            st.header('Total Accuracy is : {}'.format(acc*100))
            self.cm= confusion_matrix(y_test, pred)
            self.report=classification_report(y_test,pred)
            print("model will saving")
            pkl_filename = "Model.pkl"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(model, file)
            st.header('The Classification report : \n{}'.format(self.report))
            #st.subheader('Model Confusion matrix :\n{}'.format(self.cm))
            fig1, ax1 = plt.subplots()
            ax1 = sns.heatmap(self.cm/np.sum(self.cm), annot=True, 
                        fmt='.2%', cmap='Blues')

            ax1.set_title('Seaborn Confusion Matrix with labels\n\n');
            ax1.set_xlabel('\nPredicted Values')
            ax1.set_ylabel('Actual Values ')
            plt.savefig("Confusion_matrix.png")
            st.pyplot(fig1)
  

if __name__ == '__main__':
    app = Application()
    
    if app.add_selectbox == 'Train':
        app.train_model()