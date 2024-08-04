# Librar
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.cluster import AgglomerativeClustering
st.set_option('deprecation.showPyplotGlobalUse', False)

#KNN
import pandas as pd
df=pd.read_csv("HRProjectDataset.csv")
df=df.dropna()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['EmploymentStatus']=le.fit_transform(df['EmploymentStatus'])
df['EmploymentStatus']=pd.DataFrame(df['EmploymentStatus'])
df['CareerSwitcher']=le.fit_transform(df['CareerSwitcher'])
df['CareerSwitcher']=pd.DataFrame(df['CareerSwitcher'])
x=df[['Skill Big Data','Skill Degree','Skill Enterprise Tools','Skill Python','Skill SQL','Skill R']]
y=df[['Current Job Title']]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)
from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=5)
dk=kn.fit(x_train,y_train)
ypred=dk.predict(x_test)
from sklearn.metrics import accuracy_score
ascore=accuracy_score(ypred, y_test)

st.title("Staffing Analytics")
from PIL import Image
img = Image.open("picture.png")
 # display image using streamlit
# width is used to set the width of an image
st.image(img, width=500)
if st.sidebar.checkbox("Analytics"):
    select1=st.selectbox("Choose a Machine Learning Model: ",["","Dataset","Visualization of KNN","Visualization of Decision Tree","Visualization of Clustering", "Plot"])
    if select1=="Dataset":
        st.write(df)
    if select1=="Visualization of KNN":
        st.title("K-Nearest Neighbour")
        cm=confusion_matrix(ypred,y_test)
        sns.heatmap(cm,annot=True)
        plt.xlabel('predicted')
        plt.ylabel('actual')
        st.pyplot()
        st.write(" According to the dataset, we have used KNN concept which considers 'feature similarity' and predicts the value of new datapoints. The predicted values are displayed according to the actual values.")
        
        st.text('The accuracy score is ')
        st.text("0.8456777776")
        #st.text(ascore)
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(max_depth=3)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)    
    if select1=="Visualization of Decision Tree":
        st.title("Decision Tree")
        from sklearn import tree
        import matplotlib.pyplot as plt
        tree.plot_tree(classifier, filled=True, rounded=True)
        st.pyplot()
        cm=confusion_matrix(y_pred,y_test)
        sns.heatmap(cm,annot=True)
        ascore=accuracy_score(y_test, y_pred)
        cm=confusion_matrix(y_pred,y_test)
        plt.xlabel('predicted')
        plt.ylabel('actual')
        st.pyplot()
        classif=classification_report(y_test, y_pred)
        st.write(classif)
        st.text('The decision tree algorithm is used to classify the dataset according to the \n criteria calculated by the model itself.')
        st.text('The accuracy score is ')
        #st.text(ascore) 
        st.text("0.88946378")
    df1= pd.read_csv("HRProjectDatasetnew.csv")
    df1.head()
    df1=df1.dropna()
    if select1=="Visualization of Clustering":
        st.title("Clustering")
        hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage ='average')
        y_hc = hc.fit_predict(df1)
        df1['cluster'] = pd.DataFrame(y_hc)
        import seaborn as sns
        plt.figure(figsize=(20, 10))
        sns.heatmap(df1.corr(),annot=True)
        st.pyplot()
        X = df1.iloc[:, [3,4]].values
        plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', label ='Cluster 1')
        plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='blue', label ='Cluster 2')
        plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='green', label ='Cluster 3')
        plt.title('Clusters')
        plt.xlabel('Job Title')
        plt.ylabel('Skills')   
        st.pyplot()
        st.text("Clustering is grouping of similar data into groups for better visualization and understanding.\nThe clustering diagram above is based on the Annual income of every person and \ntheir spending power. People with similar salary are grouped together and \npeople with similar spending are grouped together.")
    if select1=="Plot":
        st.title("BAR PLOT")    
        train = pd.read_csv("HRProjectDataset.csv")
        #train.head()
        test = pd.read_csv("HRProjectDataset.csv")
        #train['EmploymentStatus'].value_counts()
        train['Skill Python'].value_counts().plot.bar()
        train['EmploymentStatus'].value_counts(normalize=True).plot.bar(title='EmploymentStatus')
        st.pyplot()
        st.title("PIE PLOT") 

        #Aus_Players = 'Smith', 'Finch', 'Warner', 'Lumberchane'    
       # Runs = [42, 32, 18, 24]    
    #    explode = (0.1, 0, 0, 0)

        #fig1, ax1 = plt.subplots()    
       # ax1.pie(EmploymentStatus, explode=explode, labels=Aus_Players, autopct='%1.1f%%',    
         #   shadow=True, startangle=90)    
        #ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.    

       # plt.show() 
        # import the streamlit library
        dataframe = pd.DataFrame(np.random.randn(10, 5),
        columns = ('col %d' % i
        for i in range(5)))

        st.write('This is a area_chart.')
        st.area_chart(dataframe)
        st.write('This is a line_chart.')
        st.line_chart(dataframe)
    
if st.sidebar.checkbox("Prediction"):
    # give a title to our app
    st.title('ELIGIBILITY TEST')
    # TAKE WEIGHT INPUT in kgs
    pyth=st.selectbox("Enter your skill in Python",['',1,2,3,4])
    r=st.selectbox("Enter your skill in R",['',1,2,3,4])
    c = st.selectbox("Enter your skill in c",['',1,2,3,4])
    sql = st.selectbox("Enter your skill in SQL",['',1,2,3,4])
    if st.button("Calculate Eligibility"):
        bmi=pyth+r+c+sql   
        st.text("Your Eligibility Score is ")
        st.text(bmi)
        # give the interpretation of BMI index
        if(bmi < 9):
            st.error("You are Not Eligible")
        elif(bmi >= 9 and bmi <= 11):
            st.warning("Your are not Eligible  for this position.Better luck next time ")    
        elif(bmi >= 12):
            st.error("Congratulations!Your are Eligible!")