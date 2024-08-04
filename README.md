# STAFFING_ANALYTICS

# Abstraction:
 This project is based on how a company decides about promotion based on employee’s skills. Machine Learning concepts like Clustering, Decision Tree, K-Nearest Neighbour are used to interpret the employees performance. Each algorithm infers different aspects of the process. I have also attached a portion of an Eligibility test, which shows if an employee is eligible to be promoted based on his skill levels.
 Here a webpage is done using streamlit and connected it to our Python code. In this webpage, you can see the visualizations using different types of plots and can also make prediction by attending the eligibility test. 
 
# Introduction:
 This project is a Python script that uses the Streamlit library to build a web application for staffing analytics. The code imports several libraries, including Pandas, Seaborn, Matplotlib, Numpy, Scikit-learn, and PIL (Python Imaging Library).The code reads in a CSV file named "HRProjectDataset.csv" containing employee data and uses LabelEncoder from Scikit-learn to encode the categorical variables into numerical values. It then selects the relevant columns for feature and target variables and splits the data into training and testing datasets.The code then trains a K-Nearest Neighbors (KNN) classifier and a Decision Tree classifier on the data and computes their accuracy scores. It also performs hierarchical clustering using the Agglomerative Clustering algorithm and visualizes the resulting clusters using Seaborn and Matplotlib.
 The web application built with Streamlit allows users to choose between different machine learning models (KNN, Decision Tree, Clustering) and visualize the results. The application includes interactive widgets such as checkboxes and select boxes that enable users to explore the dataset and models in real-time. Finally, the application uses Streamlit to display bar and pie plots of the data.

# Technologies used:
	•Clustering
	•Decision Tree
	•K-Nearest Neighbour
	•Graphs and Plots 
	•Predictions

# Tools used:
	•Jupyter Notebook
	•Streamlit

# Installations required:
	•In command prompt (Jupyter Notebook/Visual Studio)
	•conda create myenv
	•conda activate myenv 
	•pip install streamlit
	•pip install pandas
	•pip install seaborn
	•pip install matplotlib
	•Run the choice.py file by using (streamlit run Analytics.py) in the command prompt
  
# Output:

![image](https://user-images.githubusercontent.com/117114012/215810202-cd6e9e8c-c133-4d63-a8fe-249fcf1e64c3.png)

![image](https://user-images.githubusercontent.com/117114012/215810005-da6bc6ca-5a3d-4f43-b056-ce9d7c3ff227.png)

![image](https://user-images.githubusercontent.com/117114012/215809095-66119ae4-bbb8-476d-a29c-5712fd23ee07.png)

![image](https://user-images.githubusercontent.com/117114012/215809172-7050c5d4-5e5f-44fe-9832-c09eeba708c1.png)

![image](https://user-images.githubusercontent.com/117114012/215809214-f6520f7f-9191-4ed2-88fa-c8d487120bde.png)


