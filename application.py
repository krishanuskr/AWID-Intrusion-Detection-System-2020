import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

def main():
    st.title("AWID Anomaly Detection")
    st.sidebar.title("Classification")

    #st.set_option('deprecation.showPyplotGlobalUse', False)

    @st.cache(persist=True)
    def load_data():
        awid = pd.read_csv("AWID.csv")
        df=awid
        df = df.drop(df[df['class'].str.contains("normal")].sample(frac=.4).index)

        return df
            

    @st.cache(persist=True)
    def split(df):
        X, y = df.select_dtypes(['number']), df['class']
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(X)

        x_train, x_test, y_train, y_test = train_test_split(
            X_new, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test)
            st.pyplot()



    df = load_data()

    x_train, x_test, y_train, y_test = split(df)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "Classifier", ("Grid Search Pipeline CV", "Logistic Regression", "Random Forest","Support Vector Machine(SVM)"))

    if classifier == 'Grid Search Pipeline CV':
        st.sidebar.subheader("Model Hyperparameters")
        metrics = st.sidebar.multiselect(
            "What metrics to plot?", ('Confusion Matrix',''))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("GridSearch-Pipeline")
            
            preprocessing = Pipeline([("scale", StandardScaler()),])
            params = {
                "classifier__max_depth": [None, 3, 5, 10],
                }
            pipeline = Pipeline([("preprocessing", preprocessing),("classifier", DecisionTreeClassifier())])
            model = GridSearchCV(pipeline, params)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(
                y_test, y_pred, average='micro').round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, average='micro').round(2))
            plot_metrics(metrics)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input(
            "C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider(
            "Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect(
            "What metrics to plot?", ('Confusion Matrix',''))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(
                y_test, y_pred, average='micro').round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, average='micro').round(2))
            plot_metrics(metrics)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input(
            "The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input(
            "The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
        bootstrap = st.sidebar.radio(
            "Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect(
            "What metrics to plot?", ('Confusion Matrix',''))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(
                y_test, y_pred, average='micro').round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, average='micro').round(2))
            plot_metrics(metrics)

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        # choose parameters
        C = st.sidebar.number_input(
            "C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio(
            "Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect(
            "What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(
                y_test, y_pred, average='micro').round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, average='micro').round(2))
            plot_metrics(metrics)




    if st.sidebar.checkbox("Show Raw Data", False):
        st.subheader("Attack Data ML Analysis of Performance of Various Models")
        st.write(df.head())
        st.markdown("This data is provided by AWID.")


if __name__ == '__main__':
    main()
