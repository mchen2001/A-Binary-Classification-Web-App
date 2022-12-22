import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.metrics import precision_score, recall_score


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification")
    st.markdown("Are your mushrooms edible or poisonous?")
    st.sidebar.markdown("Are your mushrooms edible or poisonous?")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('./mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list, y_pred, y_score):
        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            confusion_matrix(y_test, y_pred)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader('ROC Matrix')
            # print(y_score)
            roc_curve(y_test, y_score, pos_label=1)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            precision_recall_curve(y_test, y_score, pos_label=1)
            st.pyplot()

    def processModel(metrics, model):
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        y_score = model.predict_proba(x_test)[:, 0]
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics, y_pred, y_score)

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']
    st.sidebar.subheader('Choose Classifier')
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)",
                                                     "Logistic Regression", "Random Forest"))

    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve',
                                                                   'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            processModel(metrics, model)

    elif classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            processModel(metrics, model)

    else:
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10,
                                               key="n_estimators")
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key="bootstrap")

        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         'Confusion Matrix')

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
            processModel(metrics, model)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classifier)")
        st.write(df)


if __name__ == '__main__':
    main()
