import mlflow
import dagshub
mlflow.set_tracking_uri("https://dagshub.com/Naman12769/you-tube-comment-analysis.mlflow")
dagshub.init(repo_owner='Naman12769', repo_name='you-tube-comment-analysis', mlflow=True)
mlflow.set_experiment("Handling Imbalanced Data")

from imblearn.over_sampling import SMOTE,ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import mlflow.sklearn
from matplotlib import pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import os

df=pd.read_csv('reddit_preprocessing.csv').dropna(subset=['clean_comment'])
df.shape

def run_imbalanced_experiment(imbalance_method):
  ngram_range=(1,3)
  max_features=1000
  vectorizer=TfidfVectorizer(ngram_range=ngram_range,max_features=max_features)
  X=vectorizer.fit_transform(df['clean_comment'])
  y=df['category']
  # class_weight=""

  if imbalance_method=='class_weights':
    class_weight="balanced"
  else:
    class_weight=None

    if imbalance_method=="oversampling":
      smote=SMOTE(random_state=42)
      X,y=smote.fit_resample(X,y)
    elif imbalance_method=="adasyn":
      adasyn=ADASYN(random_state=42)
      X,y=adasyn.fit_resample(X,y)
    elif imbalance_method=="undersampling":
      rus=RandomUnderSampler(random_state=42)
      X,y=rus.fit_resample(X,y)
    elif imbalance_method=="smote_enn":
      smote_enn=SMOTEENN(random_state=42)
      X,y=smote_enn.fit_resample(X,y)
  
  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

  with mlflow.start_run() as run:
    mlflow.set_tag("mlflow.runName",f"Imbalance_{imbalance_method}_Randomforest_TFIDF_Trigrams")
    mlflow.set_tag("experiment_type","imbalanced handling")
    mlflow.set_tag("model_type","RandomForestClassifier")
    mlflow.set_tag("description",f"Randomforest with Tf-IDF trigram handling imbalanced dataset method={imbalance_method}")

    mlflow.log_param("vectrizer type","TF_IDF")   
    mlflow.log_param("ngram_range",ngram_range)
    mlflow.log_param("vectorizer_max_features",max_features)

    n_estimators=200
    max_depth=15
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("imbalance_method",imbalance_method)
        # Initialize and train the model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42,class_weight=class_weight)
    model.fit(X_train, y_train)

        # Step 5: Make predictions and log metrics
    y_pred = model.predict(X_test)

        # Log accuracy
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

        # Log classification report
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    for label, metrics in classification_rep.items():
        if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

        # Log confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix: TF-IDF Trigrams, max_features={max_features}")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # Log the model
        mlflow.sklearn.log_model(model, f"random_forest_model_tfidf_trigrams__imbalanced{imbalance_method}")

imbalanced_methods=['class_weights','oversampling','adasyn','undersampling','smote_enn']

for method in imbalanced_methods:
   run_imbalanced_experiment(method)
    



      
