import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import mlflow.sklearn
import seaborn as sns
import pandas as pd
import os


df=pd.read_csv('reddit_preprocessing.csv').dropna(subset=['clean_comment'])

mlflow.set_tracking_uri("https://dagshub.com/Naman12769/you-tube-comment-analysis.mlflow")
dagshub.init(repo_owner='Naman12769', repo_name='you-tube-comment-analysis', mlflow=True)
mlflow.set_experiment("Exp2 BoW vs TFIDF")

# def run_experiment(vectorizer_type,ngram_range,vectorizer_max_features,vectorizer_name):
#   if(vectorizer_type=="BoW"):
#     vectorizer=CountVectorizer(ngram_range=ngram_range,max_features=vectorizer_max_features)
#   else:
#     vectorizer=TfidfVectorizer(ngram_range=ngram_range,max_features=vectorizer_max_features)
#   X=vectorizer.fit_transform(df['clean_comment'])
#   y=df['category']

#   X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

#   with mlflow.start_run() as run:
#     mlflow.set_tag("mlflow.runName",f"{vectorizer_name}_{ngram_range}_RandomForest")
#     mlflow.set_tag("experiment_type","feature engineering")
#     mlflow.set_tag("model_type","RandomForestClassifier")
#     mlflow.set_tag("description",f"RandomForest with {vectorizer_name}, ngram_range {ngram_range}")
#     mlflow.log_param("vectorizer_type",vectorizer_type)
#     mlflow.log_param("ngram_range",ngram_range)
#     mlflow.log_param("vectorizer max features",vectorizer_max_features)
    
#     n_estimators=200
#     max_Depth=15

#     model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_Depth,random_state=42)
#     model.fit(X_train,y_train)

#     y_pred=model.predict(X_test)
#     accuracy_score=accuracy_score(y_pred,y_test)
#     mlflow.log_metric("accuracy",accuracy_score)

#     classification_rep=classification_report(y_test,y_pred,output_dict=True)

#     for label,metrics in classification_rep:
#       if(isinstance(metrics,dict)):
#         for metric,value in metrics.items():
#           mlflow.log_metric(f"{label}_{metric}",value)
#     conf_matrix=confusion_matrix(y_test,y_pred)
#     plt.figure(figsize=(8,6))
#     sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title(f"confusion matrix: {vectorizer_name},{ngram_range}")
#     plt.savefig("confusion_matrix.png")
#     mlflow.log_artifact("confusion_matrix.png")
#     plt.close()

#     mlflow.sklearn.log_model(model,f"random _forest_model_{vectorizer_name}_{ngram_range}")

#   ngram_ranges=[(1,1),(1,2),(1,3)]
#   max_features=5000

#   for ngram_range in ngram_ranges:
#     run_experiment("BoW",ngram_range,max_features,vectorizer_name="BoW")
#     run_experiment("TF_IDF",ngram_range,max_features,vectorizer_name="TF_IDF")


# Step 1: Function to run the experiment
def run_experiment(vectorizer_type, ngram_range, vectorizer_max_features, vectorizer_name):
    # Step 2: Vectorization
    if vectorizer_type == "BoW":
        vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=vectorizer_max_features)
    else:
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=vectorizer_max_features)

    X_train, X_test, y_train, y_test = train_test_split(df['clean_comment'], df['category'], test_size=0.2, random_state=42, stratify=df['category'])

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Step 4: Define and train a Random Forest model
    with mlflow.start_run() as run:
        # Set tags for the experiment and run
        mlflow.set_tag("mlflow.runName", f"{vectorizer_name}_{ngram_range}_RandomForest")
        mlflow.set_tag("experiment_type", "feature_engineering")
        mlflow.set_tag("model_type", "RandomForestClassifier")

        # Add a description
        mlflow.set_tag("description", f"RandomForest with {vectorizer_name}, ngram_range={ngram_range}, max_features={vectorizer_max_features}")

        # Log vectorizer parameters
        mlflow.log_param("vectorizer_type", vectorizer_type)
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("vectorizer_max_features", vectorizer_max_features)

        # Log Random Forest parameters
        n_estimators = 200
        max_depth = 15

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Initialize and train the model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
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
                print(metric,value)
            print(label)
            print(metrics)

        # Log confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix: {vectorizer_name}, {ngram_range}")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # Log the model
        mlflow.sklearn.log_model(model, f"random_forest_model_{vectorizer_name}_{ngram_range}")

# Step 6: Run experiments for BoW and TF-IDF with different n-grams
ngram_ranges = [(1, 1), (1, 2), (1, 3)]  # unigrams, bigrams, trigrams
max_features = 5000  # Example max feature size

for ngram_range in ngram_ranges:
    # BoW Experiments
    run_experiment("BoW", ngram_range, max_features, vectorizer_name="BoW")

    # TF-IDF Experiments
    run_experiment("TF-IDF", ngram_range, max_features, vectorizer_name="TF-IDF")


