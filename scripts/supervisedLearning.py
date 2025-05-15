import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from scripts.utilities import one_hot_encode_df;

def supervisedLearning(df, mapping_dicts):
  X_COLS = ['SEATING_POSITION', 'HELMET_BELT_WORN', 'VEHICLE_TYPE']
  y_COL = 'INJ_LEVEL'
  
  encoded_df = one_hot_encode_df(df, X_COLS)
  
  X = encoded_df
  y = df[y_COL]

  # Split data into training and testing sets (80:20)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Fit the model using knn
  k = 3

  # Use cross validation
  cv_scores_knn = cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=5)
  print(f"KNN Cross-Validation Accuracy: {cv_scores_knn.mean():.3f} (+/- {cv_scores_knn.std():.3f})")
  
  # Knn model fitting
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train, y_train)
  accuracy = knn.score(X_test, y_test)
  print('KNN Accuracy', accuracy)

  # Decision tree model fitting
  dt = DecisionTreeClassifier(criterion='entropy')
  cv_scores_dt = cross_val_score(dt, X, y, cv=5)
  print(f"DT Cross-Validation Accuracy: {cv_scores_dt.mean():.3f} (+/- {cv_scores_dt.std():.3f})")
  dt.fit(X_train, y_train)
  dt_accuracy = dt.score(X_test, y_test)
  print('DT Accuracy:', dt_accuracy)

  # Getting class names and their displays for visualisations
  class_names = sorted([str(x) for x in y_test.unique()])
  display_class_names = [mapping_dicts["INJ_LEVEL"][int(x)] for x in sorted(y_test.unique())]

  # Visualise the decision tree
  plt.figure(figsize=(20, 8))
  plot_tree(
    dt,
    feature_names=X.columns.tolist(),
    class_names=display_class_names,
    filled=True,
    max_depth=2
  )

  plt.title("Decision Tree Classifier")
  plt.savefig("figures/decision-tree-classifier.png")
  plt.close()

  # Confusion matrix for KNN model
  y_pred_knn = knn.predict(X_test)
  plot_confusion_matrix(
    y_test,
    y_pred_knn,
    class_names,
    display_class_names,
    "KNN Confusion Matrix",
    "KNN-confusion-matrix.png"
  )

  # Confusion matrix for DT model
  y_pred_dt = dt.predict(X_test)
  plot_confusion_matrix(
    y_test,
    y_pred_dt,
    class_names,
    display_class_names,
    "Decision Tree Confusion Matrix",
    "DT-confusion-matrix.png"
  )


  # Decoding for classification report
  y_test_decoded = y_test.map(mapping_dicts["INJ_LEVEL"])
  y_pred_knn_decoded = pd.Series(y_pred_knn).map(mapping_dicts["INJ_LEVEL"])
  y_pred_dt_decoded = pd.Series(y_pred_dt).map(mapping_dicts["INJ_LEVEL"])

  # Classification report
  print("\nKNN Classification Report:")
  print(classification_report(y_test_decoded, y_pred_knn_decoded))

  print("\nDecision Tree Classification Report:")
  print(classification_report(y_test_decoded, y_pred_dt_decoded, zero_division=0))



def plot_confusion_matrix(y_true, y_pred, labels, display_labels, title, filename):
    """
    Plots and saves a confusion matrix using the sets and labels provided in the parameters.
    """
    # Convert to strings so confusion matrix can display them
    y_true_str = y_true.astype(str)
    y_pred_str = pd.Series(y_pred).astype(str)
    
    cm = confusion_matrix(y_true_str, y_pred_str, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    # Plotting confusion matrix
    display.plot()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig('figures/' + filename)
    plt.close()