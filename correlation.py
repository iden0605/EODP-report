import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score

def correlation(df, inj_level_dict, vehicle_type_dict):
  # Initialize target variable and features of interest
  target_var = df["INJ_LEVEL"]
  features = ["HELMET_BELT_WORN", "SEATING_POSITION", "VEHICLE_TYPE_DESC"]

  # Initialize a dictionary to store the NMI scores
  nmi_scores = {}

  # Compute NMI of each feature and target variable using sklearn's built-in NMI function
  for feature in features:
      nmi = normalized_mutual_info_score(target_var, df[feature], average_method = "min")
      nmi_scores[feature] = nmi

  # Ouput NMI scores
  print("Normalized mutual information (NMI) scores of each feature against the target variable:")
  for feature, score in nmi_scores.items():
      print(f"INJ_LEVEL vs {feature}, NMI: {score:.4f}")

  
