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


from utilities import merge_df


def preprocess():
  vehicle_df = pd.read_csv("./datasets/filtered_vehicle.csv", usecols=["ACCIDENT_NO", "VEHICLE_TYPE", "VEHICLE_TYPE_DESC"])
  person_df = pd.read_csv("./datasets/person.csv", usecols=["ACCIDENT_NO", "SEATING_POSITION", "HELMET_BELT_WORN", "INJ_LEVEL", "INJ_LEVEL_DESC"])

  merged_df = merge_df(vehicle_df, person_df, "ACCIDENT_NO")

  print(merged_df.head())


preprocess()