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


def preprocess():
  accident_df = pd.read_csv("./datasets/accident.csv", usecols=["ACCIDENT_NO"])
  vehicle_df = pd.read_csv("./datasets/filtered_vehicle.csv", usecols=["ACCIDENT_NO", "VEHICLE_MAKE"])
  person_df = pd.read_csv("./datasets/person.csv", usecols=["ACCIDENT_NO", "SEATING_POSITION"])

  print(accident_df.head())
  print(vehicle_df.head())
  print(person_df.head())


preprocess()