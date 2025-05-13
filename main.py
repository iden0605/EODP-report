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
from correlation import correlation
#from supervisedLearning import supervisedLearning
#from clustering import clustering


SEAT_BELT_MAP = {
  "worn": 1,
  "not worn": 8
}

SEAT_POS_MAP = {
  "D": "Driver",
  "LF": "Left Front",
  "LR": "Left Rear",
  "RR": "Right Rear",
  'OR': "Other",
  "NK": "Not Known",
  "CR": "Centre Rear",
  "CF": "Centre Front",
  "PL": "Pillion"
}

def main():
  # getting relavant columns from dataframes
  vehicle_df = pd.read_csv("./datasets/filtered_vehicle.csv", usecols=["ACCIDENT_NO", "VEHICLE_TYPE", "VEHICLE_TYPE_DESC"])
  person_df = pd.read_csv("./datasets/person.csv", usecols=["ACCIDENT_NO", "SEATING_POSITION", "HELMET_BELT_WORN", "INJ_LEVEL", "INJ_LEVEL_DESC"])

  # merging the dataframes on accident number key
  merged_df = merge_df(vehicle_df, person_df, "ACCIDENT_NO")

  # only keeping rows that involve seatbelts
  merged_df = merged_df[merged_df["HELMET_BELT_WORN"].isin(SEAT_BELT_MAP.values())]

  # removing vehicle types that are not known or not applicable
  merged_df = merged_df[~merged_df["VEHICLE_TYPE_DESC"].isin(["Not Known", "Not Applicable"])]

  # drop rows with NaN values
  merged_df = merged_df.dropna(subset=["INJ_LEVEL", "HELMET_BELT_WORN", "SEATING_POSITION", "VEHICLE_TYPE_DESC"])

  # decoding
  inj_level_dict = dict(zip(merged_df["INJ_LEVEL"], merged_df["INJ_LEVEL_DESC"]))
  vehicle_type_dict = dict(zip(merged_df["VEHICLE_TYPE"], merged_df["VEHICLE_TYPE_DESC"]))

  #correlation(merged_df, inj_level_dict, vehicle_type_dict)
  #supervisedLearning(merged_df)
  #clustering(merged_df)

main()