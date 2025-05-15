import pandas as pd

from scripts.utilities import merge_df
from scripts.correlation import correlation_analysis;
from scripts.supervisedLearning import supervisedLearning
from scripts.clustering import clustering

SEAT_BELT_MAP = {
  1.0: "Seatbelt Worn",
  8.0: "Seatbelt Not Worn"
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
  # Getting relavant columns from dataframes
  vehicle_df = pd.read_csv("./datasets/vehicle.csv", usecols=["ACCIDENT_NO", "VEHICLE_TYPE", "VEHICLE_TYPE_DESC"])
  person_df = pd.read_csv("./datasets/person.csv", usecols=["ACCIDENT_NO", "SEATING_POSITION", "HELMET_BELT_WORN", "INJ_LEVEL", "INJ_LEVEL_DESC"])

  # Merging the dataframes on accident number key
  merged_df = merge_df(vehicle_df, person_df, "ACCIDENT_NO")

  # Preprocessing
  merged_df = preprocess_df(merged_df)

  # Decoding injury level and vehicle type integer categories into their descriptions using a dictionary
  inj_level_dict = dict(zip(merged_df["INJ_LEVEL"], merged_df["INJ_LEVEL_DESC"]))
  vehicle_type_dict = dict(zip(merged_df["VEHICLE_TYPE"], merged_df["VEHICLE_TYPE_DESC"]))

  mapping_dicts = {
      "INJ_LEVEL": inj_level_dict,
      "HELMET_BELT_WORN": SEAT_BELT_MAP,
      "SEATING_POSITION": SEAT_POS_MAP,
      "VEHICLE_TYPE": vehicle_type_dict,
  }

  # Drop description columns
  merged_df.drop(columns=["VEHICLE_TYPE_DESC", "INJ_LEVEL_DESC"], inplace=True)

  print("\n=====================================CORRELATION=====================================\n")
  correlation_analysis(merged_df, mapping_dicts)

  print("\n=====================================CLUSTERING======================================\n")
  clustering(merged_df, mapping_dicts)

  print("\n=================================SUPERVISED LEARNING=================================\n")
  supervisedLearning(merged_df, mapping_dicts)
  

def preprocess_df(df):
  """
  Takes a parameter df as a dataframe and performs preprocessing on it, including keeping relevant rows,
  removing known feature columns and dropping rows with NaN values. Returns the preprocessed dataframe.
  """
  # Only keeping rows that involve seatbelts
  df = df[df["HELMET_BELT_WORN"].isin(SEAT_BELT_MAP.keys())]

  # Removing vehicle types that are not known or not applicable
  df = df[~df["VEHICLE_TYPE_DESC"].isin(["Not Known", "Not Applicable"])]

  # Drop seating positions with NaN values
  df = df.dropna(subset=["SEATING_POSITION"])

  return df


main()