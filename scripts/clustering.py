import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from scripts.utilities import one_hot_encode_df


def clustering(df, mapping_dicts):
    clustering_df = df.copy()
    categorical_columns = ['SEATING_POSITION', 'HELMET_BELT_WORN', 'VEHICLE_TYPE']
    
    encoded_df = one_hot_encode_df(df, categorical_columns)
    
    # Scale injury
    scaler = MinMaxScaler()
    numeric_features = clustering_df[['INJ_LEVEL']]
    scaled_numeric = scaler.fit_transform(numeric_features)
    scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=['INJ_LEVEL_scaled'])

    # Combine features for clustering
    cluster_input = pd.concat([encoded_df, scaled_numeric_df], axis=1)

    # Find optimal k
    find_optimal_k(cluster_input)

    # Run KMeans Clustering
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clustering_df['Cluster'] = kmeans.fit_predict(cluster_input)

    # Convert INJ_LEVEL to integer for better sorting
    clustering_df['INJ_LEVEL'] = clustering_df['INJ_LEVEL'].astype(int)
    
    # Create cross-tabulation of clusters vs injury levels
    cluster_injury_tab = pd.crosstab(
        index=clustering_df['INJ_LEVEL'], 
        columns=clustering_df['Cluster'],
        normalize=False
    )

    # Reverse INJ_LEVEL order to put highest severity at the top
    cluster_injury_tab = cluster_injury_tab.sort_index(ascending=False)


    inj_level_map = mapping_dicts["INJ_LEVEL"]
    cluster_injury_tab.index = cluster_injury_tab.index.map(lambda x: inj_level_map[int(x)])

    # Sort the columns in ascending order
    cluster_injury_tab = cluster_injury_tab.sort_index(axis=1)

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cluster_injury_tab, 
        annot=True,
        fmt='d',
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={'label': 'Count'}
    )

    plt.title("Relationship Between Injury Severity and Clusters")
    plt.ylabel("Injury Level")
    plt.xlabel("Cluster")
    plt.tight_layout()
    plt.savefig("figures/clustering-heatmap.png")
    plt.close()

    # Analyze clusters
    for i in range(optimal_k):
        print(f"\n=== Cluster {i} ===")
        cluster_data = clustering_df[clustering_df['Cluster'] == i]
        
        # Seating Position
        print_distribution(cluster_data, 'SEATING_POSITION', mapping_dicts["SEATING_POSITION"], "Top seating positions")
        
        # Seatbelt Usage
        print_distribution(cluster_data, 'HELMET_BELT_WORN', mapping_dicts["HELMET_BELT_WORN"], "Seatbelt usage")

        # Vehicle Type
        print_distribution(cluster_data, 'VEHICLE_TYPE', mapping_dicts["VEHICLE_TYPE"], "Vehicle types")

        # Average Injury Level
        print("\nAvg Injury Level:", cluster_data['INJ_LEVEL'].mean())

        # Add injury level distribution for each cluster
        inj_level_counts = cluster_data['INJ_LEVEL'].value_counts(normalize=True) * 100
        inj_level_counts = inj_level_counts.sort_index()
        print("\nInjury level distribution:")
        for level, perc in inj_level_counts.items():
            print(f"Level {level}: {perc:.2f}%")


def find_optimal_k(data):
    """
    Applies the Elbow Method onto data to find the optimal number of clusters (k)
    for KMeans clustering.
    """
    distortions = []
    k_range = range(1, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
         # Sum of squared distances to cluster centers (intertia)
        distortions.append(kmeans.inertia_) 

    plt.plot(k_range, distortions, 'bx-')
    plt.title('The Elbow Method showing the optimal k')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.savefig("figures/elbow-method.png")
    plt.close()


def print_distribution(cluster_data, column_name, mapping_dict, title):
    """
    Prints unique value counts and percentage distribution of a feature within a cluster
    using the data provided in the parameters.
    """
    counts = cluster_data[column_name].value_counts()
    percents = cluster_data[column_name].value_counts(normalize=True) * 100
    percents = percents.round(2)
    labels = counts.index.to_series().map(mapping_dict)
    
    print(f"\n{title}:")
    for label, perc in zip(labels, percents):
        print(f"{label}: {perc}%")