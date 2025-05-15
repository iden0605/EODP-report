import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import normalized_mutual_info_score

def correlation_analysis(df, mapping_dicts):
    """ 
    Takes a DataFrame containing the features of interest and the target variable. Performs 
    correlation analysis by investigating data distribution, computing normalized mutual information,
    and plotting the results.
    """
    
    # Initialize target variable and features of interest
    target_var = df["INJ_LEVEL"]
    features = ["HELMET_BELT_WORN", "SEATING_POSITION", "VEHICLE_TYPE"]

    # Compute and output distribution of each feature
    data_distribution(df, features, mapping_dicts)
    # Compute and output NMI scores between each feature and the target variable
    nmi_scores = compute_nmi_scores(df, features, target_var)

    # Plot a bar chart of the NMI scores
    plot_nmi_scores(nmi_scores)


def data_distribution(df, features, mapping_dicts):
    """ 
    Takes a DataFrame containing the features of interest and the target variable, and a list 
    of the features of interest. Computes the proportions of each unique value in each feature to 
    understand the data distribution within each feature. 
    """

    for feature in features:
        print(f"\nDistribution (%) of values for {feature}:")
        
        if feature in mapping_dicts:
            # Map values using the corresponding dictionary
            mapped_series = df[feature].map(mapping_dicts[feature])
    
            # Compute normalized value counts to get proportion
            proportions = mapped_series.value_counts(normalize=True)
            proportions.index.name = None

            # Convert proportions to percentages and output the formatted result
            percentages = proportions.apply(lambda x: f"{x * 100:.2f}%")
            print(percentages)


def compute_nmi_scores(df, features, target_var):
    """ 
    Takes a DataFrame containing the features of interest and the target variable, a list 
    of the features of interest, and the target variable. Computes the NMI between each feature and
    the target variable. 
    """

    # Initialize a dictionary to store the NMI scores
    nmi_scores = {}

    for feature in features:
        # Compute NMI of each feature and target variable using sklearn's built-in NMI function
        nmi = normalized_mutual_info_score(target_var, df[feature], average_method = "min")
        nmi_scores[feature] = nmi

    # Ouput NMI scores
    print("\nNMI scores of each feature against the target variable:")
    for feature, score in nmi_scores.items():
        print(f"{feature} vs INJ_LEVEL, NMI: {score:.4f}")

    return nmi_scores


def plot_nmi_scores(nmi_scores):
    """
    Takes a dictionary containing NMI scores. Plots a bar chart visualizing these NMI scores.
    """

    # Create a bar chart to visualize the NMI scores
    sns.barplot(x=list(nmi_scores.keys()), y=list(nmi_scores.values()))

    # Add plot title and labels
    plt.title("NMI Scores with INJ_LEVEL")
    plt.xlabel("Features")
    plt.ylabel("NMI Scores")
    
    # Set y-axis limit to 0.02 as a threshold to highlight low NMI values
    plt.ylim(0, 0.02)
    plt.axhline(y=0.02, color='red', linestyle='--', linewidth=5, label='Low NMI threshold')
    plt.legend()

    # Adjust layout and display plot
    plt.tight_layout()
    plt.savefig('figures/NMI-bar-chart.png')
    plt.close()