# EODP-report

===========================RUN CODE===========================
To download reqs: be in the eodp-report directory

For mac:
1. python3 -m venv myenv
2. source myenv/bin/activate
3. pip install -r requirements.txt

For windows:
1. python -m venv myenv
2. myenv\Scripts\activate
3. pip install -r requirements.txt

To run code:
python main.py


=======================SUMMARY OF PROGRAM=====================
- All code is run in the main.py file, all used scripts in main are in the scripts folder

- The code is first preprocessed using the preprocess_df() function in the main.py file

- Then we perform correlation analysis onto the dataset, this involves displaying the distributions of the unqiue values of each feature and calculating the NMI score for each feature. We visualised these NMI results using a barchart.

- After this, we perform K-Means clustering on the dataset The Elbow Method is used to determine the optimal number of clusters (k). We visualize the relationship between injury severity and the clusters through heatmap of injury levels across the clusters.

- Finally, we train supervised learning models (KNN and DT) and monitor their performance. We used k-fold cross-validation to obtain more reliable accuracy estimates. We checked and analyed their performance using their accuracy scores and confusion matrices. We also used classification reports to present their performance.


=======================PROGRAM OUTPUTS========================
- All images are saved into the figures folder.
- Results and some statistics are outputted into the terminal.