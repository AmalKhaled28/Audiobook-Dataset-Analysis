# Audiobook Users Analysis & Prediction using Machine Learning
# Project Description

This project analyzes audiobook user data to understand user behavior, purchase activity, and listening patterns.
The goal is to clean, preprocess, and transform the dataset, then apply classification and clustering algorithms to predict review ratings and group users based on their listening and purchasing behaviors.

# Objectives

- Preprocess and clean audiobook user data.

- Handle missing values and irrelevant features.

- Apply binning (discretization) to transform numerical data into categories.

- Explore relationships and correlations between variables.

- Build and evaluate multiple classification models to predict user review ratings.

- Use K-Means clustering to segment users based on listening time, prices, reviews, and completion rates.

# Tools & Libraries

- Python

- Pandas, NumPy – data manipulation and analysis

- Matplotlib, Seaborn – visualization

- Scikit-learn (sklearn) – ML algorithms (Naive Bayes, Decision Tree, KNN, K-Means)

- SciPy, StandardScaler, MinMaxScaler – scaling and preprocessing

# Dataset Overview

The dataset audiobook_data_2.csv includes user activity and purchase data such as:

- Book_length(mins)_overall – Total audiobook duration

- Book_length(mins)_avg – Average book length

- Price_overall, Price_avg – Purchase price metrics

- Review, Review10/10 – Review scores

- Completion – Audiobook completion percentage

- Minutes_listened – Total listening time

- Last_Visited_mins_Purchase_date – Time between last visit and purchase

- Target – Purchase likelihood or outcome

# Data Preprocessing

- Dropped irrelevant columns (Unnamed: 0, Support_Request) and handled missing values.

- Applied discretization (binning) using pd.qcut to categorize continuous features into ranked groups.

- Removed duplicate rows.

- Normalized numerical columns using StandardScaler and MinMaxScaler.

- Visualized correlations using a heatmap to identify strongly related attributes.

# Machine Learning Models
- Gaussian Naive Bayes

  - Target: Predict user review rating.

  - Simple probabilistic model based on Gaussian distribution.

Accuracy: 89%

- Decision Tree Classifier

  - Used for classification based on review scores.

  - Visualized using tree.plot_tree().

  -  Accuracy and confusion matrix calculated for evaluation.

-  K-Nearest Neighbors (KNN)

  - Tested multiple k values to minimize error rate.

  - Best accuracy achieved at k = 7.

  - Visualized error rates to determine optimal k.

- K-Means Clustering Analysis

  - Performed multiple clustering experiments to group users based on different behavioral attributes:

  - Price_overall and Book_length(mins)_overall — relationship between price and content length.

  - Price_avg and Book_length(mins)_avg — grouping by average purchase trends.

  - Minutes_listened and Book_length(mins)_overall — listening time patterns.

  - Review10/10 and Completion — correlation between satisfaction and completion rate.

  - Last_Visited_mins_Purchase_date and Target — behavior between visits and purchases.

  - last_visit and audio_target — user retention and engagement trends.

  - Each clustering process includes:

     - Using the Elbow Method to determine the optimal number of clusters.

     - Visualizing the clusters and centroids using scatter plots.

# Visualizations

- Correlation heatmap.

- K vs Error Rate (KNN).

- Elbow plots for K-Means.

- Cluster scatter plots for each feature pair.

# Key Insights

- Review and completion rate are strong indicators of engagement.

- High correlation between price and book length.

- K-Means clustering effectively separates users into behavioral groups.

- Normalization significantly improves model performance and stability.
