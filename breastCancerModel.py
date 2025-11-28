import mlcroissant as mlc
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

# Fetch the Croissant JSON-LD
croissant_dataset = mlc.Dataset('https://www.kaggle.com/datasets/erdemtaha/cancer-data/croissant/download')

# Check what record sets are in the dataset
record_sets = croissant_dataset.metadata.record_sets
print(record_sets)

# Fetch the records and put them in a DataFrame
record_set_df = pd.DataFrame(croissant_dataset.records(record_set=record_sets[0].uuid))
record_set_df.head()
# automatic fetching courtesy of the Kaggle platform

# print(record_set_df.columns.to_list()) # Lists columns to know how they are formatted in the dataframe after importation
record_set_df = record_set_df.convert_dtypes() # Data type auto conversion for proper handling
record_set_df["Cancer_Data.csv/diagnosis"] = record_set_df["Cancer_Data.csv/diagnosis"].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x) # Decodes diagnosis info who is encoded as bytes, turns it into string

# Demands before continuing the selected n neighbors we want to launch KNN at
n = int(input("Enter number of neighbors to be considered in KNN : "))

# Here, we're only keeping diagnosis and the means linked columns
dataset1 = record_set_df.drop(columns=["Cancer_Data.csv/id", "Cancer_Data.csv/radius_se", "Cancer_Data.csv/texture_se", "Cancer_Data.csv/perimeter_se", "Cancer_Data.csv/area_se", "Cancer_Data.csv/smoothness_se", "Cancer_Data.csv/compactness_se", "Cancer_Data.csv/concavity_se", "Cancer_Data.csv/concave+points_se", "Cancer_Data.csv/symmetry_se", "Cancer_Data.csv/fractal_dimension_se", "Cancer_Data.csv/radius_worst", "Cancer_Data.csv/texture_worst", "Cancer_Data.csv/perimeter_worst", "Cancer_Data.csv/area_worst", "Cancer_Data.csv/smoothness_worst", "Cancer_Data.csv/compactness_worst", "Cancer_Data.csv/concavity_worst", "Cancer_Data.csv/concave+points_worst", "Cancer_Data.csv/symmetry_worst", "Cancer_Data.csv/fractal_dimension_worst"])
print(dataset1)
datasetStandardized1 = PowerTransformer(standardize=True).fit_transform(dataset1.drop(columns=["Cancer_Data.csv/diagnosis"])) # Applying Yeo Johnson to the data for standardization
datasetStandardized1 = pd.DataFrame(datasetStandardized1, columns=dataset1.drop(columns=["Cancer_Data.csv/diagnosis"]).columns, index=dataset1.index)
print(datasetStandardized1)

data1 = datasetStandardized1.iloc[0:519] # We reserve around 50 elements of our dataset to check if the model guesses correctly.
test_data1 = datasetStandardized1.iloc[518:]
target_values1 = dataset1["Cancer_Data.csv/diagnosis"].iloc[0:519] # Our target labels for the training set
KNN1 = KNeighborsClassifier(n_neighbors=n)
KNN1.fit(data1, target_values1)
predictions = KNN1.predict(test_data1) # Predicts the labels
real_values = dataset1["Cancer_Data.csv/diagnosis"].iloc[518:] # Get the real labels assigned

accuracy = 0
truepos = 0
trueneg = 0
falsepos = 0
falseneg = 0
for i in range(0,50):
    if predictions[i] == real_values.iloc[i]:
        accuracy += 1/50
        if predictions[i] == "B":
            truepos += 1
        else:
            trueneg += 1
    else:
        if predictions[i] == "B":
            falsepos += 1
        else:
            falseneg += 1
print("Statistics of means-focused dataset training\n ---------")
print("Accuracy : " + str(accuracy*100) + " %")
print("Precision : " + str((truepos/(truepos+falsepos))*100) + " %")
print("Recall : " + str((truepos/(truepos+falseneg))*100) + " %")
print("Specificity : " + str((trueneg/(trueneg+falsepos))*100) + " %")
print("AUC : " + str((truepos/(truepos+falseneg))/(trueneg/(trueneg+falsepos))))

print("Metrics from built-in libraries :")
print(metrics.classification_report(real_values, predictions))

metrics.ConfusionMatrixDisplay.from_predictions(real_values, predictions)
plt.show()

# Here, we're only keeping diagnosis and the se linked columns
dataset1 = record_set_df.drop(columns=["Cancer_Data.csv/id", "Cancer_Data.csv/radius_mean", "Cancer_Data.csv/texture_mean", "Cancer_Data.csv/perimeter_mean", "Cancer_Data.csv/area_mean", "Cancer_Data.csv/smoothness_mean", "Cancer_Data.csv/compactness_mean", "Cancer_Data.csv/concavity_mean", "Cancer_Data.csv/concave+points_mean", "Cancer_Data.csv/symmetry_mean", "Cancer_Data.csv/fractal_dimension_mean", "Cancer_Data.csv/radius_worst", "Cancer_Data.csv/texture_worst", "Cancer_Data.csv/perimeter_worst", "Cancer_Data.csv/area_worst", "Cancer_Data.csv/smoothness_worst", "Cancer_Data.csv/compactness_worst", "Cancer_Data.csv/concavity_worst", "Cancer_Data.csv/concave+points_worst", "Cancer_Data.csv/symmetry_worst", "Cancer_Data.csv/fractal_dimension_worst"])
print(dataset1)
datasetStandardized1 = PowerTransformer(standardize=True).fit_transform(dataset1.drop(columns=["Cancer_Data.csv/diagnosis"])) # Applying Yeo Johnson to the data for standardization
datasetStandardized1 = pd.DataFrame(datasetStandardized1, columns=dataset1.drop(columns=["Cancer_Data.csv/diagnosis"]).columns, index=dataset1.index)
print(datasetStandardized1)

data1 = datasetStandardized1.iloc[0:519] # We reserve around 50 elements of our dataset to check if the model guesses correctly.
test_data1 = datasetStandardized1.iloc[518:]
target_values1 = dataset1["Cancer_Data.csv/diagnosis"].iloc[0:519] # Our target labels for the training set
KNN1 = KNeighborsClassifier(n_neighbors=n)
KNN1.fit(data1, target_values1)
predictions = KNN1.predict(test_data1) # Predicts the labels
real_values = dataset1["Cancer_Data.csv/diagnosis"].iloc[518:] # Get the real labels assigned

accuracy = 0
truepos = 0
trueneg = 0
falsepos = 0
falseneg = 0
for i in range(0,50):
    if predictions[i] == real_values.iloc[i]:
        accuracy += 1/50
        if predictions[i] == "B":
            truepos += 1
        else:
            trueneg += 1
    else:
        if predictions[i] == "B":
            falsepos += 1
        else:
            falseneg += 1
print("Statistics of se-focused dataset training\n ---------")
print("Accuracy : " + str(accuracy*100) + " %")
print("Precision : " + str((truepos/(truepos+falsepos))*100) + " %")
print("Recall : " + str((truepos/(truepos+falseneg))*100) + " %")
print("Specificity : " + str((trueneg/(trueneg+falsepos))*100) + " %")
print("AUC : " + str((truepos/(truepos+falseneg))/(trueneg/(trueneg+falsepos))))

print("Metrics from built-in libraries :")
print(metrics.classification_report(real_values, predictions))

metrics.ConfusionMatrixDisplay.from_predictions(real_values, predictions)
plt.show()

# Here, we're only keeping diagnosis and the worst linked columns
dataset1 = record_set_df.drop(columns=["Cancer_Data.csv/id", "Cancer_Data.csv/radius_se", "Cancer_Data.csv/texture_se", "Cancer_Data.csv/perimeter_se", "Cancer_Data.csv/area_se", "Cancer_Data.csv/smoothness_se", "Cancer_Data.csv/compactness_se", "Cancer_Data.csv/concavity_se", "Cancer_Data.csv/concave+points_se", "Cancer_Data.csv/symmetry_se", "Cancer_Data.csv/fractal_dimension_se", "Cancer_Data.csv/radius_mean", "Cancer_Data.csv/texture_mean", "Cancer_Data.csv/perimeter_mean", "Cancer_Data.csv/area_mean", "Cancer_Data.csv/smoothness_mean", "Cancer_Data.csv/compactness_mean", "Cancer_Data.csv/concavity_mean", "Cancer_Data.csv/concave+points_mean", "Cancer_Data.csv/symmetry_mean", "Cancer_Data.csv/fractal_dimension_mean"])
print(dataset1)
datasetStandardized1 = PowerTransformer(standardize=True).fit_transform(dataset1.drop(columns=["Cancer_Data.csv/diagnosis"])) # Applying Yeo Johnson to the data for standardization
datasetStandardized1 = pd.DataFrame(datasetStandardized1, columns=dataset1.drop(columns=["Cancer_Data.csv/diagnosis"]).columns, index=dataset1.index)
print(datasetStandardized1)

data1 = datasetStandardized1.iloc[0:519] # We reserve around 50 elements of our dataset to check if the model guesses correctly.
test_data1 = datasetStandardized1.iloc[518:]
target_values1 = dataset1["Cancer_Data.csv/diagnosis"].iloc[0:519] # Our target labels for the training set
KNN1 = KNeighborsClassifier(n_neighbors=n)
KNN1.fit(data1, target_values1)
predictions = KNN1.predict(test_data1) # Predicts the labels
real_values = dataset1["Cancer_Data.csv/diagnosis"].iloc[518:] # Get the real labels assigned

accuracy = 0
truepos = 0
trueneg = 0
falsepos = 0
falseneg = 0
for i in range(0,50):
    if predictions[i] == real_values.iloc[i]:
        accuracy += 1/50
        if predictions[i] == "B":
            truepos += 1
        else:
            trueneg += 1
    else:
        if predictions[i] == "B":
            falsepos += 1
        else:
            falseneg += 1
print("Statistics of worst-focused dataset training\n ---------")
print("Accuracy : " + str(accuracy*100) + " %")
print("Precision : " + str((truepos/(truepos+falsepos))*100) + " %")
print("Recall : " + str((truepos/(truepos+falseneg))*100) + " %")
print("Specificity : " + str((trueneg/(trueneg+falsepos))*100) + " %")
print("AUC : " + str((truepos/(truepos+falseneg))/(trueneg/(trueneg+falsepos))))

print("Metrics from built-in libraries :")
print(metrics.classification_report(real_values, predictions))

metrics.ConfusionMatrixDisplay.from_predictions(real_values, predictions)
plt.show()