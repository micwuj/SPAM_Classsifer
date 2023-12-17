import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

# Reading data from csv file
# Data was taken from http://untroubled.org/spam/
alldata = pd.read_csv(
    "data.csv",
    usecols=[
        "SPAM",
        "text",
    ],
)

# Removing 'Subject: ' from the beginning of each text in the 'text' column
alldata['text'] = alldata['text'].str.replace('Subject: ', '')

# Assigning features and labels
x = alldata['text']
y = alldata['SPAM']

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Creating a Count Vectorizer to convert text data into a bag-of-words representation
cv = CountVectorizer()

# Creating a Logistic Regression model
model = LogisticRegression(max_iter=10000)

# Fitting the model with the training data
model.fit(cv.fit_transform(x_train), y_train)

# Making predictions on the test set
y_predicted = model.predict(cv.transform(x_test))

# Calculating precision, recall, F-score, and support for the binary classification
precision, recall, fscore, support = precision_recall_fscore_support(
    y_test, y_predicted, average='binary'
)

# Evaluation metrics
print(f"Precision: {precision*100:.2f} %")
print(f"Recall: {recall*100:.2f} %")
print(f"F-score: {fscore*100:.2f} %")
