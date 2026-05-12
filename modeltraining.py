import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/spam.csv", encoding='latin-1')

# Keep useful columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Show first rows
print(df.head())



from sklearn.feature_extraction.text import CountVectorizer
# Input and output
x = df['message']
y = df['label']

# Convert text into numbers
cv = CountVectorizer()

x = cv.fit_transform(x)

print(x)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Create model
model = MultinomialNB()

# Train model
model.fit(x_train, y_train)

# Predict on test data
predictions = model.predict(x_test)

# Check accuracy
print("Accuracy:", accuracy_score(y_test, predictions))


# Train model
model = MultinomialNB()

model.fit(x_train, y_train)

# Predict on test data
predictions = model.predict(x_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, predictions))