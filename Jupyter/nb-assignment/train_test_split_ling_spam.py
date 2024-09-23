import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Load the processed data
df = pd.read_csv('data/ling_spam/processed_messages.csv')

# Split the dataset into features and labels
X = df['processed_message']
y = df['label']

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the messages
X_vectorized = vectorizer.fit_transform(X)

# Create train-test split (80-20% split)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Save test features and labels in CSV format with 0-indexing
test_features = []
for index, vector in enumerate(X_test.toarray()):
    for word_index, count in enumerate(vector):
        if count > 0:
            test_features.append([index, word_index, count])

test_df = pd.DataFrame(test_features, columns=["email_index", "word_index", "count"])
test_df.to_csv('data/ling_spam/test-features.csv', index=False)

# Save test labels in CSV format
y_test_df = pd.DataFrame(y_test.reset_index(drop=True), columns=["label"])
y_test_df.to_csv('data/ling_spam/test-labels.csv', index_label="email_index")

# Function to save train features and labels for different sizes
def save_train_data(X_train, y_train, sizes):
    for size in sizes:
        # Take the first 'size' samples
        X_train_subset = X_train[:size]
        y_train_subset = y_train[:size]
        
        # Save train features in CSV format with 0-indexing
        train_features = []
        for index, vector in enumerate(X_train_subset.toarray()):
            for word_index, count in enumerate(vector):
                if count > 0:
                    train_features.append([index, word_index, count])
        
        train_df = pd.DataFrame(train_features, columns=["email_index", "word_index", "count"])
        train_df.to_csv(f'data/ling_spam/train-features-{size}.csv', index=False)

        # Save train labels in CSV format
        y_train_df = pd.DataFrame(y_train_subset.reset_index(drop=True), columns=["label"])
        y_train_df.to_csv(f'data/ling_spam/train-labels-{size}.csv', index_label="email_index")

# Sizes for training subsets
sizes = [50, 100, 400]

# Save all training data
save_train_data(X_train, y_train, sizes)

# Save full train features and labels
train_features = []
for index, vector in enumerate(X_train.toarray()):
    for word_index, count in enumerate(vector):
        if count > 0:
            train_features.append([index, word_index, count])

train_df = pd.DataFrame(train_features, columns=["email_index", "word_index", "count"])
train_df.to_csv('data/ling_spam/train-features.csv', index=False)

# Save full train labels in CSV format
y_train_df = pd.DataFrame(y_train.reset_index(drop=True), columns=["label"])
y_train_df.to_csv('data/ling_spam/train-labels.csv', index_label="email_index")

print("Data files created successfully.")
