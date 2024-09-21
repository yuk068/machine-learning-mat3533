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

# Save test features and labels
with open('data/ling_spam/test-features.txt', 'w') as f:
    for index, vector in enumerate(X_test.toarray(), start=1):
        for word_index, count in enumerate(vector):
            if count > 0:
                f.write(f"{index} {word_index + 1} {count}\n")

with open('data/ling_spam/test-labels.txt', 'w') as f:
    for label in y_test:
        f.write(f"{label}\n")

# Function to save train features and labels for different sizes
def save_train_data(X_train, y_train, sizes):
    for size in sizes:
        # Take the first 'size' samples
        X_train_subset = X_train[:size]
        y_train_subset = y_train[:size]
        
        # Save train features
        with open(f'data/ling_spam/train-features-{size}.txt', 'w') as f:
            for index, vector in enumerate(X_train_subset.toarray(), start=1):
                for word_index, count in enumerate(vector):
                    if count > 0:
                        f.write(f"{index} {word_index + 1} {count}\n")

        # Save train labels
        with open(f'data/ling_spam/train-labels-{size}.txt', 'w') as f:
            for label in y_train_subset:
                f.write(f"{label}\n")

# Sizes for training subsets
sizes = [50, 100, 400]

# Save all training data
save_train_data(X_train, y_train, sizes)

# Save full train features and labels
with open('data/ling_spam/train-features.txt', 'w') as f:
    for index, vector in enumerate(X_train.toarray(), start=1):
        for word_index, count in enumerate(vector):
            if count > 0:
                f.write(f"{index} {word_index + 1} {count}\n")

with open('data/ling_spam/train-labels.txt', 'w') as f:
    for label in y_train:
        f.write(f"{label}\n")

print("Data files created successfully.")
