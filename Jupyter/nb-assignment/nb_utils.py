import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

def load_features_ling_spam(file_path):
    df = pd.read_csv(file_path, skiprows=1, header=None, names=['email_index', 'word_index', 'count'])
    
    num_emails = df['email_index'].max() + 1
    
    num_words = 23859
    
    return csr_matrix((df['count'], (df['email_index'], df['word_index'])), 
                      shape=(num_emails, num_words))


def load_labels_ling_spam(file_path):
    df = pd.read_csv(file_path, skiprows=1, header=None, names=['email_index', 'label'])
    
    df = df.sort_values('email_index')
    
    return df['label'].values


def check_data_integrity_ling_spam(features_file, labels_file):
    features = pd.read_csv(features_file, skiprows=1, header=None, names=["email_index", "word_index", "count"])
    labels = pd.read_csv(labels_file, skiprows=1, header=None, names=["email_index", "label"])
    
    unique_email_indices_in_features = features["email_index"].nunique()
    num_labels = len(labels)
    
    print(f"1. Number of unique emails in features: {unique_email_indices_in_features}")
    print(f"2. Number of labels: {num_labels}")
    
    if unique_email_indices_in_features != num_labels:
        print(f"   Error: Numbers do not match.")
    else:
        print(f"   Match: Numbers are equal.")
    
    label_counts = labels["label"].value_counts()
    print(f"3. Balance of dataset:")
    print(f"   Non-spam (0): {label_counts.get(0, 0)}")
    print(f"   Spam (1): {label_counts.get(1, 0)}")
    
    if len(label_counts) == 2:
        balance_ratio = label_counts.min() / label_counts.max()
        print(f"4. Balance ratio: {balance_ratio:.2f}")
    else:
        print("4. Balance ratio: Cannot be calculated (not a binary classification)")
    
    missing_values = features.isnull().sum().sum() + labels.isnull().sum().sum()
    print(f"5. Total missing values: {missing_values}")
    
    min_word_index = features["word_index"].min()
    max_word_index = features["word_index"].max()
    print(f"6. Word index range: {min_word_index} to {max_word_index}")
    
    expected_indices = set(range(num_labels))
    actual_indices = set(labels["email_index"])
    if expected_indices == actual_indices:
        print("7. Email indices: Continuous and start from 0")
    else:
        print("7. Email indices: Not continuous or don't start from 0")
    
    if (unique_email_indices_in_features == num_labels and 
        missing_values == 0 and 
        min_word_index >= 0 and 
        expected_indices == actual_indices):
        print("\nOverall: Data integrity check passed.\n")
    else:
        print("\nOverall: Data integrity check failed. Please review the above points.\n")


def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")


def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print(f"True Negatives: {cm[0, 0]}")
    print(f"False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}")
    print(f"True Positives: {cm[1, 1]}")
    print()