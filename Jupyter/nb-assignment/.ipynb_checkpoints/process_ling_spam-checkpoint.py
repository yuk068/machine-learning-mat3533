import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

df = pd.read_csv('data/ling_spam/messages.csv')

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        
        text = re.sub(r'[^a-z\s]', '', text)
        
        words = text.split()
        
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

        return ' '.join(words)
    else:
        return ''

df['processed_message'] = df['message'].apply(preprocess_text)
df['processed_subject'] = df['subject'].apply(preprocess_text)

spam_df = df[df['label'] == 1].sample(n=480, random_state=42)
ham_df = df[df['label'] == 0].sample(n=480, random_state=42)

balanced_df = pd.concat([spam_df, ham_df])

balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

balanced_df[['processed_subject', 'processed_message', 'label']].to_csv('data/ling_spam/processed_messages.csv', index=False)

print("Preprocessing complete. Processed file saved as 'processed_messages.csv'.")
