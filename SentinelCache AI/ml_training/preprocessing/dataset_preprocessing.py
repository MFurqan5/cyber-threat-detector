import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
OUTPUT_DIR = os.path.join(BASE_DIR, 'preprocessing', 'preprocessed_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Setup complete. Output will be saved to:", OUTPUT_DIR)

# PART 1: EMAIL DATASET PREPROCESSING
email_df = pd.read_csv(os.path.join(DATASET_DIR, 'emails.csv'))

print(f"\nShape: {email_df.shape}")
print(f"Columns: {email_df.columns.tolist()}")
print(f"\nFirst 3 rows:")
print(email_df.head(3))
print(f"\nData types:\n{email_df.dtypes}")
print(f"\nBasic stats:\n{email_df.describe()}")

# Missing Values Check
print("\n--- Missing Values ---")
print(email_df.isnull().sum())
print(f"\nTotal missing: {email_df.isnull().sum().sum()}")

# Duplicate Check & Removal
print("\n--- Duplicates ---")
dup_count = email_df.duplicated().sum()
print(f"Duplicate rows found: {dup_count}")

if dup_count > 0:
    email_df = email_df.drop_duplicates().reset_index(drop=True)
    print(f"After removing duplicates: {email_df.shape}")

# Class Distribution Analysis
print("\n--- Class Distribution ---")
class_counts = email_df['spam'].value_counts()
print(class_counts)
print(f"\nClass ratio (spam/ham): {class_counts[1] / class_counts[0]:.3f}")

# Visualization: Class Distribution Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart
colors = ['#2ecc71', '#e74c3c']
class_counts.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black')
axes[0].set_title('Email Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Class (0=Ham, 1=Spam)')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Ham (0)', 'Spam (1)'], rotation=0)
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 30, str(v), ha='center', fontweight='bold')

# Pie chart
axes[1].pie(class_counts.values, labels=['Ham', 'Spam'], autopct='%1.1f%%',
            colors=colors, startangle=90, explode=(0, 0.05))
axes[1].set_title('Email Class Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, 'email_class_distribution.png'), dpi=150)
plt.close()
print("Observation: Dataset is imbalanced — ~76% Ham vs ~24% Spam.")

# Text Length Analysis
print("\n--- Text Length Analysis ---")
email_df['text_length'] = email_df['text'].str.len()
email_df['word_count'] = email_df['text'].str.split().str.len()

print(email_df[['text_length', 'word_count']].describe())

# Visualization: Text length distribution by class
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for label, color, name in [(0, '#2ecc71', 'Ham'), (1, '#e74c3c', 'Spam')]:
    subset = email_df[email_df['spam'] == label]
    axes[0].hist(subset['text_length'], bins=50, alpha=0.6, color=color, label=name, edgecolor='black')
    axes[1].hist(subset['word_count'], bins=50, alpha=0.6, color=color, label=name, edgecolor='black')

axes[0].set_title('Text Length Distribution by Class', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Character Count')
axes[0].set_ylabel('Frequency')
axes[0].legend()

axes[1].set_title('Word Count Distribution by Class', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Word Count')
axes[1].set_ylabel('Frequency')
axes[1].legend()

plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, 'email_text_length_analysis.png'), dpi=150)
plt.close()
print("Observation: Spam emails tend to be shorter on average than ham emails.")

# Text Cleaning
print("\n--- Text Cleaning ---")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'^subject\s*:\s*', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

email_df['cleaned_text'] = email_df['text'].apply(clean_text)

print("Before cleaning:")
print(email_df['text'].iloc[0][:150])
print("\nAfter cleaning:")
print(email_df['cleaned_text'].iloc[0][:150])

# Stopword Removal & Lemmatization
print("\n--- Stopword Removal & Lemmatization ---")

import nltk
try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

email_df['processed_text'] = email_df['cleaned_text'].apply(preprocess_text)

print("After stopword removal & lemmatization:")
print(email_df['processed_text'].iloc[0][:200])

# Check for any empty texts after processing
empty_count = (email_df['processed_text'].str.len() == 0).sum()
print(f"\nEmpty texts after processing: {empty_count}")
if empty_count > 0:
    email_df = email_df[email_df['processed_text'].str.len() > 0].reset_index(drop=True)
    print(f"Removed empty texts. New shape: {email_df.shape}")

# TF-IDF Vectorization
print("\n--- TF-IDF Vectorization ---")

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_email_tfidf = tfidf.fit_transform(email_df['processed_text'])
y_email = email_df['spam'].values

print(f"TF-IDF matrix shape: {X_email_tfidf.shape}")
print(f"Target shape: {y_email.shape}")
print(f"Feature names (first 20): {tfidf.get_feature_names_out()[:20].tolist()}")

# Save Preprocessed Email Data
print("\n--- Saving Preprocessed Email Data ---")

from scipy import sparse

# Save TF-IDF sparse matrix
sparse.save_npz(os.path.join(OUTPUT_DIR, 'email_tfidf_features.npz'), X_email_tfidf)
# Save target labels
np.save(os.path.join(OUTPUT_DIR, 'email_labels.npy'), y_email)
# Save cleaned text CSV (useful for deep learning later)
email_df[['processed_text', 'spam']].to_csv(
    os.path.join(OUTPUT_DIR, 'emails_cleaned.csv'), index=False
)
# Save TF-IDF vectorizer feature names for later use
pd.Series(tfidf.get_feature_names_out()).to_csv(
    os.path.join(OUTPUT_DIR, 'email_tfidf_feature_names.csv'), index=False
)

print("Saved: email_tfidf_features.npz, email_labels.npy, email_tfidf_feature_names.csv")
print("Email preprocessing COMPLETE!\n")

# PART 2: WEBSITE PHISHING DATASET PREPROCESSING
url_df = pd.read_csv(os.path.join(DATASET_DIR, 'Website Phishing.csv'))

print(f"\nShape: {url_df.shape}")
print(f"Columns: {url_df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{url_df.head()}")
print(f"\nData types:\n{url_df.dtypes}")
print(f"\nBasic stats:\n{url_df.describe()}")

# Missing Values Check
print("\n--- Missing Values ---")
print(url_df.isnull().sum())
print(f"\nTotal missing: {url_df.isnull().sum().sum()}")

# Duplicate Check & Removal
print("\n--- Duplicates ---")
dup_count = url_df.duplicated().sum()
print(f"Duplicate rows found: {dup_count}")

if dup_count > 0:
    url_df = url_df.drop_duplicates().reset_index(drop=True)
    print(f"After removing duplicates: {url_df.shape}")

# Unique Values per Feature
print("\n--- Unique Values per Feature ---")
for col in url_df.columns:
    print(f"  {col}: {sorted(url_df[col].unique())}")

# Class Distribution (Target = 'Result')
print("\n--- Target Class Distribution ---")
target_counts = url_df['Result'].value_counts().sort_index()
print(target_counts)
print(f"\nClasses: -1 = Phishing, 0 = Suspicious, 1 = Legitimate")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = ['#e74c3c', '#f39c12', '#2ecc71']
labels = ['Phishing (-1)', 'Suspicious (0)', 'Legitimate (1)']

target_counts.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black')
axes[0].set_title('Website Phishing - Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Result')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(labels, rotation=15)
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

axes[1].pie(target_counts.values, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90, explode=(0.05, 0.05, 0))
axes[1].set_title('Website Phishing - Class Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, 'url_class_distribution.png'), dpi=150)
plt.close()

# Convert to Binary Classification
print("\n--- Converting to Binary Classification ---")
print("Mapping: -1 (Phishing) & 0 (Suspicious) -> 0 (Malicious)")
print("         1 (Legitimate) -> 1 (Safe)")

url_df['Result_binary'] = url_df['Result'].apply(lambda x: 1 if x == 1 else 0)

binary_counts = url_df['Result_binary'].value_counts()
print(f"\nBinary class distribution:")
print(f"  0 (Malicious): {binary_counts[0]}")
print(f"  1 (Safe):      {binary_counts[1]}")

# Correlation Heatmap
print("\n--- Correlation Analysis ---")

feature_cols = [c for c in url_df.columns if c not in ['Result', 'Result_binary']]
corr_matrix = url_df[feature_cols + ['Result_binary']].corr()

plt.figure(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            mask=mask, square=True, linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Feature Correlation Heatmap (Website Phishing)', fontsize=14, fontweight='bold')
plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, 'url_correlation_heatmap.png'), dpi=150)
plt.close()

# Correlations with target
print("\nCorrelation with target (Result_binary):")
target_corr = corr_matrix['Result_binary'].drop('Result_binary').sort_values(ascending=False)
print(target_corr)
print("\nObservation: Features most correlated with legitimacy can be identified above.")

# Feature Distribution per Class
print("\n--- Feature Distribution per Class ---")

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    for label, color, name in [(0, '#e74c3c', 'Malicious'), (1, '#2ecc71', 'Safe')]:
        subset = url_df[url_df['Result_binary'] == label][col]
        axes[i].hist(subset, alpha=0.6, color=color, label=name, edgecolor='black', bins=5)
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].legend(fontsize=8)

plt.suptitle('Feature Distributions by Class', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, 'url_feature_distributions.png'), dpi=150)
plt.close()

# Feature Scaling (StandardScaler)
print("\n--- Feature Scaling ---")

from sklearn.preprocessing import StandardScaler

X_url = url_df[feature_cols].values
y_url = url_df['Result_binary'].values

scaler = StandardScaler()
X_url_scaled = scaler.fit_transform(X_url)

print(f"Features shape: {X_url_scaled.shape}")
print(f"Target shape:   {y_url.shape}")
print(f"\nScaled feature means (should be ~0): {X_url_scaled.mean(axis=0).round(4)}")
print(f"Scaled feature stds  (should be ~1): {X_url_scaled.std(axis=0).round(4)}")

# Save Preprocessed Website Phishing Data
print("\n--- Saving Preprocessed Website Phishing Data ---")

# Save scaled features and labels
np.save(os.path.join(OUTPUT_DIR, 'url_features_scaled.npy'), X_url_scaled)
np.save(os.path.join(OUTPUT_DIR, 'url_labels.npy'), y_url)
# Save feature names
pd.Series(feature_cols).to_csv(
    os.path.join(OUTPUT_DIR, 'url_feature_names.csv'), index=False
)
# Save full cleaned dataframe
# url_df[feature_cols + ['Result_binary']].to_csv(
#     os.path.join(OUTPUT_DIR, 'website_phishing_cleaned.csv'), index=False
# )

print("Saved: url_features_scaled.npy, url_labels.npy, url_feature_names.csv")
print("Website phishing preprocessing COMPLETE!\n")

# Final Summary
print("=" * 60)
print("PREPROCESSING SUMMARY")
print("=" * 60)
print(f"""
EMAIL DATASET:
  - Original:     5728 rows × 2 cols
  - After dedup:  {len(email_df)} rows
  - Cleaning:     Lowercase, remove URLs/emails/HTML/numbers/special chars
  - NLP:          Stopword removal + Lemmatization
  - Vectorized:   TF-IDF (max_features=5000, ngrams=1-2)
  - Output shape: {X_email_tfidf.shape}
  - Saved files:  email_tfidf_features.npz, email_labels.npy, email_tfidf_feature_names.csv

WEBSITE PHISHING DATASET:
  - Original:     1353 rows × 10 cols
  - After dedup:  {len(url_df)} rows
  - Target:       Converted 3-class -> binary (Malicious=0, Safe=1)
  - Scaling:      StandardScaler applied
  - Output shape: {X_url_scaled.shape}
  - Saved files:  url_features_scaled.npy, url_labels.npy, url_feature_names.csv

All preprocessed data saved to: {OUTPUT_DIR}
""")
