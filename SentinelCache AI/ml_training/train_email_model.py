import os
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'preprocessing', 'preprocessed_data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load Preprocessed Data ──
print("=" * 60)
print("EMAIL PHISHING MODEL TRAINING")
print("=" * 60)

X = sparse.load_npz(os.path.join(DATA_DIR, 'email_tfidf_features.npz'))
y = np.load(os.path.join(DATA_DIR, 'email_labels.npy'))

print(f"Features shape: {X.shape}")
print(f"Labels shape:   {y.shape}")
print(f"Class distribution: Ham={sum(y==0)}, Spam={sum(y==1)}")

# ── Train/Test Split (80/20) ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# ── Define Models ──
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='linear', random_state=42),
    'XGBoost': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

# ── Train & Evaluate All Models ──
results = []

for name, model in models.items():
    print(f"\n{'-' * 40}")
    print(f"Training: {name}")
    print(f"{'-' * 40}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'model_obj': model
    })

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# ── Comparison Table ──
print("\n" + "=" * 60)
print("MODEL COMPARISON (Email)")
print("=" * 60)

results_df = pd.DataFrame(results).drop(columns=['model_obj'])
results_df = results_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))

# ── Best Model ──
best = max(results, key=lambda x: x['F1 Score'])
print(f"\nBest Model: {best['Model']} (F1 Score: {best['F1 Score']:.4f})")

# ── Save Best Model as .pkl ──
pkl_path = os.path.join(MODEL_DIR, 'email_model.pkl')
joblib.dump(best['model_obj'], pkl_path)
print(f"Model saved to: {pkl_path}")

print("\nEmail model training COMPLETE!")

# ============================================================
# ALTERNATIVE IMPLEMENTATION FROM HASEEB BRANCH
# ============================================================

def train_email_model_alternative():
    """
    Alternative Email phishing detection model implementation
    Uses Naive Bayes with TF-IDF pipeline
    """
    print("\n" + "=" * 50)
    print("📧 ALTERNATIVE EMAIL MODEL (Haseeb's Implementation)")
    print("=" * 50)
    
    try:
        import pickle
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        
        print("Waiting for spam.csv dataset...")
        print("Once dataset is available, implement:")
        print("1. Load spam.csv")
        print("2. Preprocess email text")
        print("3. Create TF-IDF + Naive Bayes pipeline")
        print("4. Train and evaluate model")
        print("5. Save as backend/models/email_model_alternative.pkl")
        
        # Create test pipeline for now
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000)),
            ('nb', MultinomialNB())
        ])
        
        # Train with dummy data
        dummy_texts = ['sample email text', 'test message', 'phishing attempt', 'normal email']
        dummy_labels = [0, 0, 1, 0]
        pipeline.fit(dummy_texts, dummy_labels)
        
        # Save alternative model
        alt_model_path = os.path.join(MODEL_DIR, 'email_model_alternative.pkl')
        with open(alt_model_path, 'wb') as f:
            pickle.dump(pipeline, f)
        
        print(f"✅ Alternative model saved to: {alt_model_path}")
        
        # Example metrics (placeholder)
        print("\n📈 Model Performance (placeholder):")
        print(f"Accuracy: 0.96")
        print(f"Precision: 0.95")
        print(f"Recall: 0.94")
        print(f"F1 Score: 0.94")
        
    except Exception as e:
        print(f"Alternative training skipped: {e}")

# Run both implementations if needed
if __name__ == "__main__":
    # Original training runs automatically
    # Alternative is available as a function
    pass