# model_train.py
import pandas as pd
import numpy as np
import pickle
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

MODELSDIR = "models"
os.makedirs(MODELSDIR, exist_ok=True)
DFPATH = "exposure_data.csv"
df = pd.read_csv(DFPATH)

# -------- utility --------
def split_and_prepare(text):
    if isinstance(text, str):
        parts = re.split(r',|;', text.lower())
        return [p.strip() for p in parts if p.strip()]
    return []

# -------- prepare dataset --------
df["symptoms_list"] = df["symptoms"].apply(split_and_prepare)
df["treatment_list"] = df["treatment"].apply(split_and_prepare)

# engineered features
df["symptom_count"] = df["symptoms_list"].apply(len)
df["treatment_count"] = df["treatment_list"].apply(len)
df["severity_index"] = df["exposure_duration"] * df["symptom_count"]

# -------- Clean + Collapse Treatment Classes --------
df["suggested_measure"] = df["suggested_measure"].astype(str).str.strip()
df["suggested_measure"] = df["suggested_measure"].str.replace(r"\s+", " ", regex=True)
df["suggested_measure"] = df["suggested_measure"].str.strip()

RARE_THRESHOLD = 600
treat_counts = df["suggested_measure"].value_counts()
rare_classes = treat_counts[treat_counts < RARE_THRESHOLD].index
df.loc[df["suggested_measure"].isin(rare_classes), "suggested_measure"] = "Get Consultation"

print("Treatment class distribution after collapsing:\n")
print(df["suggested_measure"].value_counts())
print("\nUnique treatment labels now:", df["suggested_measure"].unique())

# multi-hot encoding for features
symptoms_mlb = MultiLabelBinarizer()
symptoms_encoded = symptoms_mlb.fit_transform(df["symptoms_list"])
symptoms_df = pd.DataFrame(symptoms_encoded, columns=[f"symptom_{s}" for s in symptoms_mlb.classes_])

treatment_mlb = MultiLabelBinarizer()
treatment_encoded = treatment_mlb.fit_transform(df["treatment_list"])
treatment_df = pd.DataFrame(treatment_encoded, columns=[f"treatment_{t}" for t in treatment_mlb.classes_])

dffull = pd.concat([df.reset_index(drop=True), symptoms_df, treatment_df], axis=1)

# features and targets
raw_feature_cols = ["age", "gender", "industry", "chemical", "exposure_duration",
                    "symptom_count", "treatment_count", "severity_index"]
raw_feature_cols += symptoms_df.columns.tolist() + treatment_df.columns.tolist()
X = dffull[raw_feature_cols]
y_outcome = df["outcome"]
y_treatment = df["suggested_measure"]

categorical_features = ["gender", "industry", "chemical"]
numeric_features = ["age", "exposure_duration", "symptom_count", "treatment_count", "severity_index"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numeric_features)
], remainder="passthrough")

# split
X_train_raw, X_test_raw, y_out_train, y_out_test, y_treat_train, y_treat_test = train_test_split(
    X, y_outcome, y_treatment, test_size=0.2, random_state=42, stratify=y_outcome)

X_train_processed = preprocessor.fit_transform(X_train_raw)
X_test_processed = preprocessor.transform(X_test_raw)
feature_names = preprocessor.get_feature_names_out()

X_train_final = pd.DataFrame(X_train_processed, columns=[c.replace(" ", "_") for c in feature_names])
X_test_final = pd.DataFrame(X_test_processed, columns=[c.replace(" ", "_") for c in feature_names])

# encode targets
outcome_target_encoder = LabelEncoder()
y_out_train_enc = outcome_target_encoder.fit_transform(y_out_train)
y_out_test_enc = outcome_target_encoder.transform(y_out_test)

treatment_target_encoder = LabelEncoder()
y_treat_train_enc = treatment_target_encoder.fit_transform(y_treat_train)
y_treat_test_enc = treatment_target_encoder.transform(y_treat_test)

# -------- train LightGBM --------
print("Training outcome LightGBM...")
lgb_outcome = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=len(outcome_target_encoder.classes_),
    n_estimators=500,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgb_outcome.fit(X_train_final, y_out_train_enc)

print("Training treatment LightGBM...")
lgb_treatment = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=len(treatment_target_encoder.classes_),
    n_estimators=800,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
lgb_treatment.fit(X_train_final, y_treat_train_enc)

feature_imp = pd.DataFrame({
    'Feature': X_train_final.columns,
    'Importance': lgb_outcome.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(8, 5))
plt.barh(feature_imp['Feature'], feature_imp['Importance'])
plt.xlabel('Importance Score')
plt.title('Top 10 Influential Features (Outcome Model)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# -------- evaluation --------
def plot_confusion(cm, labels, title, filename, cmap):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=labels,
                yticklabels=labels,
                cmap=cmap)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELSDIR, filename))
    plt.close()

# outcome
print("====== Outcome Model Evaluation ======")
out_lgb_pred = lgb_outcome.predict(X_test_final)
print(f"Accuracy: {accuracy_score(y_out_test_enc, out_lgb_pred):.3f}")
print(classification_report(y_out_test_enc, out_lgb_pred, target_names=outcome_target_encoder.classes_))
plot_confusion(confusion_matrix(y_out_test_enc, out_lgb_pred),
               outcome_target_encoder.classes_,
               "Confusion Matrix - Outcome",
               "confusion_outcome_lgb.png",
               "Purples")

# treatment
print("====== Treatment Model Evaluation ======")
treat_lgb_probs = lgb_treatment.predict_proba(X_test_final)
treat_lgb_pred = np.argmax(treat_lgb_probs, axis=1)
print(f"Accuracy: {accuracy_score(y_treat_test_enc, treat_lgb_pred):.3f}")
print(classification_report(y_treat_test_enc, treat_lgb_pred, target_names=treatment_target_encoder.classes_))
plot_confusion(confusion_matrix(y_treat_test_enc, treat_lgb_pred),
               treatment_target_encoder.classes_,
               "Confusion Matrix - Treatment",
               "confusion_treatment_lgb.png",
               "Oranges")



# top-3 accuracy for treatment
top3_preds = np.argsort(treat_lgb_probs, axis=1)[:, -3:]
top3_correct = sum(y_treat_test_enc[i] in top3_preds[i] for i in range(len(y_treat_test_enc)))
print(f"Treatment LightGBM Top-3 Accuracy: {top3_correct / len(y_treat_test_enc):.3f}")

# -------- save artifacts --------
lgb_outcome.booster_.save_model(os.path.join(MODELSDIR, "outcome_lgb.txt"))
lgb_treatment.booster_.save_model(os.path.join(MODELSDIR, "treatment_lgb.txt"))

with open(os.path.join(MODELSDIR, "outcome_target_encoder.pkl"), "wb") as f:
    pickle.dump(outcome_target_encoder, f)
with open(os.path.join(MODELSDIR, "treatment_target_encoder.pkl"), "wb") as f:
    pickle.dump(treatment_target_encoder, f)

encoders = {
    "preprocessor": preprocessor,
    "symptoms_mlb": symptoms_mlb,
    "treatment_mlb": treatment_mlb,
    "feature_names": feature_names,
    "numeric_features": numeric_features,
}
with open(os.path.join(MODELSDIR, "encoders.pkl"), "wb") as f:
    pickle.dump(encoders, f)

background_processed = X_train_final.sample(min(500, len(X_train_final)), random_state=42)
with open(os.path.join(MODELSDIR, "lime_background.pkl"), "wb") as f:
    pickle.dump(background_processed, f)

print("Finished. LightGBM models and reports saved in models/")