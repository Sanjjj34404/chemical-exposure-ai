# explain.py
import streamlit as st
import pandas as pd
import pickle
import os
import re
import numpy as np
import lightgbm as lgb
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from groq import Groq
from dotenv import load_dotenv
import textstat

# =======================
# Groq Setup
# =======================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

MODELS_DIR = "models"

# ---------------- utils ----------------
def split_and_prepare(text):
    if isinstance(text, str):
        parts = re.split(r'[;,]', text.lower())
        return [p.strip() for p in parts if p.strip()]
    return []

CORE_PLAIN = {
    'symptom_count': "the number of symptoms you reported",
    'treatment_count': "the number of treatments you received",
    'exposure_duration': "the duration of your exposure",
    'severity_index': "the overall severity of your exposure (duration × symptoms)",
    'age': "your age"
}

def humanize_feature(feat_descr, input_df=None):
    feat_descr = feat_descr.replace("≤","").replace("<=","").replace(">","").replace("<","")

    for key, phrase in CORE_PLAIN.items():
        if key in feat_descr:
            return phrase

    if "symptom_" in feat_descr:
        symptom_name = feat_descr.split("symptom_",1)[1].replace("_"," ")
        if input_df is not None and symptom_name in input_df["symptoms"].iloc[0].lower():
            return f"reported symptom: {symptom_name}"

    if "treatment_" in feat_descr:
        treat_name = feat_descr.split("treatment_",1)[1].replace("_"," ")
        if input_df is not None and treat_name in input_df["treatment"].iloc[0].lower():
            return f"your treatment included: {treat_name}"

    if "industry_" in feat_descr:
        ind = feat_descr.split("industry_",1)[1].replace("_"," ")
        if input_df is not None and ind.lower() in str(input_df["industry"].iloc[0]).lower():
            return f"your industry: {ind}"

    if "chemical_" in feat_descr:
        chem = feat_descr.split("chemical_",1)[1].replace("_"," ")
        if input_df is not None and chem.lower() in str(input_df["chemical"].iloc[0]).lower():
            return f"exposure to chemical: {chem}"

    return None

# -------- load artifacts --------
try:
    outcome_model = lgb.Booster(model_file=os.path.join(MODELS_DIR, "outcome_lgb.txt"))
    treatment_model = lgb.Booster(model_file=os.path.join(MODELS_DIR, "treatment_lgb.txt"))
    with open(os.path.join(MODELS_DIR, "outcome_target_encoder.pkl"), "rb") as f:
        outcome_target_encoder = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "treatment_target_encoder.pkl"), "rb") as f:
        treatment_target_encoder = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "lime_background.pkl"), "rb") as f:
        lime_background = pickle.load(f)
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

FEATURE_NAMES = [c.replace(" ", "_") for c in list(encoders['feature_names'])]
numeric_features = encoders['numeric_features']
preprocessor = encoders['preprocessor']
symptoms_mlb = encoders['symptoms_mlb']
treatment_mlb = encoders['treatment_mlb']

lime_explainer = LimeTabularExplainer(
    training_data=lime_background.values,
    feature_names=FEATURE_NAMES,
    class_names=outcome_target_encoder.classes_,
    mode='classification',
    discretize_continuous=True
)

# -------- predict wrappers --------
def predict_outcome(x_df):
    return outcome_model.predict(x_df)

def predict_treatment(x_df):
    return treatment_model.predict(x_df)

# -------- preprocess input --------
def preprocess_input_to_processed_df(input_df):
    tmp = input_df.copy()
    tmp['symptoms_list'] = tmp['symptoms'].apply(split_and_prepare)
    tmp['treatment_list'] = tmp['treatment'].apply(split_and_prepare)
    tmp['symptom_count'] = tmp['symptoms_list'].apply(len)
    tmp['treatment_count'] = tmp['treatment_list'].apply(len)
    tmp['severity_index'] = tmp['exposure_duration'] * tmp['symptom_count']

    sym_df = pd.DataFrame(symptoms_mlb.transform(tmp['symptoms_list']),
                          columns=[f"symptom_{s}" for s in symptoms_mlb.classes_])
    treat_df = pd.DataFrame(treatment_mlb.transform(tmp['treatment_list']),
                            columns=[f"treatment_{t}" for t in treatment_mlb.classes_])

    base = tmp[['age','gender','industry','chemical','exposure_duration',
                'symptom_count','treatment_count','severity_index']]
    combined = pd.concat([base.reset_index(drop=True), sym_df, treat_df], axis=1)
    processed_arr = preprocessor.transform(combined)
    return pd.DataFrame(processed_arr, columns=FEATURE_NAMES)

# -------- explanation builders --------
def build_worker_friendly_outcome_explanation(X_processed_df, outcome_idx, input_df=None):
    exp = lime_explainer.explain_instance(
        X_processed_df.iloc[0].values, predict_outcome, num_features=8, labels=[outcome_idx]
    )
    positives, negatives = [], []
    for feat_descr, weight in exp.as_list(label=outcome_idx):
        if abs(weight) < 0.01:
            continue
        mapped = humanize_feature(feat_descr, input_df)
        if not mapped:
            continue
        (positives if weight>0 else negatives).append(mapped)

    pred_label = outcome_target_encoder.inverse_transform([outcome_idx])[0]
    text = f"### Why Predicted Outcome: **{pred_label}**\n\n"
    if positives:
        text += "Factors that made recovery harder:\n" + "".join([f"- {p}\n" for p in positives])
    if negatives:
        text += "\nFactors that improved your chances:\n" + "".join([f"- {n}\n" for n in negatives])
    if not positives and not negatives:
        text += "Your health history and exposure influenced this outcome."
    return text

def build_treatment_explanation(X_processed_df, treatment_idx, input_df=None):
    exp = lime_explainer.explain_instance(
        X_processed_df.iloc[0].values, predict_treatment, num_features=8, labels=[treatment_idx]
    )
    positives, negatives = [], []
    for feat_descr, weight in exp.as_list(label=treatment_idx):
        if abs(weight) < 0.01:  
            continue
        mapped = humanize_feature(feat_descr, input_df)
        if not mapped:
            continue
        (positives if weight>0 else negatives).append(mapped)

    label = treatment_target_encoder.inverse_transform([treatment_idx])[0]
    if label == "Get Consultation":
        text = "### Suggested Treatment: **Consult a specialist / seek human consultation**\n\n"
    else:
        text = f"### Suggested Treatment: **{label}**\n\n"

    if positives:
        text += "Reasons behind this suggestion:\n" + "".join([f"- {p}\n" for p in positives])
    if negatives:
        text += "\nOther factors that reduced the need:\n" + "".join([f"- {n}\n" for n in negatives])
    if not positives and not negatives:
        text += "Your exposure and reported symptoms influenced this recommendation."
    return text

# -------- Groq-based Natural Explanation --------
def generate_groq_explanation(pred_label, raw_explanation_text, language="English"):
    """
    Uses Groq API to rewrite the explanation in clear and simple language.
    Supports English, Hindi, and Tamil.
    """
    try:
        prompt = f"""
        Rewrite the following machine learning explanation in simple, 
        everyday {language}. The goal is to make it clear and easy to understand 
        for a non-technical person.

        Prediction: {pred_label}
        Technical Explanation:
        {raw_explanation_text}

        Keep it short, conversational, and natural in {language}.
        """
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Groq explanation unavailable: {e})"

def calculate_readability_metrics(raw_text, groq_text):
    """
    Compute readability improvement using Flesch Reading Ease.
    """
    try:
        lime_score = textstat.flesch_reading_ease(raw_text)
        groq_score = textstat.flesch_reading_ease(groq_text)
        improvement = groq_score - lime_score
        return lime_score, groq_score, improvement
    except Exception as e:
        return None, None, None

# -------- frontend --------
st.set_page_config(layout="wide")
st.title("🔬 Health & Safety Predictor — Outcome + Suggested Treatment")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age (years)", 18, 100, 35, 1)
    gender = st.selectbox("Gender", sorted(list(preprocessor.named_transformers_['cat'].categories_[0])))
with col2:
    industry = st.selectbox("Industry", sorted(list(preprocessor.named_transformers_['cat'].categories_[1])))
    chemical = st.selectbox("Chemical", sorted(list(preprocessor.named_transformers_['cat'].categories_[2])))
with col3:
    exposure_duration = st.slider("Exposure Duration (hours)", 0.0, 72.0, 3.0, 0.1)

st.subheader("Symptoms (comma or semicolon separated)")
symptoms_text = st.text_area("Symptoms", "Respiratory irritation")

st.markdown("---")

# Language selector for explanations
language_choice = st.selectbox(
    "Select explanation language",
    ["English", "Hindi", "Tamil"]
)

if st.button("Predict"):
    input_df = pd.DataFrame([[age, gender, industry, chemical, exposure_duration, symptoms_text, ""]],
        columns=['age','gender','industry','chemical','exposure_duration','symptoms','treatment'])
    X_processed = preprocess_input_to_processed_df(input_df)

    outcome_probs = predict_outcome(X_processed)
    treatment_probs = predict_treatment(X_processed)

    outcome_idx = np.argmax(outcome_probs, axis=1)[0]
    treatment_idx = np.argmax(treatment_probs, axis=1)[0]

    outcome_label = outcome_target_encoder.inverse_transform([outcome_idx])[0]
    treatment_label = treatment_target_encoder.inverse_transform([treatment_idx])[0]

    st.success(f"### Predicted Outcome: **{outcome_label}**")
    if treatment_label == "Get Consultation":
        st.warning("### Suggested Treatment: **Consult a specialist / seek expert consultation**")
    else:
        st.info(f"### Suggested Treatment: **{treatment_label}**")

    st.subheader("Top-3 Suggested Treatments")
    top3_idx = np.argsort(treatment_probs, axis=1)[0][-3:][::-1]
    for i, idx in enumerate(top3_idx, start=1):
        lbl = treatment_target_encoder.inverse_transform([idx])[0]
        if lbl == "Get Consultation":
            lbl = "Consult a specialist / seek expert consultation"
        st.write(f"{i}. {lbl}")

    st.markdown("---")
    st.header("Explanation (Outcome)")
    raw_outcome_exp = build_worker_friendly_outcome_explanation(X_processed, outcome_idx, input_df=input_df)
    groq_outcome_exp = generate_groq_explanation(outcome_label, raw_outcome_exp, language_choice)
    st.markdown(generate_groq_explanation(outcome_label, raw_outcome_exp, language_choice))

    st.header("Explanation (Treatment Suggestion)")
    raw_treat_exp = build_treatment_explanation(X_processed, treatment_idx, input_df=input_df)
    groq_treat_exp = generate_groq_explanation(treatment_label, raw_treat_exp, language_choice)
    st.markdown(generate_groq_explanation(treatment_label, raw_treat_exp, language_choice))

    # --- Readability Metrics Section ---
    if language_choice == "English":  # Flesch works best in English
        st.markdown("---")
        st.subheader("📖 Readability Analysis (Flesch Reading Ease)")
        lime_score, groq_score, improvement = calculate_readability_metrics(
            raw_outcome_exp + raw_treat_exp, groq_outcome_exp + groq_treat_exp
        )

    if lime_score and groq_score:
        st.write(f"**LIME Explanation Score:** {lime_score:.2f}")
        st.write(f"**Groq Explanation Score:** {groq_score:.2f}")
        st.write(f"**Improvement:** +{improvement:.2f} points")

        if groq_score > lime_score:
            st.success("✅ Groq explanations are easier to read and understand!")
        else:
            st.warning("⚠️ Groq explanations did not significantly improve readability.")

    # Optional LIME weights visualization
    try:
        exp = lime_explainer.explain_instance(X_processed.iloc[0].values, predict_outcome,
                                             num_features=6, labels=[outcome_idx])
        items = exp.as_list(label=outcome_idx)
        labels, weights = [i[0] for i in items], [i[1] for i in items]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(range(len(weights)), weights, align='center')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([l.replace('<=','≤') for l in labels])
        ax.invert_yaxis()
        ax.set_xlabel("Influence on Prediction")
        st.pyplot(fig)
    except Exception:
        pass
