import sys
import pandas as pd
import spacy
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split


import time
import numpy as np
import openai
from openai import OpenAI
import os
from collections import Counter


mt1= "Stage 1 started!"
print("\n"+mt1)
print("-"*len(mt1))
print("\n")




# Log setup to save console output into a .txt file, used for making the report writting easier
class Tee:
    """Duplicates everything written to it into both the console and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

log_file = open("stage1_log_out.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)

nlp = spacy.load("en_core_web_sm")
df = pd.read_csv("dataset.csv")
df["Classification"] = (df["Classification"].str.lower().str.strip())
print(df["Classification"].value_counts())

# Feature extraction
def pos_ratios(text):
    doc = nlp(str(text))
    counts = {}

    for token in doc:
        counts[token.pos_] = counts.get(token.pos_, 0) + 1
    total = len(doc) if len(doc) > 0 else 1

    return {
        tag: value / total
        for tag, value in counts.items()
    }   


def ner_ratios(text):
    doc = nlp(str(text))
    counts = {}

    for ent in doc.ents:
        counts[ent.label_] = counts.get(ent.label_, 0) + 1
    total = len(doc) if len(doc) > 0 else 1

    return {
        tag: value / total
        for tag, value in counts.items()
    }

# Knowledge base
opinion_markers = [
    "i think", "i believe", "in my opinion", "maybe",
    "probably", "should", "best", "terrible",
    "excellent", "ridiculous", "unfortunately"]

fact_markers = [
    "according to", "reported", "researchers", "scientists",
    "published", "survey", "study", "confirmed",
    "officials", "percent"]


# Feature extraction
# ~ All of these prints were added to make sure the code was actually running

print("POS features extraction:...")
pos_df = pd.DataFrame(
    [pos_ratios(text) for text in df["Content"]]
).fillna(0)

print("NER feature extraction:...")
ner_df = pd.DataFrame(
    [ner_ratios(text) for text in df["Content"]]
).fillna(0)

print("Opinion keyword features extraction:...")
for phrase in opinion_markers:
    df[f"op_{phrase}"] = (
        df["Content"]
        .str.lower()
        .str.contains(phrase, regex=False)
        .astype(int)
    )

print("Fact keyword features extraction:...")
for phrase in fact_markers:
    df[f"fact_{phrase}"] = (
        df["Content"]
        .str.lower()
        .str.contains(phrase, regex=False)
        .astype(int)
    )

print("KB features have been extracted:...")
kb_cols = [
    c for c in df.columns
    if c.startswith("op_")
    or c.startswith("fact_")
]

kb_df = df[kb_cols]

x_train, x_test, y_train, y_test = train_test_split(
    df["Content"],
    df["Classification"],
    test_size=0.20,
    random_state=42,
    stratify=df["Classification"]
)

# TF-IDF setup

vec = TfidfVectorizer(max_features=5000)

tfidf_train = vec.fit_transform(x_train)
tfidf_test = vec.transform(x_test)

# Match feature rows to split

pos_train = pos_df.loc[x_train.index]
pos_test = pos_df.loc[x_test.index]

ner_train = ner_df.loc[x_train.index]
ner_test = ner_df.loc[x_test.index]

kb_train = kb_df.loc[x_train.index]
kb_test = kb_df.loc[x_test.index]


# Model helper
# Previously, we had separate blocks for each model repeating the same structure, we decided instead
# to create a helper function to run the models and print the results

results = []

best_f1 = -1
best_name = ""
best_model = None
best_features = None

def run_model(name, train_x, test_x):
    global best_f1
    global best_name
    global best_model
    global best_features

    clf = LogisticRegression(max_iter=1500)
    clf.fit(train_x, y_train)
    pred = clf.predict(test_x)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(
        y_test, pred, average="macro"
    )

    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, pred))

    results.append((name, acc, f1))

    if f1 > best_f1:
        best_f1 = f1
        best_name = name
        best_model = clf
        best_features = train_x

    return clf

# "Test 0" ~ Baseline: KB rule-based
def kb_predict(text):
    text = str(text).lower()
    op = sum(1 for w in opinion_markers if w in text)
    fact = sum(1 for w in fact_markers if w in text)
    if op > fact:
        return "opinion"
    return "fact"

kb_baseline_pred = x_test.apply(kb_predict)

kb_acc = accuracy_score(y_test, kb_baseline_pred)
kb_f1 = f1_score(y_test, kb_baseline_pred, average="macro")

print("\nKB rule-based baseline")
print(f"Accuracy: {kb_acc:.4f}")
print(f"F1 Score: {kb_f1:.4f}")
print(classification_report(y_test, kb_baseline_pred))

results.append(("KB baseline", kb_acc, kb_f1))

# Test 1: KB + TF-IDF
run_model(
    "KB + TF-IDF",
    hstack([kb_train.values, tfidf_train]),
    hstack([kb_test.values, tfidf_test])
)

# Test 2: KB + TF-IDF + POS
run_model(
    "KB + TF-IDF + POS",
    hstack([kb_train.values, tfidf_train, pos_train.values]),
    hstack([kb_test.values, tfidf_test, pos_test.values])
)

# Test 3: KB + TF-IDF + NER
run_model(
    "KB + TF-IDF + NER",
    hstack([kb_train.values, tfidf_train, ner_train.values]),
    hstack([kb_test.values, tfidf_test, ner_test.values])
)

# Test 4: KB + TF-IDF + POS + NER (All features)
all_train = hstack([
    kb_train.values,
    tfidf_train,
    pos_train.values,
    ner_train.values
])

all_test = hstack([
    kb_test.values,
    tfidf_test,
    pos_test.values,
    ner_test.values
])

run_model(
    "KB + TF-IDF + POS + NER",
    all_train,
    all_test
)

# Comparison table
print("Comparison of models")

for name, acc, f1 in results:
    print(
        f"{name:<25} "
        f"Accuracy: {acc:.4f} "
        f"F1-Score: {f1:.4f}"
    )

print("\nBest model:")
print(best_name)
print(f"Best F1-Score: {best_f1:.4f}")


###### Validation set segment ######
val_df = pd.read_csv("validationset.csv")

val_tfidf = vec.transform(
    val_df["Content"]
)

print("\nValidation set POS features extraction...")

val_pos = pd.DataFrame(
    [pos_ratios(text) for text in val_df["Content"]]
).fillna(0)

val_pos = val_pos.reindex(
    columns=pos_df.columns,
    fill_value=0
)

print("\nValidation set NER features extraction...")

val_ner = pd.DataFrame(
    [ner_ratios(text) for text in val_df["Content"]]
).fillna(0)

val_ner = val_ner.reindex(
    columns=ner_df.columns,
    fill_value=0
)

for phrase in opinion_markers:
    val_df[f"op_{phrase}"] = (
        val_df["Content"]
        .str.lower()
        .str.contains(phrase, regex=False)
        .astype(int)
    )

for phrase in fact_markers:
    val_df[f"fact_{phrase}"] = (
        val_df["Content"]
        .str.lower()
        .str.contains(phrase, regex=False)
        .astype(int)
    )

val_kb = val_df[kb_cols]

if best_name == "KB + TF-IDF":

    val_x = hstack([
        val_kb.values,
        val_tfidf
    ])

elif best_name == "KB + TF-IDF + POS":

    val_x = hstack([
        val_kb.values,
        val_tfidf,
        val_pos.values
    ])

elif best_name == "KB + TF-IDF + NER":

    val_x = hstack([
        val_kb.values,
        val_tfidf,
        val_ner.values
    ])

else:

    val_x = hstack([
        val_kb.values,
        val_tfidf,
        val_pos.values,
        val_ner.values
    ])

preds = best_model.predict(val_x)
out = pd.DataFrame()
out["Content"] = val_df["Content"]
out["Label"] = pd.Series(preds).str.capitalize()

out.to_csv(
    "group51_classifications_1.csv",
    index=False,
)

# ~Extra segment: Bar chart of methods~~~~~
# Used for the report, visual representation of which combination worked the best
methods = ["KB baseline", "KB + TF-IDF", "KB + TF-IDF + POS", "KB + TF-IDF + NER", "KB + TF-IDF + POS + NER"]
scores = [0.4462, 0.8535, 0.8654, 0.8535, 0.8713]
colors = ["steelblue"] * 5
best_idx = scores.index(max(scores))
best_label = "Best score (" + methods[best_idx] + ")"

plt.figure(figsize=(10, 5))
plt.bar(methods, scores, color=colors)
plt.axhline(y=scores[0],         color="gray",      linestyle="--", label="Baseline (KB baseline)")
plt.axhline(y=max(scores), color="steelblue", linestyle="--", label=best_label)
plt.ylim(0.4, 1.0)
plt.ylabel("F1 (macro)")
plt.title("Stage 1 — Method Comparison")
plt.xticks(rotation=15)

plt.legend()
plt.tight_layout()
plt.savefig("stage1_comparison_barchart.png", dpi=150)
plt.show()

sys.stdout = sys.__stdout__ # Closing log file
log_file.close()

print("\nScript finished: Predictions saved to \"group51_classifications_1.csv\"")
print("\n")








mt2= "Stage 2 started!"
print("\n"+mt2)
print("-"*len(mt2))
print("\n")



# Disclaimer: Two Groq API keys are required to run this script, we left two of our own keys just to facilitate your testing
# Log setup to save console output into a .txt file, used for making the report writting easier

class Tee:
    """Duplicates everything written to it into both the console and a log file."""
    def __init__(self, *streams):
        self.streams = streams
 
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
 
    def flush(self):
        for s in self.streams:
            s.flush()
 
log_file = open("stage2_log_out.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)

# ~Part 1: 1st Groq API Key (training and evaluation)~~~~~~

print(openai.__version__)
os.environ["GROQ_API_KEY"] = "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"

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY") 
)
model = "llama-3.1-8b-instant"

def ask_llm(sys_p, txt): #LLM prompting, used for all methds

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": str(txt)}
            ],
            max_tokens=20,
            temperature=0 
        )
        out = r.choices[0].message.content
        out = out.lower().strip()

        if "opinion" in out:
            return "opinion"
        if "fact" in out:
            return "fact"
        return "could not be predicted..."

    except Exception as e: # Prevents script from crashing if a network disconnection happens
        print(e)
        return "---API loading error---"

print("\n1st Groq API Key worked, Groq is ready to run~\n")

# Importing dataset, normalizing and splitting
df = pd.read_csv('dataset.csv')
df['Classification'] = df['Classification'].str.lower().str.strip()

X_train, X_test, y_train, y_test = train_test_split(
    df['Content'], df['Classification'], test_size=0.2, random_state=42, stratify=df['Classification']
)

print("\n")
print("X_train size:", len(X_train))
print("X_test  size:", len(X_test))


# 1st method (Baseline): Zero-shot learning


prompt = """
Classify the text as fact or opinion.

Fact = objective statement
Opinion = personal belief or judgement

Return only one word:
Fact
or
Opinion
"""

zs_header = "I. Baseline: Zero-Shot Learning"
print("\n" + zs_header)
print("-"*len(zs_header))

pred_zeroshot = []

for i, t in enumerate(X_test):

    x = ask_llm(prompt, str(t))
    pred_zeroshot.append(x)
    time.sleep(0.3) # Included along the code to avoid triggering "too many requests"

    if (i + 1) % 20 == 0: # Segments like this were used to keep track of the progress and to make sure the script is running
        print(i + 1, "of", len(X_test), "is finished")

zs_status = "Zero-shot classification completed"
print("-"*len(zs_status)) # We will have several of this prints along the code just to make the terminal output more readable
print(zs_status)
print("-"*len(zs_status))

# 1st method (Baseline): Zero-shot learning (evaluation)
p_clean = [] # Cleaned predictions
t_clean = [] # Cleaned true labels
u = 0 # Unknown predictions counter

for i in range(len(pred_zeroshot)):
    p = pred_zeroshot[i]
    t = y_test.iloc[i]

    if p not in ["fact", "opinion"]:
        u = u + 1
    else:
        p_clean.append(p)
        t_clean.append(t)
    

print("\nSize of Test Set:", len(pred_zeroshot))
print("Number of Unknown Predictions:", u)
print("Number of Evaluated Samples:", len(p_clean))

acc = accuracy_score(t_clean, p_clean)
f1 = f1_score(t_clean, p_clean, average="macro")

print("\n")
zs_eval_header = "Baseline: Zero-Shot Classification Evaluation"
print("\n" + zs_eval_header)
print("-"*len(zs_eval_header))
print("Accuracy: {:.2f}%".format(acc * 100))
print("f1 Score: {:.4f}".format(f1))
print(classification_report(t_clean, p_clean))

# 2nd method: Few-Shot learning
prompt_fs = """
You are an expert in text analysis. Your task is to classify the provided text as either 'fact' or 'opinion'.

Follow these classification rules:
1. 'fact': Use this label if the text contains verifiable information, statistical data, historical events, or scientific evidence.
2. 'opinion': Use this label if the text contains personal beliefs, emotions, subjective judgments, or interpretations.
3. If the text contains both, classify it based on the dominant tone.
4. If you are unsure, default to 'opinion'.

Examples:
Text: "A study published in Nature found that CO2 levels rose by 3% last year."
Label: fact

Text: "According to WHO, over 5 million people were affected by the outbreak."
Label: fact

Text: "I think the government is doing a terrible job handling this crisis."
Label: opinion

Text: "This policy is clearly the worst decision made in decades."
Label: opinion

Return ONLY the word 'fact' or 'opinion'. Do not include any punctuation or extra words.

Text: {text}
Label:
"""

fs_header= "II. Few-Shot Learning"
print("\n" + fs_header)
print("-"*len(fs_header))
pred_fewshot = []

for i, t in enumerate(X_test):

    x = ask_llm(prompt_fs, str(t))
    pred_fewshot.append(x)
    time.sleep(0.3)

    if (i + 1) % 20 == 0:
        print(i + 1, "of", len(X_test), "is finished")

fs_status = "Few-Shot classification completed"
print("-"*len(fs_status))
print(fs_status)
print("-"*len(fs_status))

# 2nd method: Few-Shot learning (evaluation)
p2_clean = []
t2_clean = []
u2 = 0

for i in range(len(pred_fewshot)):

    p = pred_fewshot[i]
    t = y_test.iloc[i]

    if p in ["fact", "opinion"]:
        p2_clean.append(p)
        t2_clean.append(t)
    else:
        u2 = u2 + 1

print("\nSize of Test Set:", len(pred_fewshot))
print("Number of Unknown Predictions:", u2)
print("Number of Evaluated Samples:", len(p2_clean))

acc2 = accuracy_score(t2_clean, p2_clean)
f12 = f1_score(t2_clean, p2_clean, average="macro")
    
fs_eval_header = "Few-Shot Classification Evaluation"
print("\n" + fs_eval_header)
print("-"*len(fs_eval_header))
print("Accuracy: {:.2f}%".format(acc2 * 100))
print("f1 Score: {:.4f}".format(f12))
print("\nClassification Report:")
print(classification_report(t2_clean, p2_clean))

# 3rd method: RAG
opinion_combinations = [
    'i think', 'i believe', 'in my opinion', 'probably', 'maybe',
    'must be', 'terrible', 'it seems', 'ridiculous', 'unfortunately',
    'simply', 'should', 'best', 'excellent', 'confusing',
]

fact_combinations = [
    'according to', 'researchers', 'found that', 'survey', 'published',
    'scientists', 'officials', 'study found', 'concluded', 'reported that',
    'percent', 'confirmed',
]

rag_header = "III. RAG"
print(rag_header)
print("-"*len(rag_header))

def get_kb(txt):
    txt = str(txt).lower()
    info = []
    for w in fact_combinations:
        if w in txt:
            info.append("fact word: " + w)
    for w in opinion_combinations:
        if w in txt:
            info.append("opinion word: " + w)
    if len(info) == 0:
        return "No keyword found."
    return ", ".join(info)

# 3rd method: RAG (prompt and classification)
prompt_kb = """
You are an expert in text analysis. Classify the following text as 'fact' or 'opinion'.

Use these guidelines:
1. If the 'Knowledge Base' contains 'fact word', it indicates objective elements.
2. If the 'Knowledge Base' contains 'opinion word', it indicates subjective judgment.
3. If both are present, determine the dominant tone.
4. If 'No keyword found', analyze the text independently based on its factual or subjective nature.

Knowledge Base:
{kb}

Text: 
{text}

Return ONLY one word:
Fact
or
Opinion
"""

pred_kb = []

for i, t in enumerate(X_test):
    kb_info = get_kb(t)
    p = prompt_kb.format(kb=kb_info, text=t)
    x = ask_llm(p, "Classify.")
    cleaned_x = str(x).lower().strip()
    
    if cleaned_x not in ['fact', 'opinion']:
        pred_kb.append("could not be predicted...")
    else:
        pred_kb.append(cleaned_x)
        
    time.sleep(0.3)

    if (i + 1) % 20 == 0:
        print(i + 1, "of", len(X_test), "is finished")

rag_status = "RAG classification completed successfully"
print("-"*len(rag_status))
print(rag_status)
print("-"*len(rag_status))

# 3rd method: RAG (evaluation)
p3_clean = []
t3_clean = []
u3 = 0

for i in range(len(pred_kb)):
    p = pred_kb[i]
    t = y_test.iloc[i]
    if p not in ["fact", "opinion"]:
        u3 = u3 + 1
    else:
        p3_clean.append(p)
        t3_clean.append(t)

acc3 = accuracy_score(t3_clean, p3_clean)
f13 = f1_score(t3_clean, p3_clean, average="macro")

print("\nSize of Test Set:", len(pred_kb))
print("Number of Unknown Predictions:", u3)
print("Number of Evaluated Samples:", len(p3_clean))

if len(p3_clean) == 0: # Ensures the script doesnt crash if array is empty
    print("\n[!] Warning: No valid predictions were received from the RAG model")
else:
    rag_eval_header = "RAG Classification Evaluation"
    print("\n" + rag_eval_header)
    print("-"*len(rag_eval_header))
    print("Accuracy: {:.2f}%".format(acc3 * 100))
    print("f1 Score: {:.4f}".format(f13))
    print("\nClassification Report:")
    print(classification_report(t3_clean, p3_clean))

# 4th method: RAG + Few-Shot Learning
prompt_mix = """
You are an expert in text analysis. Classify the following text as 'fact' or 'opinion'.

Follow these logic rules:
1. Examine the 'Knowledge Base' keywords: These provide hints about factual or subjective content.
2. Consider the provided examples: These show the pattern of classification.
3. If the text is supported by evidence/data, label as 'fact'.
4. If the text expresses personal belief, emotion, or judgment, label as 'opinion'.
5. If unsure, prioritize 'opinion'.

Knowledge Base hints: {kb}

Examples:
- Text: "A study published in Nature found that CO2 levels rose by 3% last year." -> fact
- Text: "According to WHO, over 5 million people were affected by the outbreak." -> fact
- Text: "I think the government is doing a terrible job handling this crisis." -> opinion
- Text: "This policy is clearly the worst decision made in decades." -> opinion

Text to classify: "{text}"

Return ONLY the word 'fact' or 'opinion'.
Label:
"""
rag_fs_header = "IV. RAG + Few-Shot Learning"
print("\n" + rag_fs_header)
print("-"*len(rag_fs_header))
pred_mix = []

for i, t in enumerate(X_test):
    kb_info = get_kb(t)
    p = prompt_mix.format(kb=kb_info, text=t)
    x = ask_llm(p, str(t))
    cleaned_x = str(x).lower().strip()
    
    if "fact" in cleaned_x:
        pred_mix.append("fact")
    elif "opinion" in cleaned_x:
        pred_mix.append("opinion")
    else:
        pred_mix.append("could not be predicted...")
        
    time.sleep(0.3)

    if (i + 1) % 20 == 0:
        print(i + 1, "of", len(X_test), "is finished")

rag_fs_status = "RAG + Few-Shot classification completed successfully"
print("-"*len(rag_fs_status))
print(rag_fs_status)
print("-"*len(rag_fs_status))

# 4th method: RAG + Few-Shot Learning (evaluation)
p4_clean = []
t4_clean = []
u4 = 0

for i in range(len(pred_mix)):
    p = pred_mix[i]
    t = y_test.iloc[i]

    if p not in ["fact", "opinion"]:
        u4 = u4 + 1
    else:
        p4_clean.append(p)
        t4_clean.append(t)

acc4 = accuracy_score(t4_clean, p4_clean)
f14 = f1_score(t4_clean, p4_clean, average="macro")

print("\nSize of Test Set:", len(pred_mix))
print("Number of Unknown Predictions:", u4)
print("Number of Evaluated Samples:", len(p4_clean))

mix_eval_header = "RAG + Few-Shot Classification Evaluation"
print("\n" + mix_eval_header)
print("-"*len(mix_eval_header))
print("Accuracy: {:.2f}%".format(acc4 * 100))
print("f1 Score: {:.4f}".format(f14))
print("\nClassification Report:")
print(classification_report(t4_clean, p4_clean))

# 5th method: Self-Consistency Learning
sc_header = "V. Self-Consistency Learning"
print("\n" + sc_header)
print("-"*len(sc_header))

def ask_llm_selfconsistency(sys_p, txt, runs=3):
    votes = []
    for _ in range(runs):
        result = ask_llm(sys_p, txt)
        if result in ["fact", "opinion"]:
            votes.append(result)
        time.sleep(0.3)
    
    if not votes:
        return "unknown"
    
    return Counter(votes).most_common(1)[0][0]

pred_sc = []

for i, t in enumerate(X_test):
    x = ask_llm_selfconsistency(prompt, str(t), runs=3)
    pred_sc.append(x)

    if (i + 1) % 20 == 0:
        print(i + 1, "of", len(X_test), "is finished")
        pd.Series(pred_sc).to_csv("sc_checkpoint.csv", index=False) # Since we were running into token issues, we decided to add a checkpoints here and there for safety 

sc_status = "Self-Consistency classification completed successfully"
print("-"*len(sc_status))
print(sc_status)
print("-"*len(sc_status))

# 5th method: Self-Consistency Learning (evaluation)
p5_clean = []
t5_clean = []
u5 = 0

for i in range(len(pred_sc)):
    p = pred_sc[i]
    t = y_test.iloc[i]
    if p in ["fact", "opinion"]:
        p5_clean.append(p)
        t5_clean.append(t)
    else:
        u5 = u5 + 1

acc5 = accuracy_score(t5_clean, p5_clean)
f15 = f1_score(t5_clean, p5_clean, average="macro")
print("\nSize of Test Set:", len(pred_sc))
print("Number of Unknown Predictions:", u5)
print("Number of Evaluated Samples:", len(p5_clean))

sc_eval_header = "Self-Consistency (SC) Classification Evaluation"
print("\n" + sc_eval_header)
print("-"*len(sc_eval_header))
print("Accuracy: {:.2f}%".format(acc5 * 100))
print("f1 Score: {:.4f}".format(f15))
print("\nClassification Report:")
print(classification_report(t5_clean, p5_clean))

# 6th method: RAG + Few-Shot + Self-Consistency Learning
rag_fs_sc_header = "VI. RAG + Few-Shot + Self-Consistency Learning"
print("\n" + rag_fs_sc_header)
print("-"*len(rag_fs_sc_header))

def ask_llm_selfconsistency_rag(txt, runs=3):
    votes = []
    kb = get_kb(txt)
    p = prompt_mix.format(kb=kb, text=txt)
    
    for _ in range(runs):
        result = ask_llm(p, "Classify.")
        if result in ["fact", "opinion"]: 
            votes.append(result)
        time.sleep(0.3)
    
    if not votes:
        return "unknown"
    
    return Counter(votes).most_common(1)[0][0]

pred_rag_fs_sc = []

for i, t in enumerate(X_test):
    x = ask_llm_selfconsistency_rag(t, runs=3)
    pred_rag_fs_sc.append(x)

    if (i + 1) % 20 == 0:
        print(i + 1, "of", len(X_test), "is finished")
        pd.Series(pred_rag_fs_sc).to_csv("rag_fs_sc_checkpoint.csv", index=False)

rag_fs_sc_status = "RAG + Few-Shot + Self-Consistency classification completed successfully"
print("-"*len(rag_fs_sc_status))
print(rag_fs_sc_status)
print("-"*len(rag_fs_sc_status))

# 6th method: RAG + Few-Shot + Self-Consistency Learning (evaluation)
p6_clean = []
t6_clean = []
u6 = 0

for i in range(len(pred_rag_fs_sc)):
    p = pred_rag_fs_sc[i]
    t = y_test.iloc[i]
    if p not in ["fact", "opinion"]:
        u6 = u6 + 1
    else:
        p6_clean.append(p)
        t6_clean.append(t)
        
acc6 = accuracy_score(t6_clean, p6_clean)
f16 = f1_score(t6_clean, p6_clean, average="macro")
print("\nSize of Test Set:", len(pred_rag_fs_sc))
print("Number of Unknown Predictions:", u6)
print("Number of Evaluated Samples:", len(p6_clean))

rag_fs_sc_eval_header = "RAG + Few-Shot + Self-Consistency Classification Evaluation"
print("\n" + rag_fs_sc_eval_header)
print("-"*len(rag_fs_sc_eval_header))
print("Accuracy: {:.2f}%".format(acc6 * 100))
print("f1 Score: {:.4f}".format(f16))
print("\nClassification Report:")
print(classification_report(t6_clean, p6_clean))


# ~Part 2: 2nd Groq API Key (validation set)~~~~~~

os.environ["GROQ_API_KEY"] = "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" 

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY") 
)
model = "llama-3.1-8b-instant"

def ask_llm(sys_p, txt):

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": str(txt)}
            ],
            max_tokens=20,
            temperature=0 
        )
        out = r.choices[0].message.content
        out = out.lower().strip()

        if "opinion" in out:
            return "opinion"
        if "fact" in out:
            return "fact"
        return "could not be predicted..."

    except Exception as e:
        print(e)
        return "API loading error..."

print("\n2nd Groq API Key worked, Groq is ready to continue\n")

# ~Final segment: Summary of all methods and their evaluation~~~~~~
results = {
    "Zero-Shot Learning":        (pred_zeroshot, f1),
    "Few-Shot Learning":         (pred_fewshot,  f12),
    "RAG":     (pred_kb,       f13),
    "Self-Consistency Learning": (pred_sc,       f15),
    "RAG + Few-Shot + Self-Consistency Learning":  (pred_rag_fs_sc,   f16),
    "RAG + Few-Shot Learning":     (pred_mix,      f14),
}

rows = []
for name, (preds, score) in results.items():
    p_c = [p for p in preds if p in ["fact", "opinion"]]
    t_c = [y_test.iloc[i] for i, p in enumerate(preds) if preds[i] in ["fact", "opinion"]]
    acc = accuracy_score(t_c, p_c)
    rows.append({"Method": name, "Accuracy": acc, "F1 (macro)": score})

summary_df = pd.DataFrame(rows)
best_name  = max(results, key=lambda k: results[k][1])
best_preds = results[best_name][0]

# ~Extra segment: Bar chart of methods~~~~~~
# Used for the report, visual represantation of which combination worked the best
methods = ["ZS", "FS", "RAG", "RAG + FS", "SC", "RAG + FS + SC"]
scores  = [f1, f12, f13, f14, f15, f16]
colors  = ["steelblue"] * 6
best_idx   = scores.index(max(scores))
best_label = "Best score (" + methods[best_idx] + ")"

plt.figure(figsize=(10, 5))
plt.bar(methods, scores, color=colors)
plt.axhline(y=f1,          color="gray",      linestyle="--", label="Baseline (Zero-shot)")
plt.axhline(y=max(scores), color="steelblue", linestyle="--", label=best_label)
plt.ylim(0.65, 1.0)
plt.ylabel("F1 (macro)")
plt.title("Stage 2 — Method Comparison")
plt.xticks(rotation=15)
plt.figtext(0.5, -0.05, "ZS = Zero-shot, FS = Few-shot, SC = Self-Consistency",
            ha="center", fontsize=9, style="italic")
plt.legend()
plt.tight_layout()
plt.savefig("stage2_comparison_barchart.png", dpi=150)
plt.show()

# Final results: validation set predictions
val_header = "Validation prediction generation start:"
print("-"*len(val_header))
print(val_header)
print("-"*len(val_header))
df_val = pd.read_csv('validationset.csv')
X_val = df_val['Content']

# Automatically choose best method
best_name = max(results, key=lambda k: results[k][1])
print("Best method:", best_name, "| F1:", round(results[best_name][1], 3))

# "Strategy pattern": Maps best performing method to its function
def predict_best(t):
    if best_name == "RAG + Few-Shot + Self-Consistency Learning":
        return ask_llm_selfconsistency_rag(t, runs=3)
    elif best_name == "RAG + Few-Shot Learning":
        kb = get_kb(t)
        p = prompt_mix.format(kb=kb, text=t)
        return ask_llm(p, "Classify.")
    elif best_name == "Self-Consistency Learning":
        return ask_llm_selfconsistency(prompt, t, runs=3)
    elif best_name == "Few-Shot Learning":
        return ask_llm(prompt_fs, str(t))
    elif best_name == "RAG":
        kb = get_kb(t)
        p = prompt_kb.format(kb=kb, text=t)
        return ask_llm(p, "Classify.")
    else:  # Zero-shot
        return ask_llm(prompt, str(t))

pred_val = []
for i, t in enumerate(X_val):
    x = predict_best(t)
    pred_val.append(x)
    time.sleep(0.3)
    if (i + 1) % 20 == 0:
        print(i + 1, "of", len(X_val), "is finished")

print("Finalized predictions for validation set, preparing CSV output...")

# Output CSV segment
output_final = pd.DataFrame()
output_final["Content"]  = X_val.values
output_final["Classification"] = pred_val
output_final["Classification"] = output_final["Classification"].replace("could not be predicted", "Fact") # Default to "Fact" if the model couldnt predict
output_final["Classification"] = output_final["Classification"].replace("unknown", "Fact") # Default to "Fact" if the model returns an "Unknown" label
output_final["Classification"] = output_final["Classification"].str.capitalize() # Capitalization of the first letter to match "Fact" and "Opinion" labels
output_final.to_csv("group51_classifications_2.csv", index=False)

summary_model = "Model used on validation dataset: " + best_name
summary_file = "CSV file created: group51_classifications_2.csv"
summary_count = "Predictions : " + str(len(output_final))
summary_dist = "Distribution:\n" + str(output_final['Classification'].value_counts())

opt_length = {"summary_model": summary_model, "summary_file": summary_file, "summary_count": summary_count, "summary_dist": summary_dist}
l_opt_out = max(opt_length, key=lambda k: len(opt_length[k]))
l_val = len(opt_length[l_opt_out])

print("-"*l_val)
print("\n" + summary_model)
print(summary_file)
print(summary_count)
print(summary_dist)
print("-"*l_val)

sys.stdout = sys.__stdout__ # Stopping the logging
log_file.close()