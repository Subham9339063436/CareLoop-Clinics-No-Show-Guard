import nbformat as nbf
from pathlib import Path
ROOT=Path('.')

def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)

# PART 1 notebook
nb=nbf.v4.new_notebook(); cells=[]
cells.append(md("""# Part 1 — Python & Data Foundations

**Tasks 1.1–1.6.** This notebook reads the original `data/appointments.csv` twice: first with the `csv` module for type-conversion practice, then independently with pandas for the cleaning pipeline. It generates the required hand-off files for Parts 2 and 3.

Before running, keep the original files unchanged in `data/`: `appointments.csv`, `clinics.csv`, and `followup_notes.csv`."""))
cells.append(code("""from pathlib import Path
import csv
import json
import numpy as np
import pandas as pd
import requests
import time

ROOT = Path.cwd()
DATA_DIR = ROOT / 'data'
APPOINTMENTS_FILE = DATA_DIR / 'appointments.csv'
CLINICS_FILE = DATA_DIR / 'clinics.csv'
assert APPOINTMENTS_FILE.exists(), 'Put the original appointments.csv in data/.'
assert CLINICS_FILE.exists(), 'Put the original clinics.csv in data/.'
"""))
cells.append(md("## Task 1.1 — Read with the `csv` module and reject failed conversions"""))
cells.append(code("""with open(APPOINTMENTS_FILE, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames
    raw_rows = list(reader)
print('Header:', header)
print('Row count:', len(raw_rows))

def parse_row(row):
    converted = dict(row)
    converted['age'] = int(converted['age'])
    converted['fee_inr'] = int(converted['fee_inr'])
    converted['prior_no_shows'] = int(converted['prior_no_shows'])
    converted['distance_km'] = float(converted['distance_km'])
    return converted

accepted, rejected = [], []
for row in raw_rows:
    try:
        accepted.append(parse_row(row))
    except ValueError:
        rejected.append(row)
print('Accepted:', len(accepted), 'Rejected:', len(rejected))
with open(ROOT / 'rejected_rows.json', 'w', encoding='utf-8') as f:
    json.dump(rejected, f, indent=2)
"""))
cells.append(md("## Task 1.2 — Inspect and clean, pass 1 (restart from original CSV)"""))
cells.append(code("""df = pd.read_csv(APPOINTMENTS_FILE)
print('Raw shape:', df.shape)
df.info()
print('Missing values by column:\n', df.isnull().sum())
print('Exact duplicate rows:', df.duplicated().sum())

df = df.drop_duplicates()
print('Rows after exact duplicate removal:', len(df))
df['booking_channel'] = df['booking_channel'].str.strip().str.title()
print('Rows after booking_channel cleaning:', len(df))
print('Booking-channel values:', sorted(df['booking_channel'].dropna().unique()))
df['appt_date'] = pd.to_datetime(df['appt_date'], errors='coerce')
df['booked_on'] = pd.to_datetime(df['booked_on'])
print('Rows after date conversion:', len(df))
df = df.dropna(subset=['appt_date', 'status'])
print('Rows after dropping missing appt_date/status:', len(df))
"""))
cells.append(md("## Task 1.3 — Clean, pass 2: validity, gaps and outliers"""))
cells.append(code("""invalid_age = (df['age'] < 1) | (df['age'] > 100)
print('Invalid ages:', int(invalid_age.sum()))
df.loc[invalid_age, 'age'] = np.nan
age_median = df['age'].median()
distance_median = df['distance_km'].median()
df['age'] = df['age'].fillna(age_median).round().astype(int)
df['distance_km'] = df['distance_km'].fillna(distance_median)
print('Age median used:', age_median)
print('Distance median used:', distance_median)

cap_counts = {}
for col in ['distance_km', 'fee_inr']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    before = df[col].copy()
    df[col] = np.clip(df[col], lower_fence, upper_fence)
    cap_counts[col] = int((before != df[col]).sum())
    print(f'{col}: Q1={q1}, Q3={q3}, IQR={iqr}, lower={lower_fence}, upper={upper_fence}, capped={cap_counts[col]}')

df['distance_km'] = df['distance_km'].round(1)
df['fee_inr'] = df['fee_inr'].round().astype(int)
"""))
cells.append(md("## Task 1.4 — Features and NumPy summary"""))
cells.append(code("""df['lead_days'] = (df['appt_date'] - df['booked_on']).dt.days
df['no_show'] = (df['status'] == 'No-Show').astype(int)
X = df[['age', 'distance_km', 'fee_inr', 'lead_days']].to_numpy()
column_means = np.mean(X, axis=0)
print('Column means [age, distance_km, fee_inr, lead_days]:', column_means)

long_lead_mask = df['lead_days'] > 14
print('lead_days > 14 count:', int(long_lead_mask.sum()))
print('lead_days > 14 share:', float(long_lead_mask.mean()))

distance = df['distance_km'].to_numpy()
distance_z = (distance - np.mean(distance)) / np.std(distance)
print('Rows with |distance z-score| > 2:', int((np.abs(distance_z) > 2).sum()))
print('No-shows:', int(df['no_show'].sum()), 'of', len(df), '(', round(100 * df['no_show'].mean(), 1), '% )')
"""))
cells.append(md("## Task 1.5 — Enrich clinics from Open-Meteo\n\nThe first hourly temperature is the midnight-UTC stand-in specified in the brief. Each clinic is attempted twice. If both attempts fail, the temperature remains missing and is written as SQL `NULL` later; no temperature is imputed."""))
cells.append(code("""clinics = pd.read_csv(CLINICS_FILE)
temp_values = []
for _, clinic in clinics.iterrows():
    temp_c = None
    failure_reason = None
    params = {
        'latitude': clinic['latitude'],
        'longitude': clinic['longitude'],
        'hourly': 'temperature_2m',
        'forecast_days': 1
    }
    for attempt in range(2):
        try:
            response = requests.get(
                'https://api.open-meteo.com/v1/forecast',
                params=params,
                timeout=10
            )
            if response.status_code != 200:
                failure_reason = f'HTTP status {response.status_code}'
                continue
            payload = response.json()
            hourly_values = payload.get('hourly', {}).get('temperature_2m')
            if not hourly_values:
                failure_reason = 'hourly.temperature_2m missing or empty'
                continue
            temp_c = float(hourly_values[0])
            failure_reason = None
            break
        except requests.exceptions.RequestException as exc:
            failure_reason = f'RequestException: {exc}'
        except (ValueError, TypeError, KeyError) as exc:
            failure_reason = f'Invalid API response: {exc}'
    if temp_c is None:
        print(f"{clinic['clinic_name']}: {failure_reason}; temp_c will be NULL")
    temp_values.append(temp_c)

clinics['temp_c'] = temp_values
clinics.to_csv(ROOT / 'clinics_enriched.csv', index=False)
print('Clinic rows written:', len(clinics))
"""))
cells.append(md("## Task 1.6 — Export the clean table and write SQL load statements"""))
cells.append(code("""columns = [
    'appointment_id', 'patient_id', 'clinic_id', 'booked_on', 'appt_date',
    'age', 'gender', 'distance_km', 'fee_inr', 'booking_channel', 'visit_type',
    'reminder_sent', 'prior_no_shows', 'status', 'lead_days', 'no_show'
]
df = df.sort_values('appointment_id').copy()
for date_col in ['booked_on', 'appt_date']:
    df[date_col] = df[date_col].dt.strftime('%Y-%m-%d')
appointments_clean = df[columns].copy()
print('appointment_id unique:', appointments_clean['appointment_id'].is_unique)
print('Missing values in the 16 export columns:\n', appointments_clean.isnull().sum())
assert appointments_clean['appointment_id'].is_unique
assert not appointments_clean.isnull().any().any()
appointments_clean.to_csv(ROOT / 'appointments_clean.csv', index=False)

def sql_value(value, quote=False):
    if pd.isna(value):
        return 'NULL'
    if quote:
        return "'" + str(value).replace("'", "''") + "'"
    return str(value)

clinic_text_columns = {'clinic_id', 'clinic_name', 'city', 'tier'}
appointment_text_columns = {
    'appointment_id', 'patient_id', 'clinic_id', 'booked_on', 'appt_date',
    'gender', 'booking_channel', 'visit_type', 'reminder_sent', 'status'
}
with open(ROOT / 'load_data.sql', 'w', encoding='utf-8') as sql_file:
    sql_file.write('-- Generated from clinics_enriched.csv and appointments_clean.csv.\n')
    sql_file.write('-- Create tables using 02_queries.sql before running this file.\n\n')
    for _, row in clinics.iterrows():
        values = [sql_value(row[col], col in clinic_text_columns) for col in [
            'clinic_id', 'clinic_name', 'city', 'tier', 'latitude', 'longitude', 'temp_c'
        ]]
        sql_file.write('INSERT INTO clinics VALUES (' + ','.join(values) + ');\n')
    sql_file.write('\n')
    for _, row in appointments_clean.iterrows():
        values = [sql_value(row[col], col in appointment_text_columns) for col in columns]
        sql_file.write('INSERT INTO appointments VALUES (' + ','.join(values) + ');\n')

print('appointments_clean.csv shape:', appointments_clean.shape)
print('clinics_enriched.csv shape:', clinics.shape)
print('SQL INSERT statements:', len(clinics) + len(appointments_clean))
"""))
cells.append(md("""## Part 1 hand-off and validation notes

The exported `appointments_clean.csv` is the single input for Parts 2 and 3. `clinics_enriched.csv` and `load_data.sql` are the database hand-off. If Open-Meteo cannot be reached after the retry, its `temp_c` fields are intentionally blank and become unquoted `NULL` in SQL."""))
nb.cells=cells
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}}
nbf.write(nb, ROOT/'01_ingest_clean.ipynb')

# PART 3 notebook — separate task cells
nb=nbf.v4.new_notebook(); cells=[]
cells.append(md("""# Part 3 — Machine Learning: Predict No-Shows

Tasks 3.1–3.5. The prediction score is produced on the evening before the appointment, after reminders have been sent or skipped. The test set is for final reporting only: model and threshold selection use training-only cross-validation.

Load `appointments_clean.csv` without shuffling or re-sorting its rows."""))
cells.append(code("""import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path.cwd()
df = pd.read_csv(ROOT / 'appointments_clean.csv')
assert df.shape == (990, 16), 'Run Part 1 and verify its 990-row clean output first.'
appointment_ids = df['appointment_id'].copy()
"""))
cells.append(md("## Task 3.1 — Features, split and baseline"""))
cells.append(code("""X = df[[
    'age', 'distance_km', 'fee_inr', 'lead_days', 'prior_no_shows',
    'reminder_sent', 'booking_channel', 'visit_type'
]].copy()
X['reminder_sent'] = X['reminder_sent'].map({'Yes': 1, 'No': 0})
X = pd.get_dummies(X, columns=['booking_channel', 'visit_type'])
y = df['no_show'].copy()
assert X.shape[1] == 11, f'Expected 11 features, received {X.shape[1]}'

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
test_appointment_ids = appointment_ids.loc[X_test.index]
baseline_accuracy = (y_test == 0).mean()
print('Features:', X.shape[1], list(X.columns))
print('Training rows:', len(X_train), 'Test rows:', len(X_test))
print('No-shows in test set:', int(y_test.sum()))
print('Always-Attended baseline accuracy:', baseline_accuracy)
"""))
cells.append(md("## Task 3.2 — Logistic regression"""))
cells.append(code("""scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logistic = LogisticRegression(max_iter=1000)
logistic.fit(X_train_scaled, y_train)
logistic_test_prob = logistic.predict_proba(X_test_scaled)[:, 1]
logistic_test_pred = logistic.predict(X_test_scaled)

logistic_metrics = {
    'accuracy': accuracy_score(y_test, logistic_test_pred),
    'precision': precision_score(y_test, logistic_test_pred, zero_division=0),
    'recall': recall_score(y_test, logistic_test_pred, zero_division=0),
    'f1': f1_score(y_test, logistic_test_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, logistic_test_prob),
    'confusion_matrix': confusion_matrix(y_test, logistic_test_pred).tolist()
}
print('Logistic regression test metrics:', logistic_metrics)
print('Coefficients in feature order:')
for feature, coefficient in zip(X.columns, logistic.coef_[0]):
    print(feature, coefficient)

reminder_index = list(X.columns).index('reminder_sent')
reminder_coefficient = logistic.coef_[0][reminder_index]
reminder_sd = scaler.scale_[reminder_index]
print('exp(scaled reminder coefficient):', np.exp(reminder_coefficient))
print('Reminder feature standard deviation:', reminder_sd)
print('exp(coefficient / feature SD), No-to-Yes effect:', np.exp(reminder_coefficient / reminder_sd))
print('Interpretation: in this model, holding other features constant, reminder_sent=Yes is associated with lower no-show odds; this observational association does not prove reminders cause the reduction.')
"""))
cells.append(md("## Task 3.3 — Decision trees and overfitting"""))
cells.append(code("""full_tree = DecisionTreeClassifier(random_state=42)
full_tree.fit(X_train, y_train)
depth3_tree = DecisionTreeClassifier(random_state=42, max_depth=3)
depth3_tree.fit(X_train, y_train)
for label, model in [('Full-depth tree', full_tree), ('Depth-3 tree', depth3_tree)]:
    print(label,
          'depth=', model.get_depth(),
          'training accuracy=', accuracy_score(y_train, model.predict(X_train)),
          'test accuracy=', accuracy_score(y_test, model.predict(X_test)))
print('A large train/test gap in the full-depth tree indicates overfitting; the depth-3 limit reduces memorisation.')
"""))
cells.append(md("## Task 3.4 — Random forest and model comparison"""))
cells.append(code("""random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)
rf_test_prob = random_forest.predict_proba(X_test)[:, 1]
rf_test_pred = random_forest.predict(X_test)
rf_metrics = {
    'accuracy': accuracy_score(y_test, rf_test_pred),
    'recall': recall_score(y_test, rf_test_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, rf_test_prob),
    'training_accuracy': accuracy_score(y_train, random_forest.predict(X_train))
}
print('Random forest metrics:', rf_metrics)
feature_importance = pd.Series(random_forest.feature_importances_, index=X.columns).sort_values(ascending=False)
print('Top three feature importances:\n', feature_importance.head(3))

comparison = pd.DataFrame([
    {'model': 'Logistic regression', 'accuracy': logistic_metrics['accuracy'], 'recall': logistic_metrics['recall'], 'roc_auc': logistic_metrics['roc_auc']},
    {'model': 'Decision tree (depth 3)', 'accuracy': accuracy_score(y_test, depth3_tree.predict(X_test)), 'recall': recall_score(y_test, depth3_tree.predict(X_test), zero_division=0), 'roc_auc': roc_auc_score(y_test, depth3_tree.predict_proba(X_test)[:, 1])},
    {'model': 'Random forest (100 trees)', 'accuracy': rf_metrics['accuracy'], 'recall': rf_metrics['recall'], 'roc_auc': rf_metrics['roc_auc']}
])
print(comparison.to_string(index=False))
"""))
cells.append(md("## Task 3.5 — Training-only CV selection, OOF threshold, one test evaluation"""))
cells.append(code("""# (a) Tune the random forest using training data only.
param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5, 8]}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42), param_grid,
    cv=5, scoring='roc_auc'
)
rf_grid.fit(X_train, y_train)
print('Tuned forest best parameters:', rf_grid.best_params_)
print('Tuned forest best CV ROC-AUC:', rf_grid.best_score_)

# (b) Compare all candidate models using training-only cross-validation.
# The LR pipeline refits StandardScaler inside each CV fold.
logistic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])
cv_candidates = {
    'Logistic regression': logistic_pipeline,
    'Decision tree depth 3': DecisionTreeClassifier(random_state=42, max_depth=3),
    'Random forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Tuned random forest': rf_grid.best_estimator_
}
cv_auc = {
    name: cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    for name, model in cv_candidates.items()
}
print('Training CV ROC-AUC:', cv_auc)
winning_name = max(cv_auc, key=cv_auc.get)
winning_model = cv_candidates[winning_name]
print('Selected model (highest training CV ROC-AUC):', winning_name)

# (c) Threshold choice is based only on out-of-fold training probabilities.
oof_prob = cross_val_predict(
    winning_model, X_train, y_train, cv=5, method='predict_proba'
)[:, 1]
threshold_results = []
for threshold in [0.5, 0.35]:
    flags = (oof_prob >= threshold).astype(int)
    threshold_results.append({
        'threshold': threshold,
        'precision': precision_score(y_train, flags, zero_division=0),
        'recall': recall_score(y_train, flags, zero_division=0),
        'f1': f1_score(y_train, flags, zero_division=0),
        'flagged': int(flags.sum())
    })
threshold_table = pd.DataFrame(threshold_results)
print(threshold_table.to_string(index=False))
chosen_threshold = float(threshold_table.sort_values('f1', ascending=False).iloc[0]['threshold'])
print('Chosen threshold (higher training OOF F1):', chosen_threshold)
print('Recall matters because a missed warning may leave a doctor slot empty, while an extra follow-up call costs staff time.')

# (d) Fit the winner on all training rows and evaluate the test set once.
winning_model.fit(X_train, y_train)
test_risk_prob = winning_model.predict_proba(X_test)[:, 1]
test_flag = (test_risk_prob >= chosen_threshold).astype(int)
final_test_metrics = {
    'precision': precision_score(y_test, test_flag, zero_division=0),
    'recall': recall_score(y_test, test_flag, zero_division=0),
    'flagged': int(test_flag.sum()),
    'roc_auc': roc_auc_score(y_test, test_risk_prob)
}
print('Final held-out test metrics:', final_test_metrics)
print('Clinic threshold should be set using the number of follow-up calls the front desk can make in an evening against the cost of an empty slot.')

# (e) Export one risk score for every held-out test row.
test_risk_scores = pd.DataFrame({
    'appointment_id': test_appointment_ids.values,
    'no_show': y_test.values,
    'risk_prob': test_risk_prob,
    'flag': test_flag
})
test_risk_scores.to_csv(ROOT / 'test_risk_scores.csv', index=False)
print('Saved test_risk_scores.csv:', test_risk_scores.shape,
      '| flagged:', int(test_risk_scores['flag'].sum()))
"""))
cells.append(md("""## Required limits to report in the final recommendation

The medians, quartiles and outlier fences in Part 1 were computed on all 990 rows before splitting, so the test rows had a small indirect influence on cleaning. A patient can have multiple bookings, so the same patient may appear in both train and test. The random split also does not simulate a real past-to-future deployment."""))
nb.cells=cells
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}}
nbf.write(nb,ROOT/'03_model.ipynb')

# PART 4 notebook
nb=nbf.v4.new_notebook(); cells=[]
cells.append(md("""# Part 4 — GenAI: Structured Follow-up Note Analysis

Tasks 4.1–4.4. This notebook requires a free Gemini API key entered at runtime. **Never paste the key into a code cell, notebook text, or GitHub.** Clear notebook outputs before committing. The first 10 call-centre notes are analysed; blank note N004 is skipped without an API call."""))
cells.append(code("""!pip install -q google-genai pydantic

import json
import time
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai import types

ROOT = Path.cwd()
NOTES_FILE = ROOT / 'data' / 'followup_notes.csv'
APPOINTMENTS_FILE = ROOT / 'appointments_clean.csv'
MODEL_NAME = 'gemini-3.8-flash'  # Current Flash model; confirm the name is available to your AI Studio key.
API_KEY = input('Paste your Gemini API key (runtime only): ')
client = genai.Client(api_key=API_KEY)
API_CALLS = 0
ALLOWED_ROOT_CAUSES = [
    'forgot_or_no_reminder', 'distance_or_transport', 'cost_or_fee',
    'illness_or_emergency', 'work_or_schedule', 'other'
]

SCHEMA = {
    'type': 'object',
    'properties': {
        'root_cause': {'type': 'string'},
        'confidence': {'type': 'number'},
        'summary': {'type': 'string'},
        'recommended_action': {'type': 'string'}
    },
    'required': ['root_cause', 'confidence', 'summary', 'recommended_action']
}

SYSTEM_TEXT = (
    'You are a careful outpatient-clinic call-centre note analyst. '
    'Classify only what the note supports. Do not infer medical details or invent facts. '
    'Use other when the cause is ambiguous or does not fit a listed category.'
)
"""))
cells.append(md("## Task 4.1 — One validated call: N001\n\nThe CRAFT prompt supplies the context, role, action, tone and required output format. The schema keeps `root_cause` as a plain string; the allowed-value rule is enforced afterward by Pydantic/application validation."""))
cells.append(code("""def build_craft_prompt(note_text):
    return f\"\"\"Context: CareLoop Clinics is reviewing short call-centre notes recorded after a patient missed an outpatient appointment.
Role: You are a careful clinical-administration data analyst, not a clinician.
Action: Classify the main reason supported by the note. Choose exactly one root_cause from this list: {', '.join(ALLOWED_ROOT_CAUSES)}. Use 'other' rather than guessing when the note is unclear or does not support a listed reason. Keep summary to at most 25 words. Recommend a practical administrative follow-up, not medical advice.
Tone: Neutral, concise, factual, non-judgmental.
Format: Return only the required JSON object with exactly these fields: root_cause (string), confidence (number from 0 to 1), summary (string, at most 25 words), recommended_action (string).

Note to classify:\n{note_text}\"\"\"


def generate_json(prompt):
    global API_CALLS
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_TEXT,
        temperature=0,
        max_output_tokens=1000,
        response_mime_type='application/json',
        response_schema=SCHEMA,
        thinking_config=types.ThinkingConfig(thinking_level='minimal')
    )
    API_CALLS += 1
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=config
    )
    return response.text or ''


def analyse_note(note_text):
    raw_text = generate_json(build_craft_prompt(note_text))
    return json.loads(raw_text)

n001 = pd.read_csv(NOTES_FILE).loc[lambda d: d['note_id'].eq('N001'), 'note_text'].iloc[0]
parsed_n001 = analyse_note(n001)
print(parsed_n001)
assert set(parsed_n001) == {'root_cause', 'confidence', 'summary', 'recommended_action'}
"""))
cells.append(md("## Task 4.2 — Pydantic validation and process first 10 notes"""))
cells.append(code("""class FollowupInsight(BaseModel):
    root_cause: str = Field(min_length=3)
    confidence: float = Field(ge=0, le=1)
    summary: str = Field(min_length=5)
    recommended_action: str = Field(min_length=5)


def validate_allowed_root_cause(data):
    if data.root_cause not in ALLOWED_ROOT_CAUSES:
        raise ValueError(f'root_cause must be one of {ALLOWED_ROOT_CAUSES}; received {data.root_cause!r}')
    return data

notes_df = pd.read_csv(NOTES_FILE).head(10)
analysed_records = []
for row_number, row in notes_df.iterrows():
    note_text = row['note_text']
    base = {
        'note_id': row['note_id'],
        'appointment_id': row['appointment_id'],
        'root_cause': '', 'confidence': '', 'summary': '',
        'recommended_action': '', 'validation_status': '', 'raw_response': ''
    }
    # Check pandas NaN before str(...).strip(), because NaN is a float.
    if pd.isna(note_text) or str(note_text).strip() == '':
        base.update({
            'root_cause': 'other', 'confidence': 0.0,
            'summary': 'No note recorded',
            'recommended_action': 'Re-call patient',
            'validation_status': 'skipped_blank', 'raw_response': ''
        })
        analysed_records.append(base)
        print(row['note_id'], 'blank note: skipped API call')
        continue

    raw_text = ''
    response_data = {}
    try:
        raw_text = generate_json(build_craft_prompt(str(note_text)))
        response_data = json.loads(raw_text)
        if not isinstance(response_data, dict):
            raise ValueError('Model response JSON must be an object')
        # Copy only fields actually returned; do not repair missing/invalid content.
        for key in ['root_cause', 'confidence', 'summary', 'recommended_action']:
            if key in response_data:
                base[key] = response_data[key]
        validated = FollowupInsight(**response_data)
        validate_allowed_root_cause(validated)
        base['validation_status'] = 'valid'
    except Exception as exc:
        base['validation_status'] = 'invalid'
        print(row['note_id'], 'reply retained as invalid:', str(exc))
    base['raw_response'] = raw_text
    analysed_records.append(base)
    time.sleep(13)  # Free-tier rate-limit pause between nonblank calls.

notes_analysed = pd.DataFrame(analysed_records, columns=[
    'note_id', 'appointment_id', 'root_cause', 'confidence', 'summary',
    'recommended_action', 'validation_status', 'raw_response'
])
notes_analysed.to_csv(ROOT / 'notes_analysed.csv', index=False)
print(notes_analysed[['note_id', 'root_cause', 'confidence', 'validation_status']].to_string(index=False))
print('Part 4.2 API calls:', API_CALLS)
"""))
cells.append(md("## Task 4.3 — Deliberately invalid replies and prompt comparison"""))
cells.append(code("""bad_replies = [
    {
        'root_cause': 'cost_or_fee', 'confidence': 1.7,
        'summary': 'Fee was too high', 'recommended_action': 'Offer fee information'
    },
    {
        'root_cause': 'work_or_schedule', 'confidence': 0.8,
        'summary': 'Work conflict was reported'
        # recommended_action intentionally missing
    }
]
for number, bad_reply in enumerate(bad_replies, start=1):
    try:
        FollowupInsight(**bad_reply)
    except ValidationError as exc:
        print(f'Bad reply {number} validation error:\\n{exc}\\n')

# V1: intentionally minimal prompt. V2: reuse the CRAFT replies already generated in Task 4.2.
def get_raw_root_cause(raw_text):
    try:
        parsed = json.loads(raw_text)
        value = parsed.get('root_cause', '') if isinstance(parsed, dict) else ''
        return value, value in ALLOWED_ROOT_CAUSES
    except Exception:
        return '', False

ab_rows = []
for note_id in ['N007', 'N009']:
    note_text = notes_df.loc[notes_df['note_id'].eq(note_id), 'note_text'].iloc[0]
    v1_prompt = 'Classify this note. Return JSON matching the required schema.\\n\\n' + str(note_text)
    v1_raw = generate_json(v1_prompt)
    v1_cause, v1_allowed = get_raw_root_cause(v1_raw)

    v2_row = notes_analysed.loc[notes_analysed['note_id'].eq(note_id)].iloc[0]
    v2_cause, v2_allowed = get_raw_root_cause(v2_row['raw_response'])
    ab_rows.append({'note_id': note_id, 'prompt_version': 'V1 basic', 'root_cause': v1_cause, 'allowed_value': v1_allowed})
    ab_rows.append({'note_id': note_id, 'prompt_version': 'V2 CRAFT (reused Task 4.2)', 'root_cause': v2_cause, 'allowed_value': v2_allowed})

prompt_comparison = pd.DataFrame(ab_rows)
print(prompt_comparison.to_string(index=False))
print('Deploy V2 CRAFT: it specifies the allowed categories and requires other rather than guessing when the note is ambiguous.')
print('Total API calls across Tasks 4.1–4.3:', API_CALLS)
"""))
cells.append(md("## Task 4.4 — Merge notes with appointment data and write recommendation evidence"""))
cells.append(code("""appointments_clean = pd.read_csv(APPOINTMENTS_FILE)
merged_notes = notes_analysed.merge(
    appointments_clean[['appointment_id', 'reminder_sent', 'no_show', 'clinic_id', 'lead_days']],
    on='appointment_id', how='left', validate='one_to_one'
)
print('Merged note rows:', len(merged_notes))
print('Root cause by reminder status:\\n', pd.crosstab(merged_notes['root_cause'], merged_notes['reminder_sent']))
print('Notes linked to reminder_sent=No:', int((merged_notes['reminder_sent'] == 'No').sum()), 'of', len(merged_notes))
print('Final note output file:', ROOT / 'notes_analysed.csv')
"""))
cells.append(md("""## Submission and privacy check

The final notebook must not contain the API key. The key is entered via `input()` at runtime. Clear outputs before committing if any output could contain the key. Commit the generated `notes_analysed.csv` after the run and copy the measured Part 4 counts and crosstab into `README.md`."""))
nb.cells=cells
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}}
nbf.write(nb,ROOT/'04_genai.ipynb')
print('Wrote 01_ingest_clean.ipynb, 03_model.ipynb, 04_genai.ipynb')
