from pathlib import Path
import csv, json, time
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
APPT_FILE = DATA / 'appointments.csv'
CLINIC_FILE = DATA / 'clinics.csv'
OUT = ROOT

# Task 1.1 — CSV-module ingestion and type conversion
with open(APPT_FILE, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames
    raw_rows = list(reader)
print('Task 1.1 header:', header)
print('Task 1.1 row count:', len(raw_rows))

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
with open(OUT / 'rejected_rows.json', 'w', encoding='utf-8') as f:
    json.dump(rejected, f, indent=2)
print('Accepted:', len(accepted), 'Rejected:', len(rejected))

# Task 1.2 — pandas cleaning, restarted from the original CSV
# Keep the original string values and let pandas infer missing values.
df = pd.read_csv(APPT_FILE)
print('\nTask 1.2 raw shape:', df.shape)
print('Task 1.2 info:')
df.info()
print('Task 1.2 nulls:\n', df.isnull().sum())
print('Task 1.2 exact duplicates:', df.duplicated().sum())

df = df.drop_duplicates()
print('Rows after exact duplicate removal:', len(df))
df['booking_channel'] = df['booking_channel'].str.strip().str.title()
print('Rows after booking_channel cleaning:', len(df), 'unique channels:', sorted(df['booking_channel'].dropna().unique().tolist()))
df['appt_date'] = pd.to_datetime(df['appt_date'], errors='coerce')
df['booked_on'] = pd.to_datetime(df['booked_on'])
print('Rows after date conversion:', len(df))
df = df.dropna(subset=['appt_date', 'status'])
print('Rows after dropping missing appt_date/status:', len(df))

# Task 1.3 — validity, gaps and outliers
invalid_age = (df['age'] < 1) | (df['age'] > 100)
print('Invalid ages:', int(invalid_age.sum()))
df.loc[invalid_age, 'age'] = np.nan
age_median = df['age'].median()
distance_median = df['distance_km'].median()
df['age'] = df['age'].fillna(age_median).round().astype(int)
df['distance_km'] = df['distance_km'].fillna(distance_median)
print('Age median used:', age_median, 'Distance median used:', distance_median)

cap_counts = {}
fences = {}
for col in ['distance_km', 'fee_inr']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    before = df[col].copy()
    df[col] = np.clip(df[col], low, high)
    cap_counts[col] = int((before != df[col]).sum())
    fences[col] = {'Q1': float(q1), 'Q3': float(q3), 'IQR': float(iqr), 'lower': float(low), 'upper': float(high)}
    print(f'{col}: Q1={q1}, Q3={q3}, lower={low}, upper={high}, capped={cap_counts[col]}')
df['distance_km'] = df['distance_km'].round(1)
df['fee_inr'] = df['fee_inr'].round().astype(int)

# Task 1.4 — features and NumPy summary
df['lead_days'] = (df['appt_date'] - df['booked_on']).dt.days
df['no_show'] = (df['status'] == 'No-Show').astype(int)
X_summary = df[['age', 'distance_km', 'fee_inr', 'lead_days']].to_numpy()
means = np.mean(X_summary, axis=0)
long_lead_mask = df['lead_days'] > 14
long_lead_count = int(long_lead_mask.sum())
long_lead_share = float(long_lead_mask.mean())
distance = df['distance_km'].to_numpy()
distance_z = (distance - np.mean(distance)) / np.std(distance)
z_outlier_count = int((np.abs(distance_z) > 2).sum())
print('Means [age, distance_km, fee_inr, lead_days]:', means)
print('No-shows:', int(df['no_show'].sum()), '/', len(df), 'rate:', round(100*df['no_show'].mean(), 1))
print('lead_days > 14:', long_lead_count, long_lead_share)
print('Distance |z| > 2:', z_outlier_count)

# Task 1.5 — clinic enrichment via Open-Meteo; no invented values on failure
clinics = pd.read_csv(CLINIC_FILE)
temps = []
for _, clinic in clinics.iterrows():
    temp = None
    reason = None
    params = {'latitude': clinic['latitude'], 'longitude': clinic['longitude'], 'hourly': 'temperature_2m', 'forecast_days': 1}
    for attempt in range(2):
        try:
            response = requests.get('https://api.open-meteo.com/v1/forecast', params=params, timeout=10)
            if response.status_code != 200:
                reason = f'HTTP status {response.status_code}'
                continue
            payload = response.json()
            values = payload.get('hourly', {}).get('temperature_2m')
            if not values:
                reason = 'hourly.temperature_2m missing or empty'
                continue
            temp = float(values[0])
            reason = None
            break
        except requests.exceptions.RequestException as exc:
            reason = f'RequestException: {exc}'
        except (ValueError, TypeError) as exc:
            reason = f'Invalid API response: {exc}'
    if temp is None:
        print(f"Open-Meteo failed for {clinic['clinic_name']}: {reason}; temp_c set to NULL")
    temps.append(temp)
clinics['temp_c'] = temps
clinics.to_csv(OUT / 'clinics_enriched.csv', index=False)

# Task 1.6 — export exact 16-column hand-off table and SQL load statements
cols = ['appointment_id','patient_id','clinic_id','booked_on','appt_date','age','gender','distance_km','fee_inr','booking_channel','visit_type','reminder_sent','prior_no_shows','status','lead_days','no_show']
df = df.sort_values('appointment_id').copy()
for date_col in ['booked_on', 'appt_date']:
    df[date_col] = df[date_col].dt.strftime('%Y-%m-%d')
export_df = df[cols].copy()
print('appointment_id unique:', bool(export_df['appointment_id'].is_unique))
print('missing values by exported column:\n', export_df.isnull().sum())
if not export_df['appointment_id'].is_unique:
    raise ValueError('appointment_id must be unique before export')
if export_df.isnull().any().any():
    raise ValueError('The 16-column export must have no missing values')
export_df.to_csv(OUT / 'appointments_clean.csv', index=False)

# SQL text is permitted for this controlled educational dataset per assignment brief.
def sql_value(value, is_text=False):
    if pd.isna(value):
        return 'NULL'
    if is_text:
        return "'" + str(value).replace("'", "''") + "'"
    return str(value)

clinic_text_cols = ['clinic_id','clinic_name','city','tier']
clinic_lines = []
for _, row in clinics.iterrows():
    vals = [sql_value(row[c], c in clinic_text_cols) for c in ['clinic_id','clinic_name','city','tier','latitude','longitude','temp_c']]
    clinic_lines.append('INSERT INTO clinics VALUES (' + ','.join(vals) + ');')
app_text_cols = ['appointment_id','patient_id','clinic_id','booked_on','appt_date','gender','booking_channel','visit_type','reminder_sent','status']
app_lines = []
for _, row in export_df.iterrows():
    vals = [sql_value(row[c], c in app_text_cols) for c in cols]
    app_lines.append('INSERT INTO appointments VALUES (' + ','.join(vals) + ');')
with open(OUT / 'load_data.sql', 'w', encoding='utf-8') as f:
    f.write('-- Generated from clinics_enriched.csv and appointments_clean.csv\n')
    f.write('-- Create the tables using 02_queries.sql before running this file.\n\n')
    f.write('\n'.join(clinic_lines))
    f.write('\n\n')
    f.write('\n'.join(app_lines))
    f.write('\n')
print('appointments_clean.csv:', export_df.shape, 'clinics_enriched.csv:', clinics.shape)
print('SQL INSERT statements:', len(clinic_lines)+len(app_lines))

# Save a machine-readable audit summary for reproducibility / notebook validation.
summary = {
    'raw_rows': len(raw_rows), 'accepted': len(accepted), 'rejected': len(rejected),
    'deduplicated_rows': len(df), 'clean_rows': len(export_df), 'no_shows': int(export_df['no_show'].sum()),
    'no_show_rate_pct': round(100*export_df['no_show'].mean(), 1), 'means': means.tolist(),
    'long_lead_count': long_lead_count, 'long_lead_share': long_lead_share,
    'distance_abs_z_gt_2': z_outlier_count, 'cap_counts': cap_counts, 'fences': fences,
    'invalid_age_count': int(invalid_age.sum()), 'age_median': float(age_median),
    'distance_median': float(distance_median), 'api_temp_nonnull': int(clinics['temp_c'].notna().sum()),
    'api_temp_null': int(clinics['temp_c'].isna().sum()),
}
with open(OUT / 'part1_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
