# CareLoop Clinics — No-Show Guard

**Public repository:** https://github.com/Subham9339063436/CareLoop-Clinics-No-Show-Guard

End-to-end outpatient appointment no-show analysis: data foundations, SQL verification, machine-learning risk scoring, and structured follow-up-note review using Gemini.

## Four-part findings summary

- **Part 1 — Data foundations:** 1,020 raw bookings were parsed; 32 rows failed the CSV type-conversion exercise, and pandas cleaning produced 990 unique bookings with 16 complete output columns. There were 218 no-shows (22.0%).
- **Part 2 — SQL analysis:** SQL verified 8 clinics, 990 appointments and 218 no-shows. The 31+ day lead-time band had a 60.5% no-show rate; Gomti Nagar Care Point recorded 33.3% from 90 appointments.
- **Part 3 — Machine learning:** Logistic regression won training-only five-fold cross-validation (ROC-AUC 0.774). At the selected 0.35 threshold it achieved test precision 0.571, recall 0.571 and ROC-AUC 0.789, flagging 49 of 198 test bookings.
- **Part 4 — GenAI follow-up review:** Of the first 10 notes, 8 received valid structured classifications, N004 was correctly skipped as blank, and N008 remained invalid after Gemini service/quota errors. The two valid forgot/no-reminder notes (N001 and N007) were both linked to `reminder_sent = No`.

## Business question

Which appointments are most likely to be missed, why do patients miss them, and what should CareLoop do first?

## Workflow

1. **Python foundations:** `01_ingest_clean.ipynb` reads the original bookings via `csv.DictReader`, records conversion failures, restarts from the raw CSV for pandas cleaning, derives features, enriches clinics with Open-Meteo when reachable and exports the shared clean table plus SQL inserts.
2. **SQL analysis:** `02_queries.sql` creates the tables, loads the Part 1 outputs through `load_data.sql`, and provides clinic, lead-time, monthly and within-city analyses.
3. **Machine learning:** `03_model.ipynb` compares logistic regression, decision trees and random forests. Training-only cross-validation selects the model and threshold; the held-out test set is used for final reporting and `test_risk_scores.csv`.
4. **GenAI:** `04_genai.ipynb` classifies the first 10 call-centre notes, validates JSON with Pydantic, tests invalid replies and compares a basic prompt with the CRAFT prompt. The generated `notes_analysed.csv` retains raw model text and validation status. N008 has an empty raw response because the Gemini service/quota errors prevented a model reply; it is retained as invalid rather than guessed or repaired.

## Main findings and recommendation

The cleaned table contains 990 distinct bookings and 218 no-shows (22.0%). Long booking lead times are an important risk signal: bookings made 31 or more days ahead had 26 no-shows among 43 appointments (60.5%), compared with 16.2% for the 1–7 day band. Among clinics with at least 95 bookings, Hadapsar Care Centre had the highest no-show rate at 27.3% (39/143); when all clinics are included, Gomti Nagar Care Point is highest at 33.3% (30/90), but its smaller sample warrants caution. Logistic regression was selected using training-only five-fold cross-validation (ROC-AUC 0.774). The 0.35 threshold improved the training out-of-fold F1 to 0.489 (recall 0.467), compared with F1 0.347 (recall 0.249) at 0.50; on the held-out test set it produced ROC-AUC 0.789, precision 0.571, recall 0.571 and 49 flags out of 198. In the first 10 notes, 8 classifications were valid; the two patients classified as forgot/no-reminder both had `reminder_sent = No`, and 5 of the 10 linked bookings had no reminder recorded. CareLoop should first use the model to prioritize evening-before calls for bookings with risk probability at least 0.35, especially long-lead bookings, while piloting reminder-delivery/contact checks, transport assistance, fee-support information and evening/Saturday rescheduling options. The live operating threshold should be adjusted to the number of calls the front desk can make in an evening and the cost of an empty doctor slot. The note evidence is exploratory, not clinic-wide proof.

## Part 4 — note reasons and reminder cross-tab

The first 10 rows of `followup_notes.csv` produced 8 valid model replies, 1 required blank-note skip (N004), and 1 invalid row (N008) with no raw response because the API calls failed. Among the 8 valid classifications, the observed counts were: forgot/no-reminder 2, distance/transport 1, cost/fee 1, illness/emergency 1, work/schedule 2, and other 1. N008 is unclassified; it is not counted in those reason totals. The required skipped-blank convention sets N004 to `other` with confidence 0.0, so that placeholder is visible in the cross-tab but must not be interpreted as a patient-reported reason.

The merged 10-row cross-tabulation of root cause against reminder status is:

| Root cause in output | Reminder No | Reminder Yes |
|---|---:|---:|
| forgot_or_no_reminder | 2 | 0 |
| distance_or_transport | 1 | 0 |
| cost_or_fee | 0 | 1 |
| illness_or_emergency | 0 | 1 |
| work_or_schedule | 1 | 1 |
| other (N004 skipped placeholder + N009) | 1 | 1 |
| blank/invalid (N008 API failure) | 0 | 1 |
| **Total** | **5** | **5** |

Thus, both forgotten/no-reminder cases (2/2) were associated with `reminder_sent = No`; 5 of 10 sampled bookings had no reminder recorded. This is a tiny convenience sample and supports follow-up hypotheses, not causal or clinic-wide conclusions.

## Important modelling interpretation

In the fitted logistic model, the reminder coefficient on the standardized feature is approximately -0.581. The exponentiated one-standard-deviation effect is approximately 0.560, while the No-to-Yes feature change corresponds to `exp(coef / scale) ≈ 0.309`. This is an observational association conditional on the included features, not proof that reminders cause fewer no-shows.

## Data and implementation notes

- Source booking export: 1,020 rows, 14 columns; 20 exact duplicates removed; 12 blank ages plus 5 invalid ages set missing and imputed with the age median; 20 missing distances imputed with the distance median; outliers capped with the specified IQR fences.
- `appointments_clean.csv` has 990 rows, 16 columns, unique `appointment_id`, and no missing values.
- `load_data.sql` contains 8 clinic inserts followed by 990 appointment inserts (998 total). `temp_c` is SQL `NULL` where Open-Meteo could not be reached from the preparation environment; no temperature was invented. The notebook contains the required live request, status handling and retry.
- Duplicate-key test was run in SQLite. Observed message: `UNIQUE constraint failed: appointments.appointment_id`.
- Random seed is 42, test size is 0.2, percentages are rounded to one decimal place, NumPy uses its default population standard deviation where specified, and quartiles use pandas/NumPy defaults.
- Gemini API service errors included 503/504 availability/timeouts and a 429 quota response. The notebook retains invalid outcomes and does not fabricate missing model replies.

## Limitations

1. The medians, quartiles and outlier fences were computed on all 990 cleaned rows before train/test splitting, allowing a small indirect influence from test rows.
2. The same patient may appear in both training and test sets because the split is by booking row, not patient.
3. The random split does not simulate a real past-to-future deployment; the results come from one train/test split.
4. The operating threshold depends on call-centre capacity and the trade-off between empty slots and unnecessary calls.
5. The first 10 follow-up notes are illustrative and cannot establish clinic-wide reasons; clinic-level rates with smaller sample sizes are less stable.
6. Reminder effects are observational associations and do not establish causation.
7. N008 could not be classified because the free Gemini service returned availability/quota errors; the invalid row is preserved rather than replaced with a guessed reason.
8. Open-Meteo was unavailable during preparation, so clinic temperatures remain SQL NULL until the enrichment notebook can reach the endpoint.

## Repository contents

- `data/appointments.csv`, `data/clinics.csv`, `data/followup_notes.csv` — original Data Pack files
- `01_ingest_clean.ipynb`, `rejected_rows.json`, `appointments_clean.csv`, `clinics_enriched.csv`, `load_data.sql`
- `02_queries.sql`
- `03_model.ipynb`, `test_risk_scores.csv`
- `04_genai.ipynb`, `notes_analysed.csv`

## Running the notebooks

Use Google Colab for Python and MySQL Workbench or SQLiteOnline for SQL. Run Part 1 before Parts 2 and 3. Part 4 uses a Gemini API key entered only at the runtime `input()` prompt; never hard-code or commit the key. Clear outputs before committing if any output could expose credentials.
