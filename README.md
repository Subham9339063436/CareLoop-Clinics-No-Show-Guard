# CareLoop Clinics — No-Show Guard

**Public repository:** https://github.com/Subham9339063436/CareLoop-Clinics-No-Show-Guard

End-to-end analysis of outpatient appointment no-shows: data ingestion and cleaning, SQL verification, machine-learning risk scoring, and structured call-centre note review with Gemini.

## Four-part findings summary

- **Part 1 — Data foundations:** 1,020 source bookings were cleaned to 990 unique appointments; 32 CSV type-conversion failures were retained in `rejected_rows.json`; 218 of 990 appointments were no-shows (22.0%).
- **Part 2 — SQL analysis:** SQL verified 8 clinics, 990 appointments and 218 no-shows; the 31+ day lead-time band had a 60.5% no-show rate; Gomti Nagar recorded 33.3% from 90 appointments.
- **Part 3 — Machine learning:** Logistic regression won training-only cross-validation (ROC-AUC 0.774); the selected 0.35 threshold produced 0.57 precision, 0.57 recall and 49 flagged bookings on the 198-row test set (test ROC-AUC 0.789).
- **Part 4 — GenAI follow-up review:** notebook and validation workflow are prepared; final note results and the cross-tabulation await the Gemini run with a runtime-entered API key. No model-generated note classifications have been fabricated.

## Business question

Which appointments are most likely to be missed, why do patients miss them, and what should CareLoop do first?

## Data and workflow

1. **Python foundations:** `01_ingest_clean.ipynb` reads the original bookings via `csv.DictReader`, records conversion failures, restarts from the raw CSV for pandas cleaning, derives features, calls Open-Meteo for clinic enrichment and exports the shared clean table plus SQL inserts.
2. **SQL analysis:** `02_queries.sql` creates the tables, loads the Part 1 outputs through `load_data.sql`, and provides all requested clinic, lead-time, monthly and within-city analyses.
3. **Machine learning:** `03_model.ipynb` compares logistic regression, decision trees and random forests. Training-only cross-validation selects the model and threshold; the held-out test set is used once for reporting and `test_risk_scores.csv`.
4. **GenAI:** `04_genai.ipynb` classifies the first 10 call-centre notes, validates the structured JSON with Pydantic, tests invalid replies and compares a basic prompt with the CRAFT prompt. It preserves `raw_response` and creates `notes_analysed.csv` when run.

## Main findings and operational interpretation

The clean analysis table contains 990 distinct bookings and 218 no-shows, a 22.0% rate. Long booking lead times are a high-risk signal: bookings made 31 or more days ahead had 26 no-shows among 43 appointments (60.5%), versus 16.2% for the 1–7 day band. Among clinics with at least 95 bookings, Hadapsar Care Centre had the highest rate at 27.3% (39/143); when all clinics are included, Gomti Nagar Care Point is highest at 33.3% (30/90), but the smaller count should be treated cautiously.

Logistic regression was selected because it achieved the highest training-only five-fold ROC-AUC (0.774), ahead of the depth-3 tree (0.678), 100-tree random forest (0.747), and tuned forest (0.767). Its selected operating threshold is 0.35: the training out-of-fold F1 was 0.489 (recall 0.467) compared with F1 0.347 (recall 0.249) at 0.50. On the held-out test set the selected threshold yielded ROC-AUC 0.789, precision 0.571, recall 0.571 and 49 flagged bookings out of 198. The clinic should set the real threshold against the number of calls staff can make in an evening and the cost of an unfilled doctor slot.

**Part 4 must be run before this becomes the final cross-part recommendation.** The resulting structured reasons and `pd.crosstab(root_cause, reminder_sent)` should be added here; the 10-note sample is illustrative and not clinic-wide evidence.

## Important modelling interpretation

In the fitted logistic model, the reminder coefficient on the standardized feature is approximately -0.581. The exponentiated one-standard-deviation effect is approximately 0.560, while the No-to-Yes feature change corresponds to `exp(coef / scale) ≈ 0.309`. This is an observational association conditional on the included features, not proof that reminders cause fewer no-shows.

## Data and implementation notes

- Source booking export: 1,020 rows, 14 columns; 20 exact duplicates removed; missing ages and distances imputed with the specified medians; invalid ages set missing before imputation; outliers capped with the specified IQR fences.
- `appointments_clean.csv` has 990 rows, 16 columns, unique `appointment_id`, and no missing values.
- `load_data.sql` contains 8 clinic inserts followed by 990 appointment inserts (998 total). `temp_c` is SQL `NULL` where Open-Meteo could not be reached from the execution environment; no temperature was invented. The notebook contains the required live request, status handling and retry for execution where network access is available.
- Duplicate-key test was run in the permitted SQLite alternative. Observed message: `UNIQUE constraint failed: appointments.appointment_id`.
- Random seed is 42, test size is 0.2, percentages are rounded to one decimal place, NumPy uses its default population standard deviation where specified, and quartiles use pandas/NumPy defaults.

## Limitations

1. The medians, quartiles and outlier fences were computed on all 990 cleaned rows before train/test splitting, allowing a small indirect influence from test rows.
2. The same patient may appear in both training and test sets because the split is by booking row, not patient.
3. The random split does not simulate a real past-to-future deployment; the results come from one train/test split.
4. The operating threshold depends on call-centre capacity and the trade-off between empty slots and unnecessary calls.
5. The first 10 follow-up notes are only an illustration and cannot establish clinic-wide reasons; clinic-level rates with smaller sample sizes are less stable.
6. Reminder effects are observational associations and do not establish causation.

## Repository contents

- `data/appointments.csv`, `data/clinics.csv`, `data/followup_notes.csv` — original files supplied in the assignment Data Pack (kept unchanged)
- `01_ingest_clean.ipynb`, `rejected_rows.json`, `appointments_clean.csv`, `clinics_enriched.csv`, `load_data.sql`
- `02_queries.sql`
- `03_model.ipynb`, `test_risk_scores.csv`
- `04_genai.ipynb`, `notes_analysed.csv` (generated after completing the Gemini run)

## Running the notebooks

Use Google Colab for Python and MySQL Workbench or SQLiteOnline for SQL. Run Part 1 before Parts 2 and 3. Run Part 4 with your own free Gemini API key entered at the `input()` prompt; never commit the key or notebook output containing it. After Part 4 executes, commit `notes_analysed.csv`, then replace the pending Part 4 findings paragraph with the observed root-cause counts and reminder cross-tab.
