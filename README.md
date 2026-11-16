# CareLoop Clinics — No-Show Guard

**Repository:** https://github.com/Subham9339063436/CareLoop-Clinics-No-Show-Guard

End-to-end outpatient appointment no-show analysis: data ingestion and cleaning, SQL verification, machine-learning risk scoring, and structured call-centre note review with Gemini.

> **Data requirement:** Put the three untouched assignment files in `data/`: `appointments.csv`, `clinics.csv`, and `followup_notes.csv`. Every derived result must be generated from these originals. Do not substitute reference/acceptance figures for measured outputs.

## Business question

Which appointments are most likely to be missed, why do patients miss them, and what should CareLoop do first?

## Project structure

```text
CareLoop-Clinics-No-Show-Guard/
├── data/
│   ├── appointments.csv
│   ├── clinics.csv
│   └── followup_notes.csv
├── 01_ingest_clean.ipynb
├── rejected_rows.json
├── appointments_clean.csv
├── clinics_enriched.csv
├── load_data.sql
├── 02_queries.sql
├── 03_model.ipynb
├── test_risk_scores.csv
├── 04_genai.ipynb
├── notes_analysed.csv
└── README.md
```

## Workflow

1. **Python foundations:** read raw bookings with `csv.DictReader`; inspect, clean, validate, engineer features and export with pandas/NumPy; enrich the clinic master from Open-Meteo; create SQL insert statements.
2. **SQL analysis:** load the exported tables into MySQL 8 (or SQLiteOnline) and reproduce totals, clinic rates, lead-time bands, monthly changes and within-city ranks.
3. **Machine learning:** compare logistic regression, decision trees and random forests. Select model and threshold with training-only cross-validation; evaluate the held-out test set once and export risk scores.
4. **GenAI follow-up review:** classify the first 10 call-centre notes with Gemini JSON output and Pydantic validation; preserve invalid replies and skipped blanks; compare a basic prompt with the CRAFT prompt.

## Prediction point and feature rules

The risk score is generated on the evening before the appointment, after reminders have been sent or skipped. Therefore `reminder_sent` and `prior_no_shows` are available. Do not use identifiers, dates, clinic ID, gender or status as model features. Do not use the test set to choose a model, setting or threshold.

## Deliverables

| File | Purpose |
|---|---|
| `01_ingest_clean.ipynb` | Part 1 tasks 1.1–1.6 |
| `rejected_rows.json` | Rows rejected by CSV-module conversion |
| `appointments_clean.csv` | Clean 16-column table shared by Parts 2 and 3 |
| `clinics_enriched.csv` | Clinic master plus Open-Meteo temperature |
| `load_data.sql` | Clinic inserts followed by appointment inserts |
| `02_queries.sql` | Part 2 SQL queries and duplicate-key error comment |
| `03_model.ipynb` | Part 3 model comparison and threshold selection |
| `test_risk_scores.csv` | Test-set probabilities and risk flags |
| `04_genai.ipynb` | Part 4 Gemini analysis, validation and prompt tests |
| `notes_analysed.csv` | Structured results plus raw model responses |

## Reproducibility rules

- `random_state=42`; `test_size=0.2`
- Percentages rounded to one decimal place
- Use NumPy's default population standard deviation where specified
- Use pandas/NumPy default quartile method
- Gemini API key is entered at runtime only; never hard-code or commit it

## Findings (populate from executed outputs)

- **Part 1 — Data foundations:** pending execution on the supplied original CSVs.
- **Part 2 — SQL:** pending execution against the Part 1 exports.
- **Part 3 — Risk model:** pending training-only cross-validation and one held-out evaluation.
- **Part 4 — Follow-up reasons:** pending analysis of the first 10 notes.

## Recommendation

The recommendation will be grounded in the actual cleaned data, SQL cluster analysis, test-set model results and the small qualitative sample of call-centre notes. It will include at least four measured numbers from Parts 1–4 and a practical call-capacity trade-off. No result will be invented before the source files have been processed.

## Limitations to report

Discuss the limitations that apply to this run: the first 10 notes are illustrative and cannot establish clinic-wide reasons; cleaning statistics are calculated on all rows before the split; the same patient may appear in both train and test; the random split does not simulate past-to-future deployment; a single split is uncertain; threshold choice depends on evening call capacity; small clinic sample sizes can make rates volatile; and observational associations do not prove causation.

## Run environment

The assignment targets Google Colab for Python and MySQL Workbench / SQLiteOnline for SQL. Follow task instructions in the notebooks. In the GenAI notebook, enter your own Gemini free-tier API key privately at runtime and clear all outputs before committing.

## Submission checklist

- [ ] Put the three untouched originals in `data/`.
- [ ] Run Part 1 and verify every acceptance check.
- [ ] Run the SQL queries and reconcile their counts with Python.
- [ ] Train and compare models; select model and threshold using training-only cross-validation.
- [ ] Evaluate the selected model once on the held-out test set.
- [ ] Analyse the first 10 notes; skip the blank note without an API call and preserve raw replies.
- [ ] Write the recommendation using numbers from the actual outputs.
- [ ] Search all committed files for secrets before submission.
