# CareLoop execution status

The public repository now contains the three untouched source CSVs and the required Part 1–4 output files.

## Verified deliverables

- Part 1: 1,020 raw bookings; 988 accepted and 32 rejected by CSV conversion; 990 unique cleaned bookings; 218 no-shows; 16 clean output columns with no missing values.
- Part 2: 8 clinics, 990 appointments, 218 no-shows. All requested clinic, lead-time, monthly and city-ranking queries were verified against the SQLite database. Duplicate-key test returned `UNIQUE constraint failed: appointments.appointment_id`.
- Part 3: 792 training and 198 test rows, with 49 no-shows in the test set. Logistic regression selected by training-only five-fold ROC-AUC (about 0.774); threshold 0.35; test ROC-AUC about 0.789, precision 0.571, recall 0.571, 49 flags.
- Part 4: `notes_analysed.csv` contains 10 rows and 8 columns. Eight valid Gemini classifications, N004 skipped blank, N008 retained as invalid with empty raw response because API requests failed. Two forgot/no-reminder cases both correspond to `reminder_sent = No`; the full 10-row reminder cross-tab is in README.

## API reliability note

Gemini requests encountered 503/504 service errors and a 429 quota response. The analysis file preserves the actual valid replies and explicitly marks the failed N008 row invalid; it does not invent a reason. The notebook includes the model/prompt and Pydantic validation workflow. The final README documents the partial API failure.

## Weather API note

Open-Meteo could not be reached during preparation. Consequently `clinics_enriched.csv` contains blank `temp_c` values and `load_data.sql` writes them as unquoted SQL `NULL`. The Part 1 notebook includes the required HTTP status check, timeout, one retry and failure reporting; rerun it in Colab when the API is reachable to fetch temperatures.

## Final checks before submission

- Confirm the notebook outputs do not reveal the Gemini key.
- Confirm repository visibility remains Public.
- Submit only this public repository URL.
