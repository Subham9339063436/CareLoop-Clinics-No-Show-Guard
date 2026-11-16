# CareLoop execution status

The repository now contains the three original CSVs, the full Part 1, Part 2 and Part 3 notebooks/queries, and the generated Part 1–3 outputs. The files were checked against the assignment acceptance figures, and the SQL was loaded into SQLite to verify the queries.

## Verified in this run

- Part 1: 1,020 raw bookings; 988 accepted and 32 rejected by CSV conversion; 990 unique cleaned bookings; 218 no-shows; 16 output columns with no missing values.
- Part 2: 8 clinics, 990 appointments, 218 no-shows; all clinic, lead-band, monthly and ranking queries run successfully in SQLite. Duplicate key test produced `UNIQUE constraint failed: appointments.appointment_id`.
- Part 3: 792 train / 198 test rows; 49 test no-shows; logistic regression selected by training-only five-fold ROC-AUC (0.7737); threshold 0.35; final test ROC-AUC 0.7889, precision 0.5714, recall 0.5714 and 49 flagged bookings.

## Remaining work

- Open `04_genai.ipynb` in Google Colab. Its first cell clones this public repository automatically when needed.
- Generate a free Gemini API key in Google AI Studio and enter it only at the notebook's runtime `input()` prompt. Never paste it into a cell or commit it.
- Run the notebook top to bottom. It analyses the first 10 notes, skips blank N004 without an API call, performs the Pydantic failure tests, compares V1/V2 prompts and saves `notes_analysed.csv`.
- Verify the API call count in the notebook, download `notes_analysed.csv`, and upload it to the repository root.
- Replace the pending Part 4 paragraph in `README.md` with the actual note-reason counts and reminder cross-tabulation.

## Weather API

The Open-Meteo API could not be reached from the preparation runtime. Consequently `clinics_enriched.csv` contains blank `temp_c` values and `load_data.sql` writes them as unquoted SQL `NULL`. Part 1 includes the required request, timeout, one retry and failure reporting; rerun in Colab to retrieve temperatures if the endpoint is reachable.

## Submission rule

Submit one link only: this public GitHub repository. Do not commit the Gemini API key or notebook output containing it.
