# Part 4 final run required

The first 10 source notes are included in `data/followup_notes.csv`, but no Gemini API key was supplied. To avoid inventing model-generated results, `notes_analysed.csv` has intentionally not been fabricated.

1. Open `04_genai.ipynb` in Google Colab.
2. Confirm `MODEL_NAME` is available to your Gemini API key in Google AI Studio. The current Flash model in the notebook is `gemini-3.8-flash`.
3. Run the notebook and enter the Gemini API key only at the runtime `input()` prompt. Never paste the key into a code cell or commit it.
4. Confirm N004 is `skipped_blank`, review the 9 nonblank results and raw responses, and check the API call count.
5. Download `notes_analysed.csv`, remove any notebook outputs that might expose credentials, and add both files to the GitHub repository.
6. Update the Part 4 findings paragraph in `README.md` using the actual generated notes and cross-tabulation.

The notebook uses 12 expected API calls when Task 4.3 reuses the Task 4.2 V2/CRAFT outputs (1 call for Task 4.1, 9 nonblank calls for Task 4.2, and 2 V1 calls for N007 and N009).
