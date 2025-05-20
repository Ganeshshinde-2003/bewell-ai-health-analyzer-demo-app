import google.generativeai as genai
import os
import fitz  # PyMuPDF - for PDF
import io  # To handle file bytes
import pandas as pd  # for Excel and CSV
from docx import Document  # for .docx files
import json
import re

# --- Text Extraction Function (Handles multiple file types) ---
def extract_text_from_file(file_path):
    """
    Extracts text content from various file types (PDF, DOCX, XLSX, XLS, TXT, CSV)
    from a given file path.
    Returns extracted text as a string or an error marker.
    """
    if not file_path:
        return ""

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return f"[Error: File not found at {file_path}]"

    file_extension = os.path.splitext(file_path)[1].lower()
    text = ""

    try:
        with open(file_path, 'rb') as f:
            file_bytes = f.read()

        if file_extension == ".pdf":
            try:
                pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
                for page_num in range(pdf_document.page_count):
                    page = pdf_document.load_page(page_num)
                    text += page.get_text("text") + "\n"
                pdf_document.close()
            except Exception as e:
                print(f"Error processing PDF file '{file_path}': {e}")
                return "[Error Processing PDF File]"

        elif file_extension == ".docx":
            try:
                doc = Document(io.BytesIO(file_bytes))
                for para in doc.paragraphs:
                    text += para.text + "\n"
            except Exception as e:
                print(f"Error processing DOCX file '{file_path}': {e}")
                return "[Error Processing DOCX File]"

        elif file_extension in [".xlsx", ".xls"]:
            try:
                excel_data = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
                for sheet_name, df in excel_data.items():
                    text += f"--- Sheet: {sheet_name} ---\n"
                    text += df.to_string(index=False) + "\n\n"
            except Exception as excel_e:
                print(f"Warning: Could not read Excel file '{file_path}'. Ensure 'openpyxl' (for .xlsx) or 'xlrd' (for .xls) is installed if needed. Error: {excel_e}")
                return f"[Could not automatically process Excel file {file_path} - Please try pasting the text or saving as PDF/TXT.]"

        elif file_extension in [".txt", ".csv"]:
            text = file_bytes.decode('utf-8', errors='ignore')

        else:
            print(f"Warning: Unsupported file type: {file_extension} for file '{file_path}'")
            return "[Unsupported File Type]"

    except Exception as e:
        print(f"An error occurred while processing '{file_path}': {e}")
        return "[Error Processing File]"

    if not text.strip() and file_extension not in [".txt", ".csv"] and not text.startswith("["):
        print(f"Warning: No readable text extracted from '{file_path}'. The file might be scanned, empty, or have complex formatting. Consider checking the file.")

    return text

# --- API Key Handling ---
# For a plain script, you can set it directly or load from an environment variable.
# DO NOT hardcode your API key in a production application. Use environment variables.
api_key = "AIzaSyArQ9zeya1SO-IwsMappkLStXYT0W7WXfk" # Replace with your actual API key or load from os.getenv("GOOGLE_API_KEY")

# --- Gemini Model Configuration ---
model_name = "gemini-2.0-flash" # Use a fast model for this purpose

generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain", # Request text/plain, but instruct AI to return JSON
}

# --- The Main Prompt Construction for Monthly Report ---
MONTHLY_REPORT_PROMPT_INSTRUCTIONS = """
Role: You are the Bewell AI Monthly Health Report Assistant. You are a holistic health expert focused on women’s hormonal balance, precision medicine, and root-cause insights.
Tone: Empathetic, empowering, accessible, science-backed — never condescending. Avoid jargon.

Objective: Generate a comprehensive, personalized, and science-backed monthly health summary for the user. The analysis should synthesize the following data:
{previous_lab_report_data}
{daily_logs_data}
{weekly_assessments_data}

Instructions for Output:
Generate ONE complete response as a JSON object. Do NOT include any introductory or concluding text outside the JSON object. **Do NOT wrap the JSON object in markdown code blocks (```json ... ``` or ``` ... ```).** Populate all fields based *only* on the provided user data.

**Understanding the Daily Logs Structure (crucial for accurate analysis):**
The daily logs will contain two main sections with specific columns:

1.  **Logged Routines:**
    * `Date`
    * `Category` (e.g., Sleep Well, Eat Well, Move Well, Recover Well)
    * `Time` (This column is for sleep duration in hours and meal times for 'Eat Well' entries)
    * `Meal_Order`
    * `Image`
    * `Feeling` (General feeling/qualitative note about the routine)
    * `Earning Balance` (Factors that positively contributed to hormonal balance)
    * `Losing Balance` (Factors that negatively impacted hormonal balance)

2.  **Logged Symptoms:**
    * `Date`
    * `Mood`
    * `Symptoms`
    * `Energy Level`

**Specific Instructions for Analysis:**

**For numerical metrics (e.g., average sleep hours, activity days percentage, scores):**
* **Prioritize identifying and extracting explicit numerical values** where they appear in structured columns (like `Time` for sleep hours).
* If explicit numerical data is *not* present for a metric, **infer a reasonable numerical score (e.g., on a 0-5 scale, where 5 is best) based on the qualitative descriptions** in the associated columns (e.g., `Feeling`, `Earning Balance`, `Losing Balance`, `Mood`, `Energy Level`). For example, "Slept soundly" could infer a sleep quality score of 4 or 5, while "Nauseous" after a meal could infer a meal satisfaction score of 1 or 2.
* If *no* relevant data (neither explicit numbers nor inferable qualitative descriptions) is available for a metric, then return `0` or `null`.

**For 'hormonal_balance_insight' section:**
* Analyze the user's daily logs, *specifically looking at the `Earning Balance` and `Losing Balance` columns* for direct indicators of hormonal impact. Also consider the `Feeling` column for qualitative input on balance.
* The `score_or_message` field must always provide a relevant, non-empty assessment of their hormonal balance status based on this analysis. It should *not* be "0" or generic if relevant data is present.
* When populating `tied_to_logs` and `likely_root_causes`, **explicitly mention how specific actions or patterns within the 'Sleep Well', 'Move Well', 'Eat Well', and 'Recover Well' pillars (derived from the `Category` column and associated entries like `Time`, `Feeling`, `Earning Balance`, `Losing Balance`) contributed to (or detracted from) hormonal balance.** For example, 'short sleep duration (Sleep Well) on average X hours, contributing to losing balance' or 'regular mindful eating (Eat Well) from the `Earning Balance` column'.
* Identify `likely_root_causes` based on patterns observed across all data, linking them to specific pillars and `Losing Balance` factors where relevant.

**For 'monthly_overview_summary' and 'top_symptoms':**
* Utilize the `Mood`, `Symptoms`, and `Energy Level` columns from the "Logged Symptoms" section to populate `top_symptoms` and inform the `summary` and `health_reflection`.

Here is the required JSON object structure:
"""

MONTHLY_REPORT_JSON_STRUCTURE = """
{
  "monthly_overview_summary": {
    "summary": "string - Comprehensive summary of this month’s health trends. Mention key routines (e.g., average sleep duration, general stress level, typical movement patterns).",
    "top_symptoms": "array of strings - List the top 3-5 most reported symptoms this month.",
    "health_reflection": "string - Reflection on health gains or fluctuations based on the user's inputs (e.g., improvements in energy, increase in specific symptoms)."
  },
  "hormonal_balance_insight": {
    "score_or_message": "string - Overall hormonal balance status or message (e.g., 'Balanced', 'Slightly Imbalanced', 'At Risk', 'Needs Attention'). This should be a meaningful assessment based on the provided logs, especially 'Earning Balance' and 'Losing Balance' entries.",
    "tied_to_logs": "array of strings - Explain how this status is tied to specific logged data, explicitly referencing pillars like 'Sleep Well', 'Move Well', 'Eat Well', 'Recover Well' and factors from 'Earning Balance'/'Losing Balance' columns (e.g., 'short sleep duration (Sleep Well) on average X hours, noted under Losing Balance', 'consistent mindful eating (Eat Well) reported under Earning Balance').",
    "likely_root_causes": "array of strings - List 1-2 likely hormonal root causes identified from the logs, referencing associated pillars and 'Losing Balance' factors (e.g., 'high cortisol due to chronic stress (Recover Well), often noted in Losing Balance', 'insulin swings from inconsistent meal timing (Eat Well), possibly linked to 'Feeling' column entries')."
  },
  "logged_patterns": {
    "eat_well": {
      "did_well": "array of strings - Summarize what the user did well in the 'Eat Well' pillar (e.g., 'consistent whole food intake', 'regular meal timing', 'adequate hydration').",
      "areas_to_improve": "array of strings - Mention areas for improvement in the 'Eat Well' pillar (e.g., 'frequent consumption of processed foods', 'skipping meals', 'excessive caffeine intake').",
      "recommendations": "array of strings - Offer short, actionable recommendations based on real logs for this pillar.",
      "metrics": {
        "days_logged": "number - Total days Eat Well data was logged.",
        "average_meal_satisfaction_score": "number (0-5, or similar scale) - Average satisfaction with meals, *inferred from qualitative descriptions like 'Feeling' if no explicit score is given* (e.g., 3.5).",
        "processed_food_days_percentage": "number (0-100) - Percentage of days processed food consumption was noted, *inferred from relevant qualitative descriptions if no explicit data is given* (e.g., 40)."
      }
    },
    "sleep_well": {
      "did_well": "array of strings - Summarize what the user did well in the 'Sleep Well' pillar (e.g., 'consistent bedtime routine', 'adequate sleep duration on most nights').",
      "areas_to_improve": "array of strings - Mention areas for improvement in the 'Sleep Well' pillar (e.g., 'late-night screen exposure', 'inconsistent sleep schedule', 'alcohol before bed').",
      "recommendations": "array of strings - Offer short, actionable recommendations based on real logs for this pillar.",
      "metrics": {
        "days_logged": "number - Total days Sleep Well data was logged.",
        "average_sleep_quality_score": "number (0-5, or similar scale) - Average sleep quality, *inferred from qualitative descriptions like 'Feeling' or 'Earning Balance' if no explicit score is given* (e.g., 4.2).",
        "average_sleep_hours": "number - Average hours of sleep per night, *extracted directly from provided numerical columns/entries, specifically the 'Time' column* (e.g., 6.8).",
        "consistent_bedtime_percentage": "number (0-100) - Percentage of days with consistent bed/wake times, *inferred from qualitative descriptions or time patterns if no explicit data is given* (e.g., 75)."
      }
    },
    "move_well": {
      "did_well": "array of strings - Summarize what the user did well in the 'Move Well' pillar (e.g., 'regular walks', 'incorporation of strength training').",
      "areas_to_improve": "array of strings - Mention areas for improvement in the 'Move Well' pillar (e.g., 'prolonged sitting', 'lack of consistent exercise', 'over-exercising').",
      "recommendations": "array of strings - Offer short, actionable recommendations based on real logs for this pillar.",
      "metrics": {
        "days_logged": "number - Total days Move Well data was logged.",
        "activity_days_percentage": "number (0-100) - Percentage of days activity was logged, *inferred from relevant qualitative descriptions if no explicit data is given* (e.g., 60).",
        "average_activity_intensity_score": "number (0-5, or similar scale) - Average intensity of workouts, *inferred from qualitative descriptions if no explicit score is given* (e.g., 3)."
      }
    },
    "recover_well": {
      "did_well": "array of strings - Summarize what the user did well in the 'Recover Well' pillar (e.g., 'regular meditation', 'taking breaks during work').",
      "areas_to_improve": "array of strings - Mention areas for improvement in the 'Recover Well' pillar (e.g., 'high stress levels', 'overuse of social media', 'isolating from social connections').",
      "recommendations": "array of strings - Offer short, actionable recommendations based on real logs for this pillar.",
      "metrics": {
        "days_logged": "number - Total days Recover Well data was logged.",
        "average_stress_level_score": "number (0-5, or similar scale) - Average reported stress level, *inferred from qualitative descriptions like 'Feeling' or 'Mood' if no explicit score is given* (e.g., 3.8).",
        "recovery_activity_days_percentage": "number (0-100) - Percentage of days recovery activity was noted, *inferred from relevant qualitative descriptions if no explicit data is given* (e.g., 50)."
      }
    }
  },
  "root_cause_tags": [
    // array of objects - Based on logs and symptoms, tag their status and explain contribution.
    // Each object in this array should have the following structure:
    // {
    //   "tag": "string - e.g., '🧠 Stress-related hormonal disruption', '⚖️ Blood sugar instability', '🔥 Inflammatory response'",
    //   "explanation": "string - A sentence explaining how their routines contributed to that pattern."
    // }
  ],
  "actionable_next_steps": {
    "behavior_goals": "array of strings - Top 3 behavior goals personalized to their logs (e.g., 'improve sleep consistency by X hours', 'regulate meal timing to reduce insulin spikes', 'reduce caffeine intake to minimize anxiety').",
    "encouragement_message": "string - One encouraging sentence to motivate progress."
  }
}
"""

def clean_json_string(json_string):
    """Removes leading/trailing markdown code blocks and illegal trailing commas from a string."""
    if not isinstance(json_string, str):
        return json_string

    stripped_string = json_string.strip()

    # Remove markdown code blocks (```json or ```)
    if stripped_string.startswith('```json'):
        stripped_string = stripped_string[len('```json'):].lstrip()
    elif stripped_string.startswith('```'):
        stripped_string = stripped_string[len('```'):].lstrip()
    if stripped_string.endswith('```'):
        stripped_string = stripped_string[:-len('```')].rstrip()

    # Remove trailing commas within JSON objects and arrays
    stripped_string = re.sub(r',\s*}', '}', stripped_string)
    stripped_string = re.sub(r',\s*]', ']', stripped_string)

    return stripped_string

# --- Main Script Logic ---
def main():
    print("Bewell AI Monthly Health Reporter (Plain Python Script)")
    print("Enter file paths for your monthly health data.")

    # --- File Input Section ---
    previous_lab_report_path = input("Enter path to Previous Lab Report (optional, press Enter to skip): ").strip()
    daily_logs_path = input("Enter path to Daily Logs (symptoms, routine - required): ").strip()
    weekly_assessments_path = input("Enter path to Weekly Self-Assessments (optional, press Enter to skip): ").strip()

    # --- Process Files and Get Texts ---
    raw_lab_report_text = extract_text_from_file(previous_lab_report_path) if previous_lab_report_path else ""
    raw_daily_logs_text = extract_text_from_file(daily_logs_path)
    raw_weekly_assessments_text = extract_text_from_file(weekly_assessments_path) if weekly_assessments_path else ""

    # --- Validate daily logs file is provided ---
    if not daily_logs_path or raw_daily_logs_text.startswith("[Error"):
        print("Error: Daily Logs file is required for the monthly report and must be readable. Exiting.")
        return

    # --- Error Handling for File Processing ---
    if raw_lab_report_text.startswith("[Error"):
        print(f"Warning: Problem with Previous Lab Report: {raw_lab_report_text}")
        print("Continuing analysis without previous lab report data.")
        raw_lab_report_text = ""

    if raw_weekly_assessments_text.startswith("[Error"):
        print(f"Warning: Problem with Weekly Self-Assessments: {raw_weekly_assessments_text}")
        print("Continuing analysis without weekly self-assessment data.")
        raw_weekly_assessments_text = ""

    # --- Analyze Data ---
    if not api_key:
        print("Error: Please set your Google/Gemini API key. Exiting.")
        return

    print("Generating monthly report... (This may take a few moments)")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=model_name)

        # Prepare data sections for the prompt
        previous_lab_report_section = f"The user's initial health analysis from a prior lab report:\n{raw_lab_report_text}\n" if raw_lab_report_text else "No prior lab report data provided.\n"
        daily_logs_section = f"The user’s daily logs on symptoms and lifestyle (eat/sleep/move/recover):\n{raw_daily_logs_text}\n"
        weekly_assessments_section = f"Weekly self-assessments (e.g., stress, symptoms, effort):\n{raw_weekly_assessments_text}\n" if raw_weekly_assessments_text else "No weekly self-assessment data provided.\n"

        # Format the prompt
        prompt = MONTHLY_REPORT_PROMPT_INSTRUCTIONS.format(
            previous_lab_report_data=previous_lab_report_section,
            daily_logs_data=daily_logs_section,
            weekly_assessments_data=weekly_assessments_section
        ) + MONTHLY_REPORT_JSON_STRUCTURE

        response = model.generate_content(prompt, generation_config=generation_config)

        # --- Process and Display Results ---
        if response and response.text is not None:
            cleaned_json_string = clean_json_string(response.text)
            if cleaned_json_string.strip():
                try:
                    analysis_data = json.loads(cleaned_json_string)
                    print("\n✨ Your Personalized Bewell Monthly Health Report:")
                    print(json.dumps(analysis_data, indent=2)) # Pretty print JSON
                except json.JSONDecodeError as e:
                    print(f"Error: Failed to parse AI response as JSON. The response was not valid JSON: {e}")
                    print("The model returned text, but it was not valid JSON. Here is the raw text from the AI to help with debugging:")
                    print(response.text)
            else:
                print("Error: The model returned an empty response.")
        elif response and response.prompt_feedback:
            print(f"Analysis request was blocked. Reason: {response.prompt_feedback.block_reason}")
            if response.prompt_feedback.safety_ratings:
                print(f"Safety Ratings: {response.prompt_feedback.safety_ratings}")
            print("Please review your input and try again.")
        else:
            print("Error: Failed to generate a valid response from the model.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please check your API key, network connection, and input files.")

    # --- Display Raw Input for Debugging ---
    show_raw_inputs = input("\nDo you want to show raw inputs sent to AI for debugging? (yes/no): ").strip().lower()
    if show_raw_inputs == 'yes':
        print("\n--- Raw Inputs Sent to AI (for debugging) ---")
        print("Previous Lab Report Text:")
        print(raw_lab_report_text if raw_lab_report_text else "N/A")
        print("\nDaily Logs Text:")
        print(raw_daily_logs_text if raw_daily_logs_text else "N/A")
        print("\nWeekly Self-Assessments Text:")
        print(raw_weekly_assessments_text if raw_weekly_assessments_text else "N/A")
        print("\nFull Prompt Sent to AI:")
        print(prompt)

if __name__ == "__main__":
    main()