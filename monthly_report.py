import streamlit as st
import os
import json # Import json to parse the secret string
import fitz # PyMuPDF - for PDF
import io # To handle file bytes
import pandas as pd # for Excel and CSV
from docx import Document # for .docx files
import re
import time # Added for retry delay

# --- NEW: Vertex AI Imports ---
import vertexai
from vertexai.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold
from google.oauth2 import service_account

# --- Page Configuration ---
st.set_page_config(
    page_title="Bewell AI Monthly Health Reporter - HIPAA by Vertex AI", # Updated title
    page_icon="🗓️",
    layout="wide"
)

# --- Vertex AI Project and Location (from Streamlit Secrets) ---
# IMPORTANT: Replace "gen-lang-client-0208209080" with your actual Google Cloud Project ID
# or ensure it's set in your Streamlit secrets.
PROJECT_ID = st.secrets.get("PROJECT_ID", "gen-lang-client-0208209080")
LOCATION = st.secrets.get("LOCATION", "us-central1")

# --- Vertex AI Initialization ---
try:
    # Attempt to retrieve credentials from Streamlit secrets
    credentials_json_string = st.secrets.get("google_credentials")

    if not credentials_json_string:
        st.error("Google Cloud credentials not found in Streamlit secrets. Please configure 'google_credentials'.")
        st.stop() # Stop the app if credentials are not found

    # Parse the JSON string into a dictionary and create credentials object
    credentials_dict = json.loads(credentials_json_string)
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)

    # Initialize Vertex AI with the project, location, and credentials
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
    st.success("✅ Vertex AI initialized successfully!")

except Exception as e:
    # Display an error message if Vertex AI initialization fails
    st.error(f"❌ Failed to initialize Vertex AI. Please check your Streamlit secrets and project settings.\n\nError: {e}")
    st.stop() # Stop the app on initialization failure

# --- Gemini Model Configuration for Vertex AI ---
# Use a Vertex AI compatible model name. "gemini-1.5-flash-001" or "gemini-1.5-pro-001"
# are generally stable choices. "gemini-2.5-flash-lite" might be a preview model.
MODEL_NAME = "gemini-2.5-flash-lite" # Changed to a stable Vertex AI compatible model name for reliability

# Configuration for how the Gemini model generates content
generation_config = {
    "temperature": 0.2, # Lower temperature for more deterministic (JSON) output
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192, # Increased token limit for potentially large JSON responses
    "response_mime_type": "application/json", # CRITICAL: Instructs the model to output strict JSON
}

# Safety settings to block potentially harmful content
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# Instantiate the Vertex AI Gemini Model globally.
# This model object will be reused across all API calls, improving efficiency.
model = GenerativeModel(MODEL_NAME, generation_config=generation_config, safety_settings=safety_settings)

# --- End NEW Vertex AI Configuration ---


# --- Text Extraction Function (Handles multiple file types) ---
# Caching this function with @st.cache_data prevents re-running it if the input file
# content hasn't changed, optimizing performance.
@st.cache_data(show_spinner=False)
def extract_text_from_file(uploaded_file):
    """
    Extracts text content from various file types (PDF, DOCX, XLSX, XLS, TXT, CSV)
    from a Streamlit UploadedFile object.
    Returns extracted text as a string or an error marker.
    """
    if uploaded_file is None:
        return ""

    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    text = ""

    try:
        file_bytes = uploaded_file.getvalue()

        if file_extension == ".pdf":
            try:
                pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
                for page_num in range(pdf_document.page_count):
                    page = pdf_document.load_page(page_num)
                    text += page.get_text("text") + "\n"
                pdf_document.close()
            except Exception as e:
                st.error(f"Error processing PDF file: {e}")
                return "[Error Processing PDF File]"

        elif file_extension == ".docx":
            try:
                doc = Document(io.BytesIO(file_bytes))
                for para in doc.paragraphs:
                    text += para.text + "\n"
            except Exception as e:
                st.error(f"Error processing DOCX file: {e}")
                return "[Error Processing DOCX File]"

        elif file_extension in [".xlsx", ".xls"]:
            try:
                excel_data = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
                for sheet_name, df in excel_data.items():
                    text += f"--- Sheet: {sheet_name} ---\n"
                    text += df.to_string(index=False) + "\n\n"
            except Exception as excel_e:
                st.warning(f"Could not read Excel file '{uploaded_file.name}'. Ensure 'openpyxl' (for .xlsx) or 'xlrd' (for .xls) is installed if needed. Error: {excel_e}")
                return f"[Could not automatically process Excel file {uploaded_file.name} - Please try pasting the text or saving as PDF/TXT.]"

        elif file_extension in [".txt", ".csv"]:
            text = file_bytes.decode('utf-8', errors='ignore')

        else:
            st.warning(f"Unsupported file type uploaded: {file_extension} for file '{uploaded_file.name}'")
            return "[Unsupported File Type Uploaded]"

    except Exception as e:
        st.error(f"An error occurred while processing '{uploaded_file.name}': {e}")
        return "[Error Processing File]"

    # Provide a warning if no text could be extracted from a non-text file type
    if not text.strip() and file_extension not in [".txt", ".csv"] and not text.startswith("["):
        st.warning(f"No readable text extracted from '{uploaded_file.name}'. The file might be scanned, empty, or have complex formatting. Consider pasting the text manually.")

    return text

# --- The Main Prompt Construction for Monthly Report ---
# This multi-line string defines the role, tone, objective, and critical instructions
# for the Gemini model to generate the monthly health report.
MONTHLY_REPORT_PROMPT_INSTRUCTIONS = """
Role: You are the Bewell AI Monthly Health Report Assistant, a holistic health expert focused on women’s hormonal balance, precision medicine, and root-cause insights.
Tone: Empathetic, empowering, accessible, science-backed — never condescending. Avoid jargon.

Objective: Generate a comprehensive, personalized, science-backed monthly health summary for the user, including a radar chart visualizing scores for four pillars: Eat Well, Sleep Well, Move Well, and Recover Well. Synthesize the following data:
{previous_lab_report_data}
{daily_logs_data}
{weekly_assessments_data}

Instructions for Output:
- Generate ONE complete response as a JSON object matching the structure provided below.
- Do NOT include introductory or concluding text outside the JSON object.
- Do NOT wrap the JSON in markdown code blocks (```json ... ``` or ``` ... ```).
- Populate all fields based *only* on the provided user data. If data is missing or invalid, use default values (e.g., 0 for metrics, empty arrays for lists) and note limitations in the `summary` field.
- For fields specified as 'array of strings' (e.g., `tied_to_logs`, `likely_root_causes`), ensure each entry is a distinct string in the array. Do NOT combine multiple points into a single string. Each entry should be concise and focused on a single log or cause.

**CRITICAL TEXT MARKING FOR HIGHLIGHTING**
- Use **C1[text]C1** to highlight critical information and action items in descriptive text fields only (e.g., **C1[fatigue]C1**, **C1[eat more fiber]C1**).
- Use **C2[text]C2** to highlight notable values or alerts in descriptive text fields only (e.g., **C2[low DHEA alert]C2**).
- Apply highlighting ONLY to: 'summary', 'health_reflection', 'score_or_message', 'tied_to_logs', 'likely_root_causes', 'Guidance' (in all pillars), 'explanation' (in root_cause_tags), 'food_to_enjoy', 'food_to_limit', 'rest_and_recovery', 'daily_habits', 'movements', 'behavior_goals', 'encouragement_message', 'key_behaviors' (in radar_chart_data).
- Do NOT highlight: 'top_symptoms', 'did_well', 'areas_to_improve', 'recommendations', 'tag' (in root_cause_tags), 'radar_chart_data.scores', 'radar_chart_data.label', 'radar_chart_data.caption', or single-value fields like 'Estradiol'.
- Apply sparingly for clarity (e.g., **C1[constipation]C1**, **C2[high stress]C2**).

**Daily Logs Structure:**
The daily logs contain two sections:
1. **Logged Routines** (columns: `Date`, `Category` [Eat Well, Sleep Well, Move Well, Recover Well], `Time` [sleep hours or meal times], `Meal_Order`, `Image`, `Feeling`, `Earning Balance`, `Losing Balance`).
2. **Logged Symptoms** (columns: `Date`, `Mood`, `Symptoms`, `Energy Level`).

**Analysis Instructions:**
- **Numerical Metrics**: Extract explicit numerical values from columns like `Time` (e.g., sleep hours). If missing, infer a score (0-5, 5 best) from qualitative columns (`Feeling`, `Earning Balance`, `Losing Balance`, `Mood`, `Energy Level`). E.g., "Slept soundly" → 4, "Nauseous" → 1. If no data, use 0 or null and note in `summary` (e.g., "Limited **C2[Eat Well data]C2** detected").
- **Hormonal Balance Insight**: Analyze `Earning Balance` and `Losing Balance` for hormonal impact, using `Feeling` for context. `score_or_message`: Provide a status (e.g., "**C1[Balanced]C1**" or "**C1[Needs Attention]C1** due to **C2[high stress]C2**"). `tied_to_logs`: List specific log entries as separate strings (e.g., ["**C1[short sleep hours]C1** in Sleep Well", "**C1[high stress]C1** in Recover Well"]). `likely_root_causes`: List 1-2 distinct causes as separate strings (e.g., ["**C2[high stress]C2** from Recover Well", "**C1[irregular meal timing]C1** in Eat Well"]).
- **Monthly Overview and Symptoms**: Use `Mood`, `Symptoms`, `Energy Level` to populate `top_symptoms` (no highlighting) and inform `summary`/`health_reflection` with highlights (e.g., "**C1[fatigue]C1**").
- **Radar Chart Data**: Follow the scoring instructions provided in the JSON structure below.

Here is the required JSON object structure:
"""

# This multi-line string defines the precise JSON structure the Gemini model is expected to output.
# It acts as a schema for the AI's response.
MONTHLY_REPORT_JSON_STRUCTURE = """
{
  "monthly_overview_summary": {
    "summary": "string - A clear overview of your health trends this month, like how much you slept on average, your stress levels, and your typical movement patterns. Highlight critical trends like **C1[low energy]C1** or alerts like **C2[irregular sleep patterns]C2**.",
    "top_symptoms": "array of strings - The top 3-5 symptoms you reported most often this month, e.g., ['fatigue', 'mood swings', 'headaches'].",
    "health_reflection": "string - A reflection on how your health improved or fluctuated based on what you logged, like boosts in energy or changes in **C1[symptoms]C1** such as **C2[reduced fatigue]C2**."
  },
  "hormonal_balance_insight": {
    "score_or_message": "string - Your overall hormonal balance status, like '**C1[Balanced]C1**', '**C1[Slightly Imbalanced]C1**', '**C1[At Risk]C1**', or '**C1[Needs Attention]C1**', based on your logs, especially what you noted in 'Earning Balance' and 'Losing Balance'.",
    "tied_to_logs": "array of strings - How your status connects to specific logs, like '**C1[short sleep hours]C1** in Sleep Well from Losing Balance (**C2[5.5 hours average]C2**)' or 'consistent **C1[mindful eating]C1** in Eat Well from Earning Balance'.",
    "likely_root_causes": "array of strings - 1-2 likely causes of hormonal changes based on your logs, like '**C2[high stress]C2** from Recover Well noted in Losing Balance' or '**C1[irregular meal timing]C1** in Eat Well from your Feeling notes'."
  },
  "logged_patterns": {
    "eat_well": {
      "did_well": "array of strings - Things you nailed in the Eat Well pillar, like 'eating whole foods consistently' or 'sticking to regular meal times'.",
      "areas_to_improve": "array of strings - Areas to work on in the Eat Well pillar, like 'cutting back on processed foods' or 'not skipping meals'.",
      "recommendations": "array of strings - Simple, actionable tips based on your eating logs, e.g., 'add more fiber to support digestion'.",
      "Guidance": "array of strings - 2-4 sentences offering practical advice to enhance your eating habits, like 'Try adding a variety of **C1[colorful vegetables]C1** to your meals to boost nutrient intake.' or 'Plan your meals ahead to avoid **C1[skipping them]C1** on busy days.'",
      "metrics": {
        "days_logged": "number - Total days you logged Eat Well data.",
        "average_meal_satisfaction_score": "number (0-5) - Your average satisfaction with meals, inferred from your Feeling notes if no score is given (e.g., 3.5).",
        "processed_food_days_percentage": "number (0-100) - Percentage of days you noted processed food, inferred from your logs if not explicit (e.g., 40)."
      }
    },
    "sleep_well": {
      "did_well": "array of strings - What you did great in the Sleep Well pillar, like 'keeping a consistent bedtime' or 'getting enough sleep most nights'.",
      "areas_to_improve": "array of strings - Areas to improve in Sleep Well, like 'reducing late-night screen time' or 'avoiding alcohol before bed'.",
      "recommendations": "array of strings - Simple, actionable tips based on your sleep logs, e.g., 'set a consistent bedtime'.",
      "Guidance": "array of strings - 2-4 sentences offering practical advice to improve your sleep, like 'Set a **C1[consistent bedtime routine]C1** to signal your body it’s time to rest.' or 'Limit **C1[screen time]C1** an hour before bed to improve **C2[sleep quality]C2**.'",
      "metrics": {
        "days_logged": "number - Total days you logged Sleep Well data.",
        "average_sleep_quality_score": "number (0-5) - Your average sleep quality, inferred from Feeling or Earning Balance if no score is given (e.g., 4.2).",
        "average_sleep_hours": "number - Your average sleep hours per night, taken from the Time column (e.g., 6.8).",
        "consistent_bedtime_percentage": "number (0-100) - Percentage of days with consistent bed/wake times, inferred from your logs if not explicit (e.g., 75)."
      }
    },
    "move_well": {
      "did_well": "array of strings - What you excelled at in the Move Well pillar, like 'regular walks' or 'adding strength training'.",
      "areas_to_improve": "array of strings - Areas to work on in Move Well, like 'sitting too long' or 'not exercising consistently'.",
      "recommendations": "array of strings - Simple, actionable tips based on your movement logs, e.g., 'add short walks'.",
      "Guidance": "array of strings - 2-4 sentences offering practical advice to enhance your movement, like 'Incorporate **C1[short walks]C1** during your workday to reduce **C2[sedentary time]C2**.' or 'Try a weekly **C1[yoga session]C1** to improve flexibility and reduce **C1[stress]C1**.'",
      "metrics": {
        "days_logged": "number - Total days you logged Move Well data.",
        "activity_days_percentage": "number (0-100) - Percentage of days you logged activity, inferred from your logs if not explicit (e.g., 60).",
        "average_activity_intensity_score": "number (0-5) - Your average workout intensity, inferred from your logs if no score is given (e.g., 3)."
      }
    },
    "recover_well": {
      "did_well": "array of strings - Things you did well in the Recover Well pillar, like 'meditating regularly' or 'taking work breaks'.",
      "areas_to_improve": "array of strings - Areas to improve in Recover Well, like 'managing high stress' or 'cutting back on social media'.",
      "recommendations": "array of strings - Simple, actionable tips based on your recovery logs, e.g., 'try 5-minute breathing exercises'.",
      "Guidance": "array of strings - 2-4 sentences offering practical advice to boost your recovery, like 'Schedule **C1[short breaks]C1** during your day to recharge and lower **C1[stress]C1**.' or 'Try a **C1[5-minute breathing exercise]C1** to manage **C2[anxiety]C2**.'",
      "metrics": {
        "days_logged": "number - Total days you logged Recover Well data.",
        "average_stress_level_score": "number (0-5) - Your average stress level, inferred from Feeling or Mood if no score is given (e.g., 3.8).",
        "recovery_activity_days_percentage": "number (0-100) - Percentage of days you noted recovery activities, inferred from your logs if not explicit (e.g., 50)."
      }
    }
  },
  "root_cause_tags": [
    {
      "tag": "string - e.g., 'Stress-related hormonal disruption', 'Blood sugar instability', 'Inflammatory response'",
      "explanation": "string - A sentence explaining how your routines contributed to this pattern, e.g., 'Your **C1[high stress]C1** from Recover Well logs led to **C2[hormonal disruption]C2**.'"
    }
  ],
  "actionable_next_steps": {
    "food_to_enjoy": "array of strings - Specific foods or eating habits to embrace based on your logs, like '**C1[leafy greens]C1** for nutrient density' or '**C1[healthy fats like avocado]C1** to support hormonal balance'.",
    "food_to_limit": "array of strings - Foods or eating habits to cut back on based on your logs, like '**C1[sugary snacks]C1** to stabilize **C2[blood sugar]C2**' or '**C1[excessive caffeine]C1** to reduce **C2[anxiety]C2**'.",
    "rest_and_recovery": "array of strings - Actions to prioritize for rest and recovery based on your logs, like '**C1[try 10-minute daily meditation]C1**' or '**C1[schedule downtime]C1** to lower **C2[stress]C2**'.",
    "daily_habits": "array of strings - Small daily habits to build based on your logs, like '**C1[drink water first thing in the morning]C1**' or '**C1[set a consistent bedtime routine]C1**'.",
    "movements": "array of strings - Movement practices to incorporate based on your logs, like '**C1[add 15-minute daily walks]C1**' or '**C1[try low-impact yoga]C1** for flexibility'.",
    "behavior_goals": "array of strings - Your top 3 behavior goals based on your logs, like '**C1[get 7-8 hours of sleep nightly]C1**' or '**C1[eat meals at regular times]C1** to balance **C2[insulin]C2**'.",
    "encouragement_message": "string - A motivating note to keep you going, like 'You're making great strides—keep prioritizing your **C1[health]C1**!'"
  },
  "radar_chart_data": {
    "scores": {
      "eat_well": "number - Score from 1 to 10 based on scoring logic (e.g., 3.33).",
      "sleep_well": "number - Score from 1 to 10 based on scoring logic (e.g., 1.67).",
      "move_well": "number - Score from 1 to 10 based on scoring logic (e.g., 6.67).",
      "recover_well": "number - Score from 1 to 10 based on scoring logic (e.g., 1.67)."
    },
    "key_behaviors": {
      "eat_well": "array of strings - 1-3 behaviors impacting the score, e.g., ['Logged **C1[fiber-rich meals]C1** 4 days/week', 'Reported **C2[sugar cravings]C2** 3 times'].",
      "sleep_well": "array of strings - 1-3 behaviors impacting the score, e.g., ['Averaged **C2[5.5 hours sleep]C2**', 'Logged **C1[consistent bedtime]C1**'].",
      "move_well": "array of strings - 1-3 behaviors impacting the score, e.g., ['Logged **C1[weekly workouts]C1** 3 times', 'Reported **C2[sedentary days]C2**'].",
      "recover_well": "array of strings - 1-3 behaviors impacting the score, e.g., ['Practiced **C1[meditation]C1** 4 days/week', 'Reported **C2[high stress]C2**']."
    },
    "label": "string - Overall label based on average score: 'Aligned' (9-10), 'On Track' (7-8), 'Needs Attention' (5-6), 'At Risk' (<5).",
    "caption": "string - Caption based on average score, e.g., 'Your routines are deeply supportive of hormonal health.'",
  }
}
"""

# --- Modified call_gemini_with_retry for Vertex AI ---
def call_gemini_with_retry(prompt, max_retries=3):
    """
    Calls the Vertex AI Gemini model with a prompt, handling retries for JSON errors.
    Uses the globally defined 'model' object.
    Returns parsed JSON data and the raw response string, or raises an exception.
    """
    raw_response_for_debugging = ""
    for attempt in range(max_retries):
        if attempt > 0:
            st.info(f"Attempt {attempt + 1} for AI response is ongoing...")
            time.sleep(2) # Small delay before retrying the call

        try:
            # Use the global 'model' directly
            response = model.generate_content(prompt)
            full_response_text = response.text
            raw_response_for_debugging = full_response_text

            if full_response_text:
                cleaned_json_string = clean_json_string(full_response_text)
                return json.loads(cleaned_json_string), raw_response_for_debugging
            else:
                raise ValueError("Model returned an empty response.")

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed (Invalid JSON). Retrying...")
            else:
                st.error(f"Attempt {attempt + 1} failed. Failed to get valid JSON after {max_retries} attempts: {e}")
                # For debugging, show the raw response that caused the JSON error
                st.code(raw_response_for_debugging, language='json', label="Raw response causing JSON error (for debugging)")
                raise Exception(f"Failed to get valid JSON after {max_retries} attempts: {e}")
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed (Unexpected error: {e}). Retrying...")
            else:
                st.error(f"Attempt {attempt + 1} failed. An unexpected error occurred after {max_retries} attempts: {e}")
                raise
    raise Exception("Max retries reached without a successful response.")


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

# --- Main Streamlit App ---
def main():
    st.title("Bewell AI Monthly Health Reporter – HIPAA Secure by Bewell + Vertex AI") # Updated title
    st.write("Upload your monthly health data (daily logs, self-assessments) and optionally a previous lab report for a comprehensive monthly analysis.")

    # --- File Upload Section ---
    previous_lab_report = st.file_uploader("Upload Previous Lab Report (optional)", type=[".pdf", ".docx", ".xlsx", ".xls", ".txt", ".csv"])
    daily_logs = st.file_uploader("Upload Daily Logs (symptoms, routine - required)", type=[".pdf", ".docx", ".xlsx", ".xls", ".txt", ".csv"])
    weekly_assessments = st.file_uploader("Upload Weekly Self-Assessments (stress, symptoms, effort - optional)", type=[".pdf", ".docx", ".xlsx", ".xls", ".txt", ".csv"])

    # --- Process Files and Get Texts ---
    raw_lab_report_text = extract_text_from_file(previous_lab_report) if previous_lab_report else ""
    raw_daily_logs_text = extract_text_from_file(daily_logs)
    raw_weekly_assessments_text = extract_text_from_file(weekly_assessments) if weekly_assessments else ""

    # --- Validate daily logs file is provided ---
    if not daily_logs:
        st.info("Daily Logs file is required for the monthly report. Please upload a valid file to proceed.")
        st.stop()

    # --- Error Handling for File Processing ---
    if raw_daily_logs_text.startswith("[Error"):
        st.error(f"Error processing Daily Logs: {raw_daily_logs_text}")
        st.stop() # Stop execution if critical file cannot be processed

    if raw_lab_report_text.startswith("[Error"):
        st.warning(f"Warning: Problem with Previous Lab Report: {raw_lab_report_text}")
        st.write("Continuing analysis without previous lab report data.")
        raw_lab_report_text = "" # Clear content if there was an error

    if raw_weekly_assessments_text.startswith("[Error"):
        st.warning(f"Warning: Problem with Weekly Self-Assessments: {raw_weekly_assessments_text}")
        st.write("Continuing analysis without weekly self-assessment data.")
        raw_weekly_assessments_text = "" # Clear content if there was an error

    # --- Analyze Data and Display Results on Button Click---
    if st.button("Generate Monthly Report ✨", type="primary"):
        with st.status("Generating monthly report...", expanded=True) as status_message_box:
            status_message_box.write("⚙️ Preparing data for AI analysis...")
            try:
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

                status_message_box.write("🧠 Sending data to Bewell AI (Vertex AI Gemini)...")
                # Use the call_gemini_with_retry function
                analysis_data, raw_response_for_debugging = call_gemini_with_retry(prompt)

                # --- Process and Display Results ---
                if analysis_data:
                    # Removed 'icon' as it's not supported in older Streamlit versions for .update()
                    status_message_box.update(label="Monthly Report Generated!", state="complete")
                    st.header("✨ Your Personalized Bewell Monthly Health Report:")
                    st.json(analysis_data, expanded=True) # Display interactive JSON

                    st.markdown("---")
                    st.header("📋 Copy Full JSON Output (Plain Text):")
                    plain_json_string = json.dumps(analysis_data, indent=2)
                    st.text_area(
                        "Select the text below and copy it to your clipboard:",
                        plain_json_string,
                        height=400,
                        disabled=True
                    )
                else:
                    status_message_box.error("❌ No analysis data could be generated.")

            except Exception as e:
                status_message_box.error(f"❌ An error occurred during report generation: {e}")
                st.error("Please check your input files and ensure Vertex AI is correctly configured.")

        # --- Debugging Information Expander (MOVED OUTSIDE st.status) ---
        # This expander will now appear below the status message box after generation
        # It's crucial that this block is NOT indented under the 'with st.status(...)'
        # block, as that causes the "Expanders may not be nested" error.
        with st.expander("Show Debug Information (Raw Response & Prompt)"):
            st.subheader("Full Prompt Sent to Vertex AI:")
            st.code(prompt, language='markdown') # Changed to markdown for better readability of prompt
            st.subheader("Raw Response from Vertex AI (before JSON parsing):")
            st.code(raw_response_for_debugging, language='json') # Use json language hint


if __name__ == "__main__":
    main()