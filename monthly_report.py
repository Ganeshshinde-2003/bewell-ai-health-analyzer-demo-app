import streamlit as st
import google.generativeai as genai
import os
import fitz  # PyMuPDF - for PDF
import io  # To handle file bytes
import pandas as pd  # for Excel and CSV
from docx import Document  # for .docx files
import json
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Bewell AI Monthly Health Reporter",
    page_icon="🗓️",
    layout="wide"
)

# --- Text Extraction Function (Handles multiple file types) ---
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

    if not text.strip() and file_extension not in [".txt", ".csv"] and not text.startswith("["):
        st.warning(f"No readable text extracted from '{uploaded_file.name}'. The file might be scanned, empty, or have complex formatting. Consider pasting the text manually.")

    return text

# --- API Key Handling ---
api_key = "AIzaSyArQ9zeya1SO-IwsMappkLStXYT0W7WXfk" # Ensure your actual API key is handled securely
if not api_key:
    # This section is if you plan to use Streamlit Cloud secrets or environment variables
    # For local development, directly assigning the key here is fine for testing.
    # For deployment, remove the hardcoded key and rely on secrets/env vars.
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.warning("Please add your Google/Gemini API key to your environment variables (e.g., in a .env file as GOOGLE_API_KEY), Streamlit secrets, or paste it below.")
        # Fallback to text input in the app if key is not found elsewhere
        api_key = st.text_input("Or paste your Google/Gemini API Key here:", type="password")


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
- Use **C1[text]C1** for critical information and action items in descriptive text fields only (e.g., **C1[fatigue]C1**, **C1[eat more fiber]C1**).
- Use **C2[text]C2** for notable values or alerts in descriptive text fields only (e.g., **C2[low DHEA alert]C2**).
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
    st.title("Bewell AI Monthly Health Reporter-Final")
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
        st.error("Daily Logs file is required for the monthly report. Please upload a valid file.")
        return

    # --- Error Handling for File Processing ---
    if raw_daily_logs_text.startswith("[Error"):
        st.error(f"Error processing Daily Logs: {raw_daily_logs_text}")
        return

    if raw_lab_report_text.startswith("[Error"):
        st.warning(f"Warning: Problem with Previous Lab Report: {raw_lab_report_text}")
        st.write("Continuing analysis without previous lab report data.")
        raw_lab_report_text = ""

    if raw_weekly_assessments_text.startswith("[Error"):
        st.warning(f"Warning: Problem with Weekly Self-Assessments: {raw_weekly_assessments_text}")
        st.write("Continuing analysis without weekly self-assessment data.")
        raw_weekly_assessments_text = ""

    # --- Analyze Data and Display Results on Button Click---
    if st.button("Generate Monthly Report"):
        if not api_key:
            st.error("Please provide a Google/Gemini API key to proceed.")
            return

        with st.spinner("Generating monthly report... (This may take a few moments)"):
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
                            st.header("✨ Your Personalized Bewell Monthly Health Report:")
                            st.json(analysis_data)
                             # Display raw response after JSON
                            st.subheader("Raw Response from Gemini API (Post-Markdown Removal):")
                            st.text(cleaned_json_string)
                        except json.JSONDecodeError as e:
                            st.error(f"Error: Failed to parse AI response as JSON. The response was not valid JSON: {e}")
                            st.write("The model returned text, but it was not valid JSON. Here is the raw text from the AI to help with debugging:")
                            st.text(response.text)
                    else:
                        st.error("Error: The model returned an empty response.")
                elif response and response.prompt_feedback:
                    st.error(f"Analysis request was blocked. Reason: {response.prompt_feedback.block_reason}")
                    if response.prompt_feedback.safety_ratings:
                        st.write(f"Safety Ratings: {response.prompt_feedback.safety_ratings}")
                    st.write("Please review your input and try again.")
                else:
                    st.error("Error: Failed to generate a valid response from the model.")

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.write("Please check your API key, network connection, and input files.")


if __name__ == "__main__":
    main()