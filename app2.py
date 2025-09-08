import streamlit as st
import os
import json # Import json to parse the secret string
import fitz
import io
import pandas as pd
from docx import Document
import re
import time
import jsonschema


# --- NEW: Vertex AI Imports ---
import vertexai
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
from google.oauth2 import service_account 


# --- Page Configuration ---
st.set_page_config(
    page_title="Bewell AI Health Analyzer - HIPAA by Vertex AI - VP1",
    page_icon="🌿",
    layout="wide"
)


PROJECT_ID = st.secrets.get("PROJECT_ID") # Replace with your actual project ID or keep the one from secrets
LOCATION = st.secrets.get("LOCATION", "us-central1")


try:
    credentials_json_string = st.secrets.get("google_credentials")


    if not credentials_json_string:
        st.error("Google Cloud credentials not found in Streamlit secrets. Please configure 'google_credentials'.")
        st.stop()


    credentials_dict = json.loads(credentials_json_string)
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)  # ✅ Use credentials directly


    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)  # ✅ No temp file, no metadata
    st.success("✅ Vertex AI initialized successfully!")


except Exception as e:
    st.error(f"❌ Failed to initialize Vertex AI. Please check your Streamlit secrets and project settings.\n\nError: {e}")
    st.stop()


# The model name for Gemini 1.5 Pro on Vertex AI
# Use "gemini-1.5-pro" for the stable version.
# If you want the latest preview features (at your own risk), check Vertex AI documentation for current preview model names.
MODEL_NAME = "gemini-2.5-flash-lite" 


# --- Safety Settings (Recommended for production) ---
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}


# --- Gemini Model Generation Configuration ---
generation_config = {
    "temperature": 0.2,  # Low for deterministic JSON
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,  # Increased for large responses
    "response_mime_type": "application/json"  # Ensures strict JSON
}


# Instantiate the Vertex AI Gemini Model globally.
# This model object will be reused across all calls.
model = GenerativeModel(MODEL_NAME, generation_config=generation_config, safety_settings=safety_settings)


# --- End NEW Vertex AI Configuration ---



# --- Text Extraction Function (Handles multiple file types) ---
# @st.cache_data is okay here because file contents are hashable
@st.cache_data(show_spinner=False)
def extract_text_from_file(uploaded_file):
    """
    Extracts text content from various file types (PDF, DOCX, XLSX, XLS, TXT, CSV).
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
                st.warning(
                    f"Could not read Excel file '{uploaded_file.name}'. Ensure 'openpyxl' (for .xlsx) or 'xlrd' (for .xls) is installed. Error: {excel_e}")
                return f"[Could not automatically process Excel file {uploaded_file.name} - Please try pasting the text or saving as PDF/TXT.]"


        elif file_extension in [".txt", ".csv"]:
            text = file_bytes.decode('utf-8', errors='ignore')


        else:
            st.warning(f"Unsupported file type: {file_extension} for file '{uploaded_file.name}'")
            return "[Unsupported File Type Uploaded]"


    except Exception as e:
        st.error(f"An error occurred while processing '{uploaded_file.name}': {e}")
        return "[Error Processing File]"


    if not text.strip() and file_extension not in [".txt", ".csv"]:
        st.warning(
            f"No readable text extracted from '{uploaded_file.name}'. The file might be scanned, empty, or have complex formatting. Consider pasting the text manually.")
    return text


# --- Base Prompt Instructions (Common to all calls) ---
BASE_PROMPT_COMMON = """
Role & Persona:
You are the Bewell AI Assistant. Your persona is a blend of a holistic women's health expert, a functional medicine practitioner, and a precision medicine specialist, with a dedicated focus on women's care.


Tone & Voice:
Adopt an approachable, empathetic, supportive, clear, and accessible tone. Always speak directly to the user using "you" and "your." Use simple, beginner-friendly language and relatable analogies, as if you are explaining complex health concepts to someone with no prior health knowledge. Avoid technical jargon, and if a term is necessary, define it in simple, everyday words.   
Never use alarming or fear-based language. Always frame recommendations as empowering guidance focused on women’s health and well-being.



---
🔑 Your Input Data:
• Health Assessment Text: {health_assessment_text}
• Lab Report Text: {lab_report_section_placeholder}


---
⚠️ Critical Instructions & Output Format:
1. **JSON Output ONLY**: Generate ONE single, complete, and comprehensive JSON object that strictly follows the `JSON_STRUCTURE_DEFINITION` provided below. There must be NO other text, explanations, or markdown code blocks (``````) outside of this single JSON object.
2. **NO EXTRA KEYS**: You MUST NOT generate any keys or objects that are not explicitly defined in the provided `JSON_STRUCTURE_DEFINITION`.
3. **Comprehensive & Personalized Array Requirement**:
   • **NO GENERIC FILLERS**: Avoid generic placeholder text. Every item in every array must be personalized, actionable, and tied to the user’s actual data.   
   • **USEFUL GENERAL ADVICE ONLY IF NEEDED**: If no personalization is possible, give specific, practical advice (e.g., “Try 30 minutes of brisk walking daily to help balance hormones and improve energy.”).
4. **Text Highlighting Rules**:
   • Use **C1[text]C1** to highlight primary symptoms or critical action steps within descriptive text.   
   • Use **C2[text]C2** for specific biomarker results and values (e.g., “Your **C2[high cortisol]C2** (**C2[20.3 ug/dL]C2**) may be affecting your sleep.”).   
   • Do NOT use these markers in single-value fields like ‘name’ or ‘result.’


⚠️ ABSOLUTELY NO EXTRA KEYS OR OBJECTS
You must only generate a single JSON object that matches the exact structure shown in the active `JSON_STRUCTURE_DEFINITION` below.   
- Do NOT add any extra keys, arrays, objects, fields, sections, or nesting at any level.   
- The output must include ONLY the keys, arrays, and objects shown in the provided structure.   
- If you provide keys such as "supplements", "recommendation", "lab_reports", or any section not listed in the definition, your response is incorrect and will be rejected.
- No summary, explanation, or markdown code block should be present anywhere outside of the single allowed JSON object.


Your answer must be a valid JSON object and **match the provided structure exactly**. If your answer contains any extra, missing, or differently named keys, it will be rejected by the system.


---
🧠 Core Analysis & Content Requirements:



1. **Women-First Biomarker Analysis**   
   Every biomarker explanation must explicitly connect to women’s health, such as menstrual cycles, energy, fertility, weight, skin, mood, and longevity. Example: “TSH is slightly elevated, which can affect your energy levels, weight, and menstrual regularity.”



2. **Optimal vs Clinical Ranges**   
   Evaluate biomarkers against both standard reference ranges and functional *optimal ranges for women*.   
   - Estradiol, Progesterone, LH, and FSH → assess relative to cycle phase (follicular, ovulatory, luteal).   
   - Thyroid function (TSH, Free T3, Free T4) → flag subclinical changes when symptoms suggest impact.   
   - Iron/ferritin → highlight women’s optimal ranges, noting earlier risks for fatigue and hair loss.   



3. **Cycle-Phase Sensitivity**   
   If menstrual cycle phase is provided → interpret hormone results accordingly.   
   If not provided → explain how values may shift depending on cycle phase and suggest retesting at key points (“Consider retesting progesterone 7 days after ovulation for best accuracy.”).   



4. **Symptom-Biomarker Linking**   
   Every flagged biomarker must reference user’s reported symptoms. Example: “Your low estradiol may be contributing to your **C1[mood swings]C1** and **C1[irregular cycles]C1**.”   



5. **Empowering Educational Explanations**   
   Clearly explain the “why” behind guidance. Use simple analogies (e.g., “Think of cortisol as your body’s stress alarm—if it stays on too long, it can wear down your energy like a phone that never fully charges.”).   



6. **Precision Testing Guidance**   
   Provide specific, proactive retesting recommendations.   
   Example: “Retest Progesterone in mid-luteal phase (about day 21 of a 28-day cycle) to confirm support for a healthy luteal phase.”   
   Example: “If Ferritin remains below 50, retest after 3 months of dietary iron support.”   
"""


# --- JSON Structure Definitions for each part ---


JSON_STRUCTURE_BIOMARKERS_ORIG = """
{
  "lab_analysis": {
    "overall_summary": "string - Summarizes your health status using your lab results and health details in simple terms, as if explaining to someone new to health topics. Highlights your key issues and actions (e.g., 'Your **C1[tiredness]C1** and **C2[high thyroid hormone]C2** suggest checking your thyroid'). If no lab report, notes analysis uses only your health details. For unreadable lab data, states: 'Some of your lab results couldn’t be read due to file issues. Advice uses your available data.'",
    "biomarkers_tested_count": "integer - Count of 'detailed_biomarkers' objects. Verified to match the array length. Set to 0 if you provided no lab report.",
    "biomarker_categories_summary": {
      "description_text": "string - Summarizes your biomarker categories in simple terms (e.g., 'Out of **C1[{biomarkers_tested_count}]C1** tests, **C2[{optimal_count}]C2** are good, **C2[{keep_in_mind_count}]C2** need watching, and **C2[{attention_needed_count}]C2** need action'). Excludes 'Not Performed' tests.",
      "optimal_count": "integer - Count of your 'optimal' biomarkers. Set to 0 if no lab report.",
      "keep_in_mind_count": "integer - Count of your 'keep_in_mind' biomarkers. Set to 0 if no lab report.",
      "attention_needed_count": "integer - Count of your 'attention_needed' biomarkers. Set to 0 if no lab report."
    },
    "detailed_biomarkers": [
      {
        "name": "string - Full biomarker name (e.g., 'Estradiol').",
        "status": "string - 'optimal', 'keep_in_mind', or 'attention_needed'.",
        "status_label": "string - Simple label (e.g., 'Good (Green)' for optimal).",
        "result": "string - Your result with units (e.g., '4.310 uIU/mL').",
        "range": "string - Normal range (e.g., '0.450-4.500').",
        "cycle_impact": "string - Explains how the biomarker affects your menstrual cycle in simple terms (e.g., 'Estradiol changes during your cycle and may cause **C1[cramps]C1**').",
        "why_it_matters": "string - Explains the biomarker’s role in your health, linked to your data, in simple terms (e.g., 'High thyroid hormone may cause your **C1[tiredness]C1**, like a slow energy engine')."
      }
    ],
    "crucial_biomarkers_to_measure": [
      {
        "name": "string - Biomarker name (e.g., 'DHEA, Serum').",
        "importance": "string - Simple explanation of why you should test it (e.g., 'Test DHEA, Serum to check your stress levels because you feel **C1[tired]C1**')."
      }
    ],
    "health_recommendation_summary": ["string - Simple, actionable steps for you (e.g., 'Retest DHEA, Serum to understand your **C1[stress]C1**')."]
  }
}
"""

JSON_STRUCTURE_4PILLARS_ORIG = """
{
  "four_pillars": {
    "introduction": "string - Summarizes your health status and lab findings in simple terms for the four areas (eating, sleeping, moving, recovering).",
    "pillars": [
      {
        "name": "string - Pillar name (e.g., 'Eat Well').",
        "score": "integer - 1 (needs improvement) to 10 (great), based on your data (e.g., '4' for skipping meals).",
        "score_rationale": ["string - A clear explanation in simple language for why your score was given, tied to your data (e.g., ['Your Eat Well score is 4 because **C1[skipping meals]C1** means you miss energy.', 'Eating regularly helps your body stay strong.'])."],
        "why_it_matters": "string - Explains relevance to your data in simple terms (e.g., 'Good food helps balance your hormones for **C1[bloating]C1**, like fueling a car').",
        "root_cause_correlation": "string - Links to root causes in simple terms (e.g., 'Fiber helps your **C1[constipation]C1** caused by low estrogen').",
        "science_based_explanation": "string - Simple scientific basis (e.g., 'Fiber clears extra hormones to ease your **C1[mood swings]C1**, like cleaning out clutter').",
        "additional_guidance": {
          "description": "string - Explains guidance in simple terms, notes if general due to limited data (e.g., 'Since you provided no specific data, try these general tips for your health').",
          "structure": "object - The keys in this object will vary based on the pillar, as per the instructions below. The values are arrays of objects. Example: `{'recommended_foods': [{'name': '...', 'description': '...'}]}`"
        }
      }
    ]
  }
}
"""

JSON_STRUCTURE_SUPPLEMENTS_ACTIONS_ORIG = """
{
  "supplements": {
    "description": "string - Explains supplement advice in simple terms based on your lab or health data (e.g., 'These supplements may help with your **C1[tiredness]C1**').",
    "structure": {
      "recommendations": [
        {
          "name": "string - Supplement name (e.g., 'Magnesium').",
          "rationale": "string - Simple reason linked to your data (e.g., 'For your low Estradiol and **C1[mood swings]C1**, helps calm your body').",
          "expected_outcomes": "string - Simple benefits (e.g., 'Better **C1[sleep]C1**, like a restful night').",
          "dosage_and_timing": "string - Simple dosage (e.g., '200 mg daily, evening').",
          "situational_cyclical_considerations": "string - Simple cycle-specific advice (e.g., 'Use in the second half of your cycle for **C1[cramps]C1**')."
        }
      ],
      "conclusion": "string - Encourages you to stick to the advice and check with your doctor, in simple terms."
    }
  }
}
"""


def escape_braces(s):
    return s.replace("{", "{{").replace("}", "}}")


# Escape braces here so we can safely format prompts containing literal JSON
JSON_STRUCTURE_BIOMARKERS = escape_braces(JSON_STRUCTURE_BIOMARKERS_ORIG)
JSON_STRUCTURE_4PILLARS = escape_braces(JSON_STRUCTURE_4PILLARS_ORIG)
JSON_STRUCTURE_SUPPLEMENTS_ACTIONS = escape_braces(JSON_STRUCTURE_SUPPLEMENTS_ACTIONS_ORIG)


schema_biomarkers = json.loads(JSON_STRUCTURE_BIOMARKERS_ORIG)
schema_4pillars = json.loads(JSON_STRUCTURE_4PILLARS_ORIG)
schema_supplements = json.loads(JSON_STRUCTURE_SUPPLEMENTS_ACTIONS_ORIG)


def validate_json(data, schema):
    try:
        jsonschema.validate(instance=data, schema=schema)
        return True, ""
    except jsonschema.ValidationError as e:
        return False, e.message


def clean_json_string(json_string):
    """Removes markdown code blocks and attempts to fix common JSON errors."""
    if not isinstance(json_string, str):
        return json_string
    # Remove markdown code blocks
    stripped_string = re.sub(r'^``````\s*$', '', json_string.strip(), flags=re.MULTILINE)
    
    # Remove illegal trailing commas
    stripped_string = re.sub(r',\s*}', '}', stripped_string)
    stripped_string = re.sub(r',\s*]', ']', stripped_string)
    
    return stripped_string


# --- Modified call_gemini_with_retry for Vertex AI ---
# Removed @st.cache_data from here because the global 'model' is not cacheable as an argument.
# The 'model' is loaded once globally.
def call_gemini_with_retry(prompt, schema, max_retries=3):
    """
    Calls the Vertex AI Gemini model with a prompt, handling retries for JSON errors.
    Uses the globally defined 'model' object.
    Returns parsed JSON data and the raw response string, or raises an exception.
    """
    # 'model' is already globally defined, no need to pass it or declare global here
    raw_response_for_debugging = ""
    for attempt in range(max_retries):
        if attempt > 0:
            st.info(f"Attempt {attempt + 1} is ongoing...")
            time.sleep(1) # Small delay before retrying the call


        try:
            # Use the global 'model' directly
            response = model.generate_content(prompt) 
            full_response_text = response.text
            raw_response_for_debugging = full_response_text


            if not full_response_text:
                raise ValueError("Model returned an empty response.")


            cleaned_json_string = clean_json_string(full_response_text)
            data = json.loads(cleaned_json_string)


            is_valid, err_msg = validate_json(data, schema)
            if is_valid:
                return data, raw_response_for_debugging
            else:
                st.warning(f"JSON validation failed: {err_msg}")


        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed (Invalid JSON). Retrying...")
                time.sleep(2)  # Wait 2 seconds before retrying
            else:
                st.error(f"Attempt {attempt + 1} failed. Failed to get valid JSON after {max_retries} attempts: {e}")
                # For debugging, show the raw response that caused the JSON error
                st.code(raw_response_for_debugging, language='json', label="Raw response causing JSON error (for debugging)")
                raise Exception(f"Failed to get valid JSON after {max_retries} attempts: {e}")
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed (Unexpected error: {e}). Retrying...")
                time.sleep(2)
            else:
                st.error(f"Attempt {attempt + 1} failed. An unexpected error occurred after {max_retries} attempts: {e}")
                raise
    raise Exception("Max retries reached without a successful response.")



# --- Main Streamlit App ---
def main():
    """
    Main function to run the Bewell AI Health Analyzer Streamlit app with live prompt editing.
    """
    st.title("🌿 Your Personal AI Health Assistant – HIPAA Secure by Bewell + Vertex AI (PV+2)")
    st.write("Upload your lab report(s) and health assessment files for a personalized analysis.")

    # Editable text areas for live prompt customization
    edited_base_prompt = st.text_area(
        "Edit Base Prompt (Common instructions)",
        value=BASE_PROMPT_COMMON,
        height=300,
        key="edited_base_prompt"
    )

    # Compose the full default wrapped prompts incorporating the edited base prompt
    default_biomarker_prompt = edited_base_prompt + "--- Specific Instructions for Biomarker Analysis ---\n" + JSON_STRUCTURE_BIOMARKERS
    default_four_pillars_prompt = edited_base_prompt + """
--- Specific Instructions for Four Pillars Analysis ---
🚨 **PILLAR-SPECIFIC CONTENT AND NAMING RULES - NON-NEGOTIABLE**:
This is your most important rule for the `four_pillars` section.
1.  **Fixed Pillar Names**: You MUST generate exactly four pillar objects. Their `name` fields MUST be exactly: `"Eat Well"`, `"Sleep Well"`, `"Move Well"`, and `"Recover Well"`.
2.  **Omit Irrelevant Arrays**: You MUST only include the `additional_guidance.structure` keys that are relevant to the specific pillar.
    * **For the "Eat Well" pillar**: Your `structure` object MUST ONLY contain the keys `"recommended_foods"` and `"cautious_foods"`. Do NOT include any other keys.
    * **For the "Move Well" pillar**: Your `structure` object MUST ONLY contain the keys `"recommended_workouts"` and `"avoid_habits_move"`. Do NOT include any other keys.
    * **For the "Sleep Well" pillar**: Your `structure` object MUST ONLY contain the keys `"recommended_recovery_tips"` and `"avoid_habits_rest_recover"`. Do NOT include any other keys.
    * **For the "Recover Well" pillar**: Your `structure` object MUST ONLY contain the keys `"recommended_recovery_tips"` and `"avoid_habits_rest_recover"`. Do NOT include any other keys.
""" + JSON_STRUCTURE_4PILLARS

    default_supplements_prompt = edited_base_prompt + """
--- Supplements Output Instructions ---
🔒 STRICT OUTPUT REQUIREMENTS:
- Your output MUST be ONLY ONE JSON object matching exactly the provided `JSON_STRUCTURE_SUPPLEMENTS_ACTIONS`.
- You MUST NOT include any other keys, sections, or objects besides what is defined in `JSON_STRUCTURE_SUPPLEMENTS_ACTIONS`.
- DO NOT mention or include any information about biomarkers, lab analyses, four pillars, lifestyle, or anything outside supplements.
- DO NOT add explanations, summaries, or markdown.
- Respond ONLY with the complete supplements JSON object.
📌 CONTENT FOCUS:
- Provide supplement advice strictly based on the user’s provided health assessment and lab data.
- Every recommendation must be actionable, personalized, and clearly linked to the user’s data.
- Use the given JSON keys only: `description`, `structure` with `recommendations` array (each item has `name`, `rationale`, `expected_outcomes`, `dosage_and_timing`, and `situational_cyclical_considerations`), plus `conclusion`.
--- END INSTRUCTIONS ---
""" + JSON_STRUCTURE_SUPPLEMENTS_ACTIONS

    edited_biomarker_prompt = st.text_area(
        "Edit Biomarkers Analysis Prompt (includes Base Prompt)",
        value=default_biomarker_prompt,
        height=350,
        key="edited_biomarker_prompt"
    )
    edited_four_pillars_prompt = st.text_area(
        "Edit Four Pillars Analysis Prompt (includes Base Prompt)",
        value=default_four_pillars_prompt,
        height=350,
        key="edited_four_pillars_prompt"
    )
    edited_supplements_prompt = st.text_area(
        "Edit Supplements Analysis Prompt (includes Base Prompt)",
        value=default_supplements_prompt,
        height=350,
        key="edited_supplements_prompt"
    )

    # File upload UI
    col1, col2 = st.columns(2)
    with col1:
        lab_report_files = st.file_uploader(
            "1. Upload Lab Report(s) (optional)",
            type=[".pdf", ".docx", ".xlsx", ".xls", ".txt", ".csv"],
            accept_multiple_files=True
        )
    with col2:
        health_assessment_file = st.file_uploader(
            "2. Upload Health Assessment (required)",
            type=[".pdf", ".docx", ".xlsx", ".xls", ".txt", ".csv"]
        )

    if not health_assessment_file:
        st.info("Please upload a health assessment file to begin.")
        st.stop()

    if st.button("Analyze My Data ✨", type="primary"):
        # Process uploaded files
        raw_lab_report_inputs = []
        if lab_report_files:
            for file in lab_report_files:
                extracted_text = extract_text_from_file(file)
                if "[Error" in extracted_text or "[Unsupported" in extracted_text:
                    st.warning(f"Problem with Lab Report '{file.name}': {extracted_text}")
                else:
                    raw_lab_report_inputs.append(f"--- Start Lab Report: {file.name} ---\n{extracted_text}\n--- End Lab Report: {file.name} ---")
        combined_lab_report_text = "\n\n".join(raw_lab_report_inputs)

        raw_health_assessment_input = extract_text_from_file(health_assessment_file)
        if "[Error" in raw_health_assessment_input:
            st.error(f"Critical Error processing Health Assessment: {raw_health_assessment_input}")
            st.stop()

        lab_report_section_formatted = ""
        if combined_lab_report_text:
            lab_report_section_formatted = f"""
Here is the user's Lab Report text (potentially multiple reports combined):
{combined_lab_report_text}
"""
        else:
            lab_report_section_formatted = "No Lab Report text was provided. Analysis will be based solely on Health Assessment data."

        # Initialize output containers
        final_combined_output = {}
        all_raw_responses_for_debugging = {}
        full_prompts_for_debugging = {}

        with st.status("Initiating Bewell Analysis...", expanded=True) as status_message_box:
            status_message_box.write("⚙️ Preparing data for analysis...")

            # Use edited prompts with placeholders replaced
            biomarker_prompt = st.session_state.get("edited_biomarker_prompt", default_biomarker_prompt)
            biomarker_prompt = biomarker_prompt.format(
                health_assessment_text=raw_health_assessment_input,
                lab_report_section_placeholder=lab_report_section_formatted
            )
            full_prompts_for_debugging["Biomarkers Analysis Prompt"] = biomarker_prompt

            try:
                biomarker_data_raw, raw_biomarker_response = call_gemini_with_retry(biomarker_prompt, schema_biomarkers) 
                if biomarker_data_raw:
                    final_combined_output.update(biomarker_data_raw)
                all_raw_responses_for_debugging["Biomarkers Raw Response"] = raw_biomarker_response
                status_message_box.write("✅ Biomarker analysis complete.")
            except Exception as e:
                status_message_box.error(f"Failed to analyze biomarkers: {e}")

            four_pillars_prompt = st.session_state.get("edited_four_pillars_prompt", default_four_pillars_prompt)
            four_pillars_prompt = four_pillars_prompt.format(
                health_assessment_text=raw_health_assessment_input,
                lab_report_section_placeholder=lab_report_section_formatted
            )
            full_prompts_for_debugging["Four Pillars Analysis Prompt"] = four_pillars_prompt

            try:
                four_pillars_data_raw, raw_four_pillars_response = call_gemini_with_retry(four_pillars_prompt, schema_4pillars)
                if four_pillars_data_raw:
                    final_combined_output.update(four_pillars_data_raw)
                all_raw_responses_for_debugging["Four Pillars Raw Response"] = raw_four_pillars_response
                status_message_box.write("✅ Four Pillars analysis complete.")
            except Exception as e:
                status_message_box.error(f"Failed to analyze four pillars: {e}")

            supplements_actions_prompt = st.session_state.get("edited_supplements_prompt", default_supplements_prompt)
            supplements_actions_prompt = supplements_actions_prompt.format(
                health_assessment_text=raw_health_assessment_input,
                lab_report_section_placeholder=lab_report_section_formatted
            )
            full_prompts_for_debugging["Supplements & Action Items Analysis Prompt"] = supplements_actions_prompt

            try:
                supplements_actions_data_raw, raw_supplements_actions_response = call_gemini_with_retry(supplements_actions_prompt, schema_supplements)
                if supplements_actions_data_raw:
                    final_combined_output.update(supplements_actions_data_raw)
                all_raw_responses_for_debugging["Supplements & Action Items Raw Response"] = raw_supplements_actions_response
                status_message_box.write("✅ Supplements analysis complete.")
            except Exception as e:
                status_message_box.error(f"Failed to analyze supplements and action items: {e}")

            status_message_box.write("✨ Finalizing analysis and preparing report...")

            if "lab_analysis" in final_combined_output and final_combined_output["lab_analysis"]:
                lab_analysis = final_combined_output["lab_analysis"]
                detailed_biomarkers = lab_analysis.get("detailed_biomarkers", [])
                biomarkers_count = len(detailed_biomarkers)

                if lab_analysis.get("biomarkers_tested_count") != biomarkers_count:
                    if lab_analysis.get("biomarkers_tested_count") is None or biomarkers_count > 0:
                        status_message_box.warning(f"Correcting biomarker count in final output (was {lab_analysis.get('biomarkers_tested_count')}, now {biomarkers_count}).")
                        lab_analysis["biomarkers_tested_count"] = biomarkers_count

                        optimal = sum(1 for bm in detailed_biomarkers if bm.get("status") == "optimal")
                        keep_in_mind = sum(1 for bm in detailed_biomarkers if bm.get("status") == "keep_in_mind")
                        attention = sum(1 for bm in detailed_biomarkers if bm.get("status") == "attention_needed")
                        if "biomarker_categories_summary" in lab_analysis:
                            lab_analysis["biomarker_categories_summary"]["optimal_count"] = optimal
                            lab_analysis["biomarker_categories_summary"]["keep_in_mind_count"] = keep_in_mind
                            lab_analysis["biomarker_categories_summary"]["attention_needed_count"] = attention

            if any(final_combined_output.values()):
                status_message_box.update(label="Bewell Analysis Complete!", state="complete")

                st.header("🔬 Your Personalized Bewell Analysis (Interactive JSON):")
                st.json(final_combined_output, expanded=True)

                st.markdown("---")
                st.header("📋 Copy Full JSON Output (Plain Text):")
                plain_json_string = json.dumps(final_combined_output, indent=2)
                st.text_area(
                    "Select the text below and copy it to your clipboard:",
                    plain_json_string,
                    height=400,
                    disabled=True
                )
            else:
                status_message_box.error("❌ No analysis data could be generated. Please check the inputs and ensure Vertex AI is correctly configured.")

        with st.expander("Show All Debug Information (Raw Responses & Prompts)"):
            combined_debug_output = []

            combined_debug_output.append("--- START OF RAW MODEL RESPONSES ---\n\n")
            for part_name, response_text in all_raw_responses_for_debugging.items():
                combined_debug_output.append(f"--- {part_name} ---\n")
                combined_debug_output.append(response_text)
                combined_debug_output.append(f"\n--- END OF {part_name} ---\n\n")

            combined_debug_output.append("\n--- START OF FULL PROMPTS SENT TO AI ---\n\n")
            for prompt_name, prompt_text in full_prompts_for_debugging.items():
                combined_debug_output.append(f"--- {prompt_name} ---\n")
                combined_debug_output.append(prompt_text)
                combined_debug_output.append(f"\n--- END OF {prompt_name} ---\n\n")

            st.code("".join(combined_debug_output), language='text')


if __name__ == "__main__":
    main()
