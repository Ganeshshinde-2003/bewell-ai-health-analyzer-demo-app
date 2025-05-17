import streamlit as st
import google.generativeai as genai
import os
import fitz  # PyMuPDF - for PDF
import io  # To handle file bytes
import pandas as pd # for Excel and CSV
from docx import Document # for .docx files
import json # Keep json imported for the parsing attempt in the output section

# --- Page Configuration ---
st.set_page_config(
    page_title="Bewell AI Health Analyzer",
    page_icon="🌿",
    layout="wide"
)

# --- Text Extraction Function (Handles multiple file types) ---
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
        # Read the file into a BytesIO object
        file_bytes = uploaded_file.getvalue()

        if file_extension == ".pdf":
            # PDF extraction using PyMuPDF (fitz)
            pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                # Use get_text("text") for standard text extraction
                text += page.get_text("text") + "\n" # Add newline between pages
            pdf_document.close()

        elif file_extension == ".docx":
            # DOCX extraction using python-docx
            doc = Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                text += para.text + "\n"

        elif file_extension in [".xlsx", ".xls"]:
            # Excel extraction using pandas
            try:
                # Read all sheets into a dictionary of DataFrames
                excel_data = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
                for sheet_name, df in excel_data.items():
                    text += f"--- Sheet: {sheet_name} ---\n"
                    # Convert DataFrame to a simple text representation suitable for AI prompt
                    text += df.to_string(index=False) + "\n\n" # Use to_string for better readability than CSV
            except Exception as excel_e:
                 # Catch pandas specific errors or missing dependencies (xlrd, openpyxl)
                 st.warning(f"Could not read Excel file '{uploaded_file.name}'. Ensure 'openpyxl' (for .xlsx) or 'xlrd' (for .xls) is installed if needed. Error: {excel_e}")
                 return f"[Could not automatically process Excel file {uploaded_file.name} - Please try pasting the text or saving as PDF/TXT.]" # Indicate failure

        elif file_extension in [".txt", ".csv"]:
            # Text or CSV extraction (simple read)
            text = file_bytes.decode('utf-8', errors='ignore') # Add errors='ignore' for robustness

        # Add other file types here if needed

        else:
            # Handle unsupported file types explicitly
            st.warning(f"Unsupported file type uploaded: {file_extension} for file '{uploaded_file.name}'")
            return "[Unsupported File Type Uploaded]" # Return a marker for unsupported type

    except Exception as e:
        # Catch any other errors during processing
        st.error(f"An error occurred while processing '{uploaded_file.name}': {e}")
        return "[Error Processing File]" # Return a marker for processing error

    # Basic check if text was extracted (ignore for txt/csv as they might be empty)
    if not text.strip() and file_extension not in [".txt", ".csv"] and not text.startswith("["): # Check if it's empty and not already an error/warning marker
         st.warning(f"No readable text extracted from '{uploaded_file.name}'. The file might be scanned, empty, or have complex formatting. Consider pasting the text manually.")

    return text


# --- API Key Handling ---
# Prioritize Streamlit secrets, then environment variable, then user input
api_key = "AIzaSyArQ9zeya1SO-IwsMappkLStXYT0W7WXfk" # Check .env first
if not api_key:
    api_key = "AIzaSyArQ9zeya1SO-IwsMappkLStXYT0W7WXfk" # Check Streamlit secrets (recommended for Streamlit Cloud)
    if not api_key:
        st.warning("Please add your Google/Gemini API key to your environment variables (e.g., in a .env file as GOOGLE_API_KEY), Streamlit secrets, or paste it below.")
        # Fallback to text input in the app if key is not found elsewhere
        api_key = st.text_input("Or paste your Google/Gemini API Key here:", type="password")

# --- Gemini Model Configuration ---
model_name = "gemini-2.0-flash"

generation_config = {
    "temperature": 0.7, # Adjust for creativity vs. predictability. For strict JSON, lower might be better (e.g., 0.1-0.5)
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192, # Generous output size
    "response_mime_type": "text/plain", # Request text/plain, but instruct AI to return JSON
}

# --- The Main Prompt Construction (using the JSON structure definition) ---
# This section now defines the two parts of the prompt
BASE_PROMPT_INSTRUCTIONS = """
Role: Bewell AI Assistant.
Persona: Holistic women's health expert, precision medicine, functional doctor, Women focused care.
Tone: Approachable, professional, empathetic, supportive, clear, accessible. Avoid casual language. Do not use the pronoun "I". Avoid personifying the analysis tone and manner. Focus on empowering the user with clear, accurate, and personalized insights from Bewell.

Input: User's health assessment text and lab report text. Analyze *only* the information provided in these texts. Base the analysis, recommendations, and rationale *solely* on the specific biomarkers and symptoms reported by the user in the provided data.

---

User Data:

Here is the user's Lab Report text:
{lab_report_text}

Here is the user's Health Assessment text:
{health_assessment_text}

---

Instructions for Output:

Generate ONE complete response as a JSON object. Do NOT include any introductory or concluding text outside the JSON object. **Do NOT wrap the JSON object in markdown code blocks (```json ... ``` or ``` ... ```).** Populate all fields based *only* on the provided user data, adhering to the persona, tone, and data constraints specified.

**Emphasis on Detail and Completeness:**
*   Provide comprehensive and detailed descriptions in all relevant string fields, not just one or two sentences.
*   Include specific numbers or counts within descriptions where they are relevant (e.g., in summaries or when discussing biomarker categories).
*   Ensure *all* arrays/lists within the JSON structure are populated with relevant items derived *from the provided user data* and your analysis. If the input data does not provide information relevant to a specific list (e.g., no health assessment details about stress for recovery tips), populate with general guidance related to women's health that aligns with the pillar, and clearly state this is general guidance, but prioritize linking to the user's data. Avoid returning empty lists if the data allows for content.

Here is the required JSON object structure:
"""

JSON_STRUCTURE_DEFINITION = """
{
  "lab_analysis": {
    "overall_summary": "string - Provide a comprehensive synthesis of the user's current health status based solely on provided lab results and health assessment. Highlight significant areas of concern or strengths in simple, personalized, and detailed language.",
    "biomarkers_tested_count": "integer - Total number of biomarkers listed and analyzed from the provided lab report.",
    "biomarker_categories_summary": {
      "description_text": "string - A detailed summary of the biomarker categories, explicitly mentioning the total number tested and the count for each status (e.g., 'Out of [total count] biomarkers tested, [optimal count] are Optimal, [keep in mind count] require Keep in Mind, and [attention needed count] need Attention Needed').",
      "optimal_count": "integer - Count of biomarkers whose status is 'Optimal (Green)'.",
      "keep_in_mind_count": "integer - Count of biomarkers whose status is 'Keep in Mind (Yellow)'.",
      "attention_needed_count": "integer - Count of biomarkers whose status is 'Attention Needed (Orange-Red)'."
    },
    "detailed_biomarkers": "array of objects - Contains comprehensive details for *each* biomarker listed in the provided lab report.",
    "detailed_biomarkers_item_structure": {
      "name": "string - The full name of the biomarker.",
      "status": "string - Machine-readable status: 'optimal', 'keep_in_mind', or 'attention_needed'. Determine this based on the result relative to the provided range.",
      "status_label": "string - Display label mirroring original status labels (e.g., 'Optimal (Green)', 'Keep in Mind (Yellow)', 'Attention Needed (Orange-Red)').",
      "result": "string - User's specific measured result, including units.",
      "range": "string - The provided reference or optimal range.",
      "cycle_impact": "string - Provide a detailed explanation of any known fluctuations or impacts specific menstrual cycle phases have on this biomarker relevant to women's health, or state 'Not typically impacted by cycle' / 'Cycle impact not well-established'.",
      "why_it_matters": "string - Provide a detailed explanation of the biomarker's primary function, its importance specifically to women's health, and the potential practical implications if *this user's specific level* is abnormal or borderline. Use clear, science-backed explanations without medical jargon, assuming the user has no medical background."
    },
    "crucial_biomarkers_to_measure": "array of objects - A list of essential biomarkers women should measure regularly, with a brief explanation for each.",
     "crucial_biomarkers_item_structure": {
        "name": "string - The name of the crucial biomarker.",
        "importance": "string - Provide a brief yet clear explanation of its importance in accessible language."
     },
    "health_recommendation_summary": "array of strings - Provide clear, concise, *actionable*, and specific steps tailored *specifically* to the user's *provided* lab results and health assessment findings. Use complete phrases for each step."
  },
  "four_pillars": {
    "introduction": "string - Provide a detailed introduction summarizing the user's overall health status and key findings from the lab analysis section in clear, accessible language, setting the stage for the Four Pillars analysis.",
    "pillars": "array of objects - Contains the detailed analysis and recommendations for each of the four pillars: Eat Well, Sleep Well, Move Well, Recover Well.",
    "pillars_item_structure": {
      "name": "string - The name of the pillar (e.g., 'Eat Well', 'Sleep Well', 'Move Well', 'Recover Well').",
      "why_it_matters": "string - Explain in detail how this pillar is *specifically relevant to this user's unique health assessment details and lab findings*. Use reachable language and explicitly link to their reported data points.",
      "personalized_recommendations": "array of strings - Provide *actionable, personalized*, and specific advice tailored to *this user's specific status and lifestyle*. Recommendations must be achievable and written as clear steps.",
      "root_cause_correlation": "string - Clearly explain in accessible language how *each recommendation* within this pillar's personalized recommendations connects directly to the *root causes or contributing factors* identified *in this user's lab results and health assessment*.",
      "science_based_explanation": "string - For *each recommendation* in this pillar, provide a clear, simple, and detailed scientific basis focused on practical user benefits, without medical jargon.",
      "additional_guidance": {
         "description": "Specific lists of recommended/avoided items for this pillar, populated based on relevance to user data or general women's health principles. If specific guidance cannot be derived from user data, provide general, relevant examples for women's health in these lists, and clearly state these are general examples.",
         "structure": {
            "recommended_foods": "array of objects - For 'Eat Well': List recommended top foods. Populate based on user data relevance AND general women's health. Each object: {name: string, description: string|null}.",
            "cautious_foods": "array of objects - For 'Eat Well': List foods to approach cautiously. Populate based on user data relevance AND general women's health. Each object: {name: string, description: string|null}.",
            "recommended_workouts": "array of objects - For 'Move Well': List recommended workouts/types of movement. Populate based on user data relevance AND general women's health. Each object: {name: string, description: string|null}.",
            "avoid_habits_move": "array of objects - For 'Move Well': List habits to avoid. Populate based on user data relevance AND general women's health. Each object: {name: string, description: string|null}.",
            "recommended_recovery_tips": "array of objects - For 'Sleep Well' and 'Recover Well': List recommended recovery tips. Populate based on user data relevance AND general women's health. Each object: {name: string, description: string|null}.",
            "avoid_habits_rest_recover": "array of objects - For 'Sleep Well' and 'Recover Well': List habits to avoid related to rest/recovery. Populate based on user data relevance AND general women's health. Each object: {name: string, description: string|null}."
         }
      }
    }
  },
  "supplements": {
    "description": "Personalized supplement recommendations based specifically and *only* on the provided blood test biomarkers and detailed health assessment responses.",
    "structure": {
      "recommendations": "array of objects - Contains detailed information for each recommended supplement.",
      "recommendations_item_structure": {
        "name": "string - Supplement Name.",
        "rationale": "string - Personalized Rationale: Provide a detailed explanation of *why* recommended based on user's specific biomarkers (mention status: Optimal, Keep in Mind, Attention Needed) and reported health assessment symptoms. Explain exactly how it addresses *this user's specific issues reported* using simple, accessible language.",
        "expected_outcomes": "string - Expected Outcomes: Describe tangible, *personalized* benefits the user can realistically notice, linked to their specific issues.",
        "dosage_and_timing": "string - Recommended Dosage & Timing: Clearly outline precise dosage instructions and optimal timing, based on general guidelines or evidence where applicable.",
        "situational_cyclical_considerations": "string - Situational/Cyclical Considerations: Clearly identify if beneficial during specific menstrual cycle phases or particular life circumstances *if applicable and relevant to the supplement and user's provided profile*. Provide a simple explanation *why* this is the case simply."
      },
      "conclusion": "string - The concluding guidance: Provide concise, reassuring guidance to encourage adherence. Clearly state that supplements are adjunctive and medical consultation is necessary for diagnosis and treatment."
    }
  },
   "action_plan": {
      "description": "Consolidated actionable recommendations for quick reference on the main summary page.",
      "structure": {
          "foods_to_enjoy": "array of strings - Compile key recommended foods (names and brief benefits/reasons) from the 'recommended_foods' list under the 'Eat Well' pillar. If the source list is empty or contains only general examples, clearly state these are general suggestions.",
          "foods_to_limit": "array of strings - Compile key cautious foods (names and brief reasons) from the 'cautious_foods' list under the 'Eat Well' pillar. If the source list is empty or contains only general examples, clearly state these are general suggestions.",
          "daily_habits": "array of strings - Compile key actionable daily practices and habits mentioned in the 'Health Recommendation Summary' and 'personalized_recommendations' across the Four Pillars (e.g., related to consistent sleep, hydration, meal timing, stress management techniques).",
          "rest_and_recovery": "array of strings - Compile key rest and recovery tips/habits (names and brief benefits/reasons) from the 'recommended_recovery_tips' lists under the 'Sleep Well' and 'Recover Well' pillars. If the source list is empty or contains only general examples, clearly state these are general suggestions.",
          "movement": "array of strings - Compile key movement/workout recommendations (names and brief benefits/reasons) from the 'recommended_workouts' list under the 'Move Well' pillar. If the source list is empty or contains only general examples, clearly state these are general suggestions."
      }
   }
}
"""


# --- Function to clean potential markdown code blocks ---
def clean_json_string(json_string):
    """Removes leading/trailing markdown code blocks (```json or ```) from a string."""
    if not isinstance(json_string, str):
        return json_string # Return as is if not a string

    stripped_string = json_string.strip()

    # Check for standard ```json or ``` at the start and ``` at the end
    if stripped_string.startswith('```json'):
        stripped_string = stripped_string[len('```json'):].lstrip() # Remove and strip leading whitespace after

    # Also check for just ```
    elif stripped_string.startswith('```'):
        stripped_string = stripped_string[len('```'):].lstrip() # Remove and strip leading whitespace after

    # Check for ``` at the end
    if stripped_string.endswith('```'):
        stripped_string = stripped_string[:-len('```')].rstrip() # Remove and strip trailing whitespace before

    return stripped_string


# --- App Title and Description ---
st.title("🌿 Bewell AI Health Analyzer")
st.markdown("""
Welcome to Bewell! Upload your lab report and health assessment files or paste the text to receive a personalized health analysis designed to be easy to understand, even without a medical background. We'll analyze your data based on the Four Pillars of wellness (Eat Well, Sleep Well, Move Well, Recover Well) and provide tailored supplement recommendations.

**Note:** This is an AI analysis tool. The insights provided are for informational purposes only and should not replace professional medical advice. Always consult with a qualified healthcare provider for any health concerns or before making decisions about your health or treatment. The quality of the analysis depends heavily on the clarity and readability of the uploaded files or pasted text.
""")

# --- Input Fields ---
st.header("Your Health Data")

st.subheader("Lab Report")
lab_report_file = st.file_uploader(
    "Upload your Lab Report (PDF, DOCX, XLSX, XLS, TXT, CSV)",
    type=["pdf", "docx", "xlsx", "xls", "txt", "csv"],
    help="Upload a file containing your lab test results. Supported formats: PDF, DOCX, XLSX, XLS, TXT, CSV."
)
st.markdown("--- OR ---")
lab_report_text_area = st.text_area(
    "Paste Lab Report Text (Optional: if file upload doesn't work or you prefer)",
    height=300,
    placeholder="Paste the text extracted from your lab report here...",
    help="Copy and paste the text content from your lab report if file upload is not used or fails."
)

st.subheader("Health Assessment")
health_assessment_file = st.file_uploader(
    "Upload your Health Assessment (PDF, DOCX, XLSX, XLS, TXT, CSV)",
    type=["pdf", "docx", "xlsx", "xls", "txt", "csv"],
    help="Upload a file containing your health assessment details. Supported formats: PDF, DOCX, XLSX, XLS, TXT, CSV."
)
st.markdown("--- OR ---")
health_assessment_text_area = st.text_area(
    "Paste Health Assessment Text (Optional: if file upload doesn't work or you prefer)",
    height=300,
    placeholder="Describe your health concerns, symptoms, lifestyle, goals, etc...",
    help="Provide details about your current health status, chronic conditions, recent symptoms, lifestyle factors, and any specific concerns if file upload is not used or fails."
)


# --- Analysis Button and Logic ---
if st.button("Generate Health Analysis"):
    if not api_key:
        st.error("Please provide your Google/Gemini API key to proceed.")
    else:
        # --- Get Input Data ---
        # Prioritize uploaded file text, fall back to text area if no file
        raw_lab_report_input = extract_text_from_file(lab_report_file) if lab_report_file else lab_report_text_area
        raw_health_assessment_input = extract_text_from_file(health_assessment_file) if health_assessment_file else health_assessment_text_area

        # Check if at least one source of input is available and is not just an error marker
        if (not raw_lab_report_input or raw_lab_report_input.startswith("[")) and \
           (not raw_health_assessment_input or raw_health_assessment_input.startswith("[")):
            st.error("Please upload valid documents or paste text into the fields. Could not extract valid data from the provided inputs.")
        else:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                    # Add safety settings if desired
                    # safety_settings=...
                )

                # Format the prompt by combining base instructions, user data, and JSON structure definition
                prompt = BASE_PROMPT_INSTRUCTIONS.format(
                    lab_report_text=raw_lab_report_input,
                    health_assessment_text=raw_health_assessment_input
                ) + JSON_STRUCTURE_DEFINITION # Concatenate the JSON structure definition string

                st.info("Extracting text from uploaded files (if any) and generating analysis. This may take a few moments...")

                # Generate the content using the Gemini API
                # Added a small spinner to indicate work is being done
                with st.spinner("Analyzing health data..."):
                    response = model.generate_content(prompt)

                # --- Display the Results ---
                st.header("Your Personalized Bewell Analysis")

                # # --- DEBUG PRINT: See what response.text contains BEFORE cleaning/parsing ---
                # st.subheader("Raw Response from AI (Before Cleaning)")
                # st.text(f"--- Debug: Raw response.text content (len: {len(response.text) if response and response.text is not None else 0}) ---")
                # st.text(repr(response.text) if response and response.text is not None else None) # Use repr to show invisible chars/whitespace
                # st.text("-----------------------------------------------------------\n")
                # --- END DEBUG PRINT ---

                # Check if response and text exist, then attempt to clean and parse
                if response and response.text is not None:
                    # --- Clean the response text before parsing ---
                    cleaned_json_string = clean_json_string(response.text)

                    # # --- DEBUG PRINT: See what the string looks like AFTER cleaning ---
                    # st.subheader("Cleaned Response String (Before JSON Parsing)")
                    # st.text(f"--- Debug: Cleaned string content (len: {len(cleaned_json_string)}) ---")
                    # st.text(repr(cleaned_json_string)) # Use repr to show invisible chars/whitespace
                    # st.text("-----------------------------------------------------------\n")
                    # # --- END DEBUG PRINT ---


                    if cleaned_json_string.strip(): # Check if the string is not empty after cleaning and stripping
                        try:
                            # Attempt to parse the cleaned string as JSON
                            analysis_data = json.loads(cleaned_json_string)
                            # Display pretty-printed JSON if successful
                            st.subheader("Parsed JSON Output")
                            st.json(analysis_data) # Streamlit's built-in JSON display is good

                        except json.JSONDecodeError as e:
                            # This block is reached if the CLEANED string is NOT empty but isn't valid JSON
                            # This will catch errors like invalid syntax *within* the JSON
                            st.error(f"Failed to parse AI response as JSON: {e}")
                            st.error("The model returned text, and potential markdown blocks were removed, but the remaining text was not valid JSON according to the requested structure.")
                            st.info("Please review the 'Cleaned Response String' above to see the output that failed parsing.")


                    else:
                        # This block is reached if the original response.text was None, empty, whitespace,
                        # OR if cleaning/stripping resulted in an empty string.
                        # This is the scenario causing "Expecting value: line 1 column 1" if json.loads was called directly.
                        st.error("Failed to generate a valid response from the model. The response text was empty or became empty after removing potential markdown code blocks.")
                        print("Error: Failed to generate a valid response from the model (text was empty after cleaning).") # Log to console as well


                elif response and response.prompt_feedback:
                     # Handle cases where the prompt was blocked
                     st.warning(f"Analysis request was blocked. Reason: {response.prompt_feedback.block_reason}")
                     if response.prompt_feedback.safety_ratings:
                          st.warning(f"Safety Ratings: {response.prompt_feedback.safety_ratings}")
                     st.info("Please review your input for any sensitive or policy-violating content and try again.")
                     st.info("Note: Health information can sometimes be flagged. If you believe this was an error, try simplifying potentially sensitive descriptions in the health assessment.")

                else:
                     # This else block now catches cases where response is None or has no text AND no feedback
                     st.error("Failed to generate a valid response from the model (response object is empty or unexpected, and no feedback provided).")
                     print("Error: Failed to generate a valid response from the model (response object empty/unexpected).") # Log to console as well


                # --- Display Raw Extracted Input ---
                st.markdown("---") # Separator
                st.subheader("Review Raw Input Sent to AI (for debugging)")
                st.info("This section shows the text extracted from your uploaded files or pasted into the text areas. This is what the AI processed.")

                # Using expanders to keep the UI clean
                if raw_lab_report_input and not raw_lab_report_input.startswith("["):
                    with st.expander("Show Raw Lab Report Text"):
                        st.text(raw_lab_report_input) # Use st.text for raw text output
                elif raw_lab_report_input.startswith("["):
                     st.warning(f"Lab Report Input Status: {raw_lab_report_input}")
                else:
                    st.info("No Lab Report file uploaded or text pasted.")

                if raw_health_assessment_input and not raw_health_assessment_input.startswith("["):
                    with st.expander("Show Raw Health Assessment Text"):
                         st.text(raw_health_assessment_input) # Use st.text for raw text output
                elif raw_health_assessment_input.startswith("["):
                     st.warning(f"Health Assessment Input Status: {raw_health_assessment_input}")
                else:
                    st.info("No Health Assessment file uploaded or text pasted.")


            except Exception as e:
                st.error(f"An unexpected error occurred during API interaction or response processing: {e}")
                st.error("Please check your API key, ensure the model is accessible, and verify your internet connection.")
                # Optional: Print the traceback for debugging server-side errors
                # import traceback
                # st.exception(e) # This prints the traceback in the Streamlit UI


# --- Footer ---
st.markdown("---")
st.markdown("Built with Streamlit and Google Gemini. This analysis is AI-generated and for informational purposes only. Always consult a medical professional for diagnosis and treatment.")