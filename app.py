import streamlit as st
import google.generativeai as genai
import os
import fitz  # PyMuPDF - for PDF
import io  # To handle file bytes
import pandas as pd  # for Excel and CSV
from docx import Document  # for .docx files
import json  # Keep json imported for the parsing attempt in the output section
import re

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
            try:
                pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
                for page_num in range(pdf_document.page_count):
                    page = pdf_document.load_page(page_num)
                    # Use get_text("text") for standard text extraction
                    text += page.get_text("text") + "\n"  # Add newline between pages
                pdf_document.close()
            except Exception as e:
                st.error(f"Error processing PDF file: {e}")
                return "[Error Processing PDF File]"

        elif file_extension == ".docx":
            # DOCX extraction using python-docx
            try:
                doc = Document(io.BytesIO(file_bytes))
                for para in doc.paragraphs:
                    text += para.text + "\n"
            except Exception as e:
                st.error(f"Error processing DOCX file: {e}")
                return "[Error Processing DOCX File]"

        elif file_extension in [".xlsx", ".xls"]:
            # Excel extraction using pandas
            try:
                # Read all sheets into a dictionary of DataFrames
                excel_data = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
                for sheet_name, df in excel_data.items():
                    text += f"--- Sheet: {sheet_name} ---\n"
                    # Convert DataFrame to a simple text representation suitable for AI prompt
                    text += df.to_string(index=False) + "\n\n"  # Use to_string for better readability than CSV
            except Exception as excel_e:
                # Catch pandas specific errors or missing dependencies (xlrd, openpyxl)
                st.warning(
                    f"Could not read Excel file '{uploaded_file.name}'. Ensure 'openpyxl' (for .xlsx) or 'xlrd' (for .xls) is installed if needed. Error: {excel_e}")
                return f"[Could not automatically process Excel file {uploaded_file.name} - Please try pasting the text or saving as PDF/TXT.]"  # Indicate failure

        elif file_extension in [".txt", ".csv"]:
            # Text or CSV extraction (simple read)
            text = file_bytes.decode('utf-8', errors='ignore')  # Add errors='ignore' for robustness

        else:
            # Handle unsupported file types explicitly
            st.warning(f"Unsupported file type uploaded: {file_extension} for file '{uploaded_file.name}'")
            return "[Unsupported File Type Uploaded]"  # Return a marker for unsupported type

    except Exception as e:
        # Catch any other errors during processing
        st.error(f"An error occurred while processing '{uploaded_file.name}': {e}")
        return "[Error Processing File]"  # Return a marker for processing error

    # Basic check if text was extracted (ignore for txt/csv as they might be empty)
    if not text.strip() and file_extension not in [".txt", ".csv"] and not text.startswith(
            "["):  # Check if it's empty and not already an error/warning marker
        st.warning(
            f"No readable text extracted from '{uploaded_file.name}'. The file might be scanned, empty, or have complex formatting. Consider pasting the text manually.")

    return text


# --- API Key Handling ---
# Prioritize Streamlit secrets, then environment variable, then user input
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
model_name = "gemini-2.0-flash"

generation_config = {
    "temperature": 0.1,  # <--- Change this for more deterministic JSON
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain", # Keep as text/plain or try "application/json"
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

Here is the user's Health Assessment text:
{health_assessment_text}

{lab_report_section_placeholder}

---

Instructions for Output:

Generate ONE complete response as a JSON object. Do NOT include any introductory or concluding text outside the JSON object. **Do NOT wrap the JSON object in markdown code blocks (```json ... ``` or ``` ... ```).** Populate all fields based *only* on the provided user data, adhering to the persona, tone, and data constraints specified.

**Conditional Analysis:**
If a lab report is provided, follow the detailed biomarker analysis instructions as outlined below.
If the user does NOT provide a lab report, clearly state that the analysis is based solely on the health assessment data provided by the user. In this case:
- Skip the biomarker-specific analysis.
- Provide comprehensive health insights, personalized recommendations, and action plans based exclusively on the user’s reported symptoms, lifestyle factors, and health concerns identified in the health assessment.
- Recommend specific biomarkers the user should consider measuring based on their health assessment responses, clearly explaining why each biomarker is important for their personalized health management.

**Emphasis on Detail and Completeness:**
* Provide comprehensive and detailed descriptions in all relevant string fields, not just one or two sentences.
* Include specific numbers or counts within descriptions where they are relevant (e.g., in summaries or when discussing biomarker categories).
* Ensure *all* arrays/lists within the JSON structure are populated with relevant items derived *from the provided user data* and your analysis. If the input data does not provide information relevant to a specific list (e.g., no health assessment details about stress for recovery tips), populate with general guidance related to women's health that aligns with the pillar, and clearly state these are general examples, but prioritize linking to the user's data. Avoid returning empty lists if the data allows for content.

**Four Pillars Scoring:**
For each of the "Four Pillars" (Eat Well, Sleep Well, Move Well, Recover Well), provide a `score` from 1 to 10 as an integer:
- A score of 1 indicates significant room for improvement or concerning habits in that pillar.
- A score of 10 indicates optimal current habits and excellent alignment with healthy practices in that pillar based on the provided user data.
The score should reflect the user's *current status and habits* in that pillar as inferred from their health assessment and (if provided) lab report.

Here is the required JSON object structure. For arrays, ensure the items within the array conform to the described structure without including the "_item_structure" keys themselves:
"""

# The JSON structure. Removed the "_item_structure" definitions as separate keys.
# Instead, the descriptions for arrays now clarify what their items should look like.
JSON_STRUCTURE_DEFINITION = """
{
  "lab_analysis": {
    "overall_summary": "string - Provide a comprehensive synthesis of the user's current health status based solely on provided lab results and health assessment. Highlight significant areas of concern or strengths in simple, personalized, and detailed language. If no lab report is provided, state that the analysis is based solely on the health assessment data.",
    "biomarkers_tested_count": "integer - Total number of biomarkers listed and analyzed from the provided lab report. If no lab report, this should be 0.",
    "biomarker_categories_summary": {
      "description_text": "string - A detailed summary of the biomarker categories, explicitly mentioning the total number tested and the count for each status (e.g., 'Out of [total count] biomarkers tested, [optimal count] are Optimal, [keep in mind count] require Keep in Mind, and [attention needed count] need Attention Needed'). If no lab report, state that no lab biomarkers were analyzed.",
      "optimal_count": "integer - Count of biomarkers whose status is 'Optimal (Green)'. If no lab report, this should be 0.",
      "keep_in_mind_count": "integer - Count of biomarkers whose status is 'Keep in Mind (Yellow)'. If no lab report, this should be 0.",
      "attention_needed_count": "integer - Count of biomarkers whose status is 'Attention Needed (Orange-Red)'. If no lab report, this should be 0."
    },
    "detailed_biomarkers": [
      // array of objects - Contains comprehensive details for *each* biomarker listed in the provided lab report. If no lab report, this array should be empty.
      // Each object in this array should have the following structure:
      // {
      //   "name": "string - The full name of the biomarker.",
      //   "status": "string - Machine-readable status: 'optimal', 'keep_in_mind', or 'attention_needed'. Determine this based on the result relative to the provided range.",
      //   "status_label": "string - Display label mirroring original status labels (e.g., 'Optimal (Green)', 'Keep in Mind (Yellow)', 'Attention Needed (Orange-Red)').",
      //   "result": "string - User's specific measured result, including units.",
      //   "range": "string - The provided reference or optimal range.",
      //   "cycle_impact": "string - Provide a detailed explanation of any known fluctuations or impacts specific menstrual cycle phases have on this biomarker relevant to women's health, or state 'Not typically impacted by cycle' / 'Cycle impact not well-established'.",
      //   "why_it_matters": "string - Provide a detailed explanation of the biomarker's primary function, its importance specifically to women's health, and the potential practical implications if *this user's specific level* is abnormal or borderline. Use clear, science-backed explanations without medical jargon, assuming the user has no medical background."
      // }
    ],
    "crucial_biomarkers_to_measure": [
      // array of objects - A list of essential biomarkers women should measure regularly, with a brief explanation for each.
      // If no lab report is provided, this list should be populated based on the health assessment responses, recommending relevant biomarkers for the user's reported concerns.
      // Each object in this array should have the following structure:
      // {
      //   "name": "string - The name of the crucial biomarker.",
      //   "importance": "string - Provide a brief yet clear explanation of its importance in accessible language."
      // }
    ],
    "health_recommendation_summary": "array of strings - Provide clear, concise, *actionable*, and specific steps tailored *specifically* to the user's *provided* lab results and health assessment findings. Use complete phrases for each step."
  },
  "four_pillars": {
    "introduction": "string - Provide a detailed introduction summarizing the user's overall health status and key findings from the lab analysis section in clear, accessible language, setting the stage for the Four Pillars analysis.",
    "pillars": [
      // array of objects - Contains the detailed analysis and recommendations for each of the four pillars: Eat Well, Sleep Well, Move Well, Recover Well.
      // Each object in this array should have the following structure:
      // {
      //   "name": "string - The name of the pillar (e.g., 'Eat Well', 'Sleep Well', 'Move Well', 'Recover Well').",
      //   "score": "integer - A numerical score for this pillar, ranging from 1 (needs significant attention) to 10 (optimal/excellent adherence). This score should reflect the user's current status and habits related to this pillar based on the provided health assessment and lab report data. Higher scores indicate better current alignment with optimal health practices in this area.",
      //   "why_it_matters": "string - Explain in detail how this pillar is *specifically relevant to this user's unique health assessment details and lab findings*. Use reachable language and explicitly link to their reported data points.",
      //   "personalized_recommendations": "array of strings - Provide *actionable, personalized*, and specific advice tailored to *this user's specific status and lifestyle*. Recommendations must be achievable and written as clear steps.",
      //   "root_cause_correlation": "string - Clearly explain in accessible language how *each recommendation* within this pillar's personalized recommendations connects directly to the *root causes or contributing factors* identified *in this user's lab results and health assessment*.",
      //   "science_based_explanation": "string - For *each recommendation* in this pillar, provide a clear, simple, and detailed scientific basis focused on practical user benefits, without medical jargon.",
      //   "additional_guidance": {
      //     "description": "Specific lists of recommended/avoided items for this pillar, populated based on relevance to user data or general women's health principles. If specific guidance cannot be derived from user data, provide general, relevant examples for women's health in these lists, and clearly state these are general examples.",
      //     "structure": {
      //       "recommended_foods": [
      //         // array of objects - For 'Eat Well': List recommended top foods. Populate based on user data relevance AND general women's health.
      //         // Each object: {name: string, description: string|null}.
      //       ],
      //       "cautious_foods": [
      //         // array of objects - For 'Eat Well': List foods to approach cautiously. Populate based on user data relevance AND general women's health.
      //         // Each object: {name: string, description: string|null}.
      //       ],
      //       "recommended_workouts": [
      //         // array of objects - For 'Move Well': List recommended workouts/types of movement. Populate based on user data relevance AND general women's health.
      //         // Each object: {name: string, description: string|null}.
      //       ],
      //       "avoid_habits_move": [
      //         // array of objects - For 'Move Well': List habits to avoid. Populate based on user data relevance AND general women's health.
      //         // Each object: {name: string, description: string|null}.
      //       ],
      //       "recommended_recovery_tips": [
      //         // array of objects - For 'Sleep Well' and 'Recover Well': List recommended recovery tips. Populate based on user data relevance AND general women's health.
      //         // Each object: {name: string, description: string|null}.
      //       ],
      //       "avoid_habits_rest_recover": [
      //         // array of objects - For 'Sleep Well' and 'Recover Well': List habits to avoid related to rest/recovery. Populate based on user data relevance AND general women's health.
      //         // Each object: {name: string, description: string|null}.
      //       ]
      //     }
      //   }
      // }
    ]
  },
  "supplements": {
    "description": "Personalized supplement recommendations based specifically and *only* on the provided blood test biomarkers and detailed health assessment responses. If no lab report is provided, base recommendations solely on the health assessment.",
    "structure": {
      "recommendations": [
        // array of objects - Contains detailed information for each recommended supplement.
        // Each object in this array should have the following structure:
        // {
        //   "name": "string - Supplement Name.",
        //   "rationale": "string - Personalized Rationale: Provide a detailed explanation of *why* recommended based on user's specific biomarkers (mention status: Optimal, Keep in Mind, Attention Needed) and reported health assessment symptoms. Explain exactly how it addresses *this user's specific issues reported* using simple, accessible language. If no lab report, base rationale solely on health assessment.",
        //   "expected_outcomes": "string - Expected Outcomes: Describe tangible, *personalized* benefits the user can realistically notice, linked to their specific issues.",
        //   "dosage_and_timing": "string - Recommended Dosage & Timing: Clearly outline precise dosage instructions and optimal timing, based on general guidelines or evidence where applicable.",
        //   "situational_cyclical_considerations": "string - Situational/Cyclical Considerations: Clearly identify if beneficial during specific menstrual cycle phases or particular life circumstances *if applicable and relevant to the supplement and user's provided profile*. Provide a simple explanation *why* this is the case simply."
        // }
      ],
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
    """
    Main function to run the Bewell AI Health Analyzer Streamlit app.
    """
    st.title("Bewell AI Health Analyzer")
    st.write("Upload your lab report and health assessment files to get a personalized analysis.")

    # --- File Upload Section ---
    lab_report_file = st.file_uploader("Upload your Lab Report (optional)", type=[".pdf", ".docx", ".xlsx", ".xls", ".txt", ".csv"])
    health_assessment_file = st.file_uploader("Upload your Health Assessment (required)", type=[".pdf", ".docx", ".xlsx", ".xls", ".txt", ".csv"])

    # --- Validate that health assessment file is provided ---
    if not health_assessment_file:
        st.error("Health Assessment file is required. Please upload a valid file.")
        return  # Stop if the health assessment file is missing

    # --- Process Files and Get Texts ---
    raw_lab_report_input = extract_text_from_file(lab_report_file) if lab_report_file else ""
    raw_health_assessment_input = extract_text_from_file(health_assessment_file)

    # --- Error Handling for File Processing ---
    if raw_health_assessment_input.startswith("[Error"):
        st.error(f"Error processing Health Assessment: {raw_health_assessment_input}")
        return  # Exit if health assessment processing failed

    if raw_lab_report_input.startswith("[Error"):
        st.warning(f"Warning: Problem with Lab Report: {raw_lab_report_input}")
        st.write("Continuing analysis without lab report data.")
        raw_lab_report_input = ""  # Set to empty string to avoid issues in prompt


    # --- Analyze Data and Display Results on Button Click---
    if st.button("Analyze Data"):
        if not api_key:
            st.error("Please provide a Google/Gemini API key to proceed.")
            return

        with st.spinner("Analyzing data... (This may take a few moments)"):
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name=model_name)

                # Conditionally build the lab report section of the prompt
                lab_report_section = ""
                if raw_lab_report_input:
                    lab_report_section = f"""
Here is the user's Lab Report text:
{raw_lab_report_input}
"""
                else:
                    lab_report_section = """
No Lab Report text was provided. The analysis will be based solely on the Health Assessment data.
"""

                # Format the prompt
                prompt = BASE_PROMPT_INSTRUCTIONS.format(
                    health_assessment_text=raw_health_assessment_input,
                    lab_report_section_placeholder=lab_report_section
                ) + JSON_STRUCTURE_DEFINITION

                response = model.generate_content(prompt)

                # --- Process and Display Results ---
                if response and response.text is not None:
                    cleaned_json_string = clean_json_string(response.text)
                    if cleaned_json_string.strip():
                        try:
                            analysis_data = json.loads(cleaned_json_string)
                            st.header("Here is your Personalized Bewell Analysis:")
                            st.json(analysis_data)  # Use st.json for formatted display
                        except json.JSONDecodeError as e:
                            st.error(f"Error: Failed to parse AI response as JSON.  The response was not valid JSON: {e}")
                            st.write("The model returned text, but it was not valid JSON.  Here is the raw text from the AI to help with debugging:")
                            st.text(response.text)  # Use st.text for raw text
                    else:
                        st.error("Error: The model returned an empty response.")
                elif response and response.prompt_feedback:
                    # Handle cases where the prompt was blocked
                    st.error(f"Analysis request was blocked. Reason: {response.prompt_feedback.block_reason}")
                    if response.prompt_feedback.safety_ratings:
                        st.write(f"Safety Ratings: {response.prompt_feedback.safety_ratings}")
                    st.write("Please review your input and try again.")
                else:
                    st.error("Error: Failed to generate a valid response from the model.")

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.write("Please check your API key, network connection, and input files.")

        # --- Display Raw Input for Debugging ---
        if st.expander("Show Raw Input Sent to AI (for debugging)"):
            st.subheader("Raw Input Sent to AI (for debugging)")
            if raw_lab_report_input:
                st.write("Raw Lab Report Text:")
                st.text(raw_lab_report_input)
            else:
                st.write("No Lab Report data was provided.")

            if raw_health_assessment_input:
                st.write("Raw Health Assessment Text:")
                st.text(raw_health_assessment_input)
            else:
                st.write("No Health Assessment data was provided.")


if __name__ == "__main__":
    main()