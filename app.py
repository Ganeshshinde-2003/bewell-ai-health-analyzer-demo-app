import streamlit as st
import google.generativeai as genai
import os
import fitz  # PyMuPDF - for PDF
import io  # To handle file bytes
import pandas as pd  # for Excel and CSV
from docx import Document  # for .docx files
import json  # For JSON parsing
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

# --- API Key Handling ---
api_key = "AIzaSyArQ9zeya1SO-IwsMappkLStXYT0W7WXfk"
if not api_key:
    st.warning("Please add your Google/Gemini API key to Streamlit secrets or environment variables, or paste it below.")
    api_key = st.text_input("Paste your Google/Gemini API Key here:", type="password")

# --- Gemini Model Configuration ---
model_name = "gemini-2.0-flash"
generation_config = {
    "temperature": 0.2,  # Low for deterministic JSON
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 16384,  # Increased for large responses
    "response_mime_type": "application/json"  # Ensures strict JSON
}

# --- The Main Prompt Construction ---
BASE_PROMPT_INSTRUCTIONS = """
Role: Bewell AI Assistant.
Persona: Holistic women's health expert, precision medicine, functional doctor, women-focused care.
Tone: Approachable, professional, empathetic, supportive, clear, accessible. Avoid casual language. Do not use the pronoun "I". Avoid personifying the analysis tone. Focus on empowering the user with clear, accurate, personalized insights.

Input: User's health assessment text and lab report text. Analyze *only* the information provided. Base analysis, recommendations, and rationale *solely* on the specific biomarkers and symptoms reported.

---

User Data:

Health Assessment text:
{health_assessment_text}

{lab_report_section_placeholder}

---

Instructions for Output:

Generate ONE complete response as a JSON object adhering strictly to the JSON_STRUCTURE_DEFINITION below. Do NOT include introductory or concluding text outside the JSON. Do NOT wrap in markdown code blocks (```json ... ``` or ``` ... ```). Populate all fields based *only* on provided user data, adhering to persona, tone, and constraints.

**CRITICAL ACCURACY REQUIREMENT: BIOMARKER COUNTING**
1. **Process All Biomarkers**: Extract all biomarkers with valid results from the lab report, identifying names, results, ranges, and statuses ("optimal", "keep_in_mind", "attention_needed"). Exclude "Not Performed" or "Request Problem" biomarkers from 'detailed_biomarkers'; list these in 'crucial_biomarkers_to_measure' with symptom-linked explanations.
2. **Count Verification**: After generating 'detailed_biomarkers', count the total objects. Assign this to 'lab_analysis.biomarkers_tested_count'. Count biomarkers by status for 'biomarker_categories_summary'. Verify that 'optimal_count' + 'keep_in_mind_count' + 'attention_needed_count' equals 'biomarkers_tested_count'. Reprocess if counts mismatch.
3. **Log Parsing Issues**: If lab report text is ambiguous (e.g., scanned PDFs, tables), note in 'lab_analysis.overall_summary': "Some lab data could not be parsed due to formatting issues. Recommendations are based on available data and health assessment."

**CRITICAL NON-EMPTY ARRAY REQUIREMENT**
- For 'additional_guidance.structure' arrays ('recommended_foods', 'cautious_foods', 'recommended_workouts', 'avoid_habits_move', 'recommended_recovery_tips', 'avoid_habits_rest_recover'), ensure ALL are populated with at least 2 relevant items. Use user data (symptoms, diagnoses, biomarkers) first. If insufficient, include general women’s health recommendations, stating they are general.
- **Symptom-to-Recommendation Mappings** (use these to populate arrays):
  - **Constipation**: 'recommended_foods': fiber-rich foods (e.g., vegetables, chia seeds); 'cautious_foods': low-fiber processed foods.
  - **Lactose sensitivity**: 'cautious_foods': dairy products (e.g., milk, cheese).
  - **Stress**: 'recommended_recovery_tips': meditation, deep breathing; 'avoid_habits_rest_recover': excessive screen time before bed.
  - **Fatigue**: 'recommended_workouts': low-impact exercises (e.g., yoga, walking); 'recommended_recovery_tips': consistent sleep schedule.
  - **Sedentary job**: 'recommended_workouts': daily movement (e.g., walking, stretching); 'avoid_habits_move': prolonged sitting.
  - **High cortisol**: 'recommended_recovery_tips': stress-reducing activities (e.g., mindfulness); 'avoid_habits_rest_recover': caffeine late in day.
  - **Low estrogen**: 'recommended_foods': phytoestrogen-rich foods (e.g., flaxseeds, soy).
- Verify all arrays are non-empty before finalizing JSON. If user data is missing, include general examples (e.g., "No dietary data provided; general recommendations for women’s health include...").

**CRITICAL SCHEMA ADHERENCE**
- Include ALL specified fields in JSON_STRUCTURE_DEFINITION, even if empty (e.g., 'detailed_biomarkers' as [] if no lab report). Do NOT add undefined fields.
- Ensure array items follow specified structures without metadata keys (e.g., no "_item_structure").
- Use detailed, complete descriptions in string fields.

**CRITICAL TEXT MARKING FOR HIGHLIGHTING**
- Use **C1[text]C1** for primary highlights: total counts, key biomarker names/statuses, diagnoses, main symptoms.
- Use **C2[text]C2** for secondary highlights: specific values, ranges, "not performed" tests, actionable keywords.
- Apply sparingly to critical terms (e.g., **C1[fatigue]C1**, **C2[4.310 uIU/mL]C2**).

**KEY PERSONALIZATION AND SCIENTIFIC LINKING**
- Anchor every recommendation to user input (symptom, diagnosis, biomarker).
- Explicitly reference symptoms/diagnoses (e.g., "Given **C1[constipation]C1**, include fiber-rich foods").
- For "Not Performed" biomarkers, list in 'crucial_biomarkers_to_measure' with symptom links (e.g., "Test **C2[DHEA, Serum]C2** for **C1[stress]C1**").
- Provide women-specific, science-backed rationale (e.g., "Fiber supports **C2[estrogen detoxification]C2** for **C1[bloating]C1**").
- Use precise language (e.g., "Based on **C1[lactose sensitivity]C1**, avoid dairy"). Avoid vague terms like "some women."
- Explain hormone interactions (e.g., "**C2[cortisol]C2** impacts **C1[bloating]C1**").

**Conditional Analysis**
- **With Lab Report**: Analyze all valid biomarkers, categorize by status, link to symptoms/diagnoses.
- **Without Lab Report**: Use health assessment only. Recommend biomarkers to test (e.g., "For **C1[fatigue]C1**, test **C2[thyroid panel]C2**").
- Populate arrays with relevant items. If data is insufficient, use general women’s health guidance, noting it’s general.

**Four Pillars Scoring**
- Score each pillar (Eat Well, Sleep Well, Move Well, Recover Well) from 1 (needs improvement) to 10 (optimal) based on user data.
- Link scores to inputs (e.g., "**C1[inconsistent eating]C1** gives Eat Well **C2[4]C2**").
"""

# --- JSON Structure Definition ---
JSON_STRUCTURE_DEFINITION = """
{
  "lab_analysis": {
    "overall_summary": "string - Synthesize health status from lab results and health assessment. Reference symptoms/diagnoses (e.g., 'Your **C1[fatigue]C1** and **C2[high TSH]C2** suggest thyroid monitoring'). If no lab report, state analysis is health assessment only. Note unreadable lab data: 'Some lab data could not be parsed.'",
    "biomarkers_tested_count": "integer - Exact count of 'detailed_biomarkers' objects. Verify against array length. Set to 0 if no lab report.",
    "biomarker_categories_summary": {
      "description_text": "string - Summarize categories (e.g., 'Out of **C1[{biomarkers_tested_count}]C1** biomarkers, **C2[{optimal_count}]C2** Optimal, **C2[{keep_in_mind_count}]C2** Keep in Mind, **C2[{attention_needed_count}]C2** Attention Needed'). Exclude 'Not Performed' tests.",
      "optimal_count": "integer - Count of 'optimal' biomarkers. Set to 0 if no lab report.",
      "keep_in_mind_count": "integer - Count of 'keep_in_mind' biomarkers. Set to 0 if no lab report.",
      "attention_needed_count": "integer - Count of 'attention_needed' biomarkers. Set to 0 if no lab report."
    },
    "detailed_biomarkers": [
      {
        "name": "string - Full biomarker name (e.g., 'Estradiol').",
        "status": "string - 'optimal', 'keep_in_mind', or 'attention_needed'.",
        "status_label": "string - Display label (e.g., 'Optimal (Green)').",
        "result": "string - Result with units (e.g., '**C2[4.310 uIU/mL]C2**').",
        "range": "string - Reference range (e.g., '0.450-4.500').",
        "cycle_impact": "string - Menstrual cycle impact or 'Not typically impacted by cycle' (e.g., 'Estradiol fluctuates, relevant to **C1[cramps]C1**').",
        "why_it_matters": "string - Role in women’s health, linked to user data (e.g., 'High **C2[TSH]C2** may cause **C1[fatigue]C1**')."
      }
    ],
    "crucial_biomarkers_to_measure": [
      {
        "name": "string - Biomarker name (e.g., 'DHEA, Serum').",
        "importance": "string - Why testing is needed (e.g., 'Test **C2[DHEA, Serum]C2** for **C1[stress]C1**')."
      }
    ],
    "health_recommendation_summary": "array of strings - Actionable steps (e.g., 'Retest **C2[DHEA, Serum]C2** for **C1[stress]C1**')."
  },
  "four_pillars": {
    "introduction": "string - Summarize health status and lab findings for pillars.",
    "pillars": [
      {
        "name": "string - Pillar name (e.g., 'Eat Well').",
        "score": "integer - 1 (needs improvement) to 10 (optimal), based on user data (e.g., '**C2[4]C2** for **C1[inconsistent eating]C1**').",
        "why_it_matters": "string - Relevance to user data (e.g., 'Nutrition impacts **C2[estrogen clearance]C2** for **C1[bloating]C1**').",
        "personalized_recommendations": "array of strings - Actionable advice (e.g., 'Eat fiber-rich foods for **C1[constipation]C1**').",
        "root_cause_correlation": "string - Link to root causes (e.g., 'Fiber addresses **C1[constipation]C1** tied to **C2[low estrogen]C2**').",
        "science_based_explanation": "string - Scientific basis (e.g., 'Fiber supports **C2[estrogen detoxification]C2** for **C1[mood swings]C1**').",
        "additional_guidance": {
          "description": "string - Describe guidance, note if general due to limited data.",
          "structure": {
            "recommended_foods": [{"name": "string", "description": "string|null"}],
            "cautious_foods": [{"name": "string", "description": "string|null"}],
            "recommended_workouts": [{"name": "string", "description": "string|null"}],
            "avoid_habits_move": [{"name": "string", "description": "string|null"}],
            "recommended_recovery_tips": [{"name": "string", "description": "string|null"}],
            "avoid_habits_rest_recover": [{"name": "string", "description": "string|null"}]
          }
        }
      }
    ]
  },
  "supplements": {
    "description": "string - Explain supplement recommendations based on lab/health assessment.",
    "structure": {
      "recommendations": [
        {
          "name": "string - Supplement name (e.g., 'Magnesium').",
          "rationale": "string - Link to biomarkers/symptoms (e.g., 'For **C2[low Estradiol]C2** and **C1[mood swings]C1**').",
          "expected_outcomes": "string - Benefits (e.g., 'Improved **C1[sleep]C1**').",
          "dosage_and_timing": "string - Dosage (e.g., '200 mg daily, evening').",
          "situational_cyclical_considerations": "string - Cycle-specific use (e.g., 'Use in **C1[luteal phase]C1** for **C1[cramps]C1**')."
        }
      ],
      "conclusion": "string - Encourage adherence, note medical consultation."
    }
  },
  "action_plan": {
    "description": "string - Summarize actionable recommendations.",
    "structure": {
      "foods_to_enjoy": "array of strings - Foods linked to symptoms (e.g., 'Fiber-rich vegetables for **C1[constipation]C1**').",
      "foods_to_limit": "array of strings - Foods to avoid (e.g., 'Dairy due to **C1[lactose sensitivity]C1**').",
      "daily_habits": "array of strings - Habits (e.g., 'Sleep **C2[7-9 hours]C2** for **C1[cortisol]C1**').",
      "rest_and_recovery": "array of strings - Recovery tips (e.g., 'Meditation for **C1[stress]C1**').",
      "movement": "array of strings - Workouts (e.g., 'Yoga for **C1[stress]C1**')."
    }
  }
}
"""

def clean_json_string(json_string):
    """Removes markdown code blocks and illegal trailing commas."""
    if not isinstance(json_string, str):
        return json_string
    stripped_string = json_string.strip()
    if stripped_string.startswith('```json'):
        stripped_string = stripped_string[len('```json'):].lstrip()
    elif stripped_string.startswith('```'):
        stripped_string = stripped_string[len('```'):].lstrip()
    if stripped_string.endswith('```'):
        stripped_string = stripped_string[:-len('```')].rstrip()
    stripped_string = re.sub(r',\s*}', '}', stripped_string)
    stripped_string = re.sub(r',\s*]', ']', stripped_string)
    return stripped_string

# --- Main Streamlit App ---
def main():
    """
    Main function to run the Bewell AI Health Analyzer Streamlit app.
    """
    st.title("Bewell AI Health Analyzer-Updated")
    st.write("Upload your lab report and health assessment files for a personalized analysis.")

    lab_report_file = st.file_uploader("Upload Lab Report (optional)", type=[".pdf", ".docx", ".xlsx", ".xls", ".txt", ".csv"])
    health_assessment_file = st.file_uploader("Upload Health Assessment (required)", type=[".pdf", ".docx", ".xlsx", ".xls", ".txt", ".csv"])

    if not health_assessment_file:
        st.error("Health Assessment file is required. Please upload a valid file.")
        return

    raw_lab_report_input = extract_text_from_file(lab_report_file) if lab_report_file else ""
    raw_health_assessment_input = extract_text_from_file(health_assessment_file)

    if raw_health_assessment_input.startswith("[Error"):
        st.error(f"Error processing Health Assessment: {raw_health_assessment_input}")
        return
    if raw_lab_report_input.startswith("[Error"):
        st.warning(f"Warning: Problem with Lab Report: {raw_lab_report_input}")
        st.write("Continuing analysis without lab report data.")
        raw_lab_report_input = ""

    if st.button("Analyze Data"):
        if not api_key:
            st.error("Please provide a Google/Gemini API key.")
            return

        with st.spinner("Analyzing data..."):
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)

                lab_report_section = ""
                if raw_lab_report_input:
                    lab_report_section = f"""
Here is the user's Lab Report text:
{raw_lab_report_input}
"""
                else:
                    lab_report_section = """
No Lab Report text was provided. Analysis will be based solely on Health Assessment data.
"""

                prompt = BASE_PROMPT_INSTRUCTIONS.format(
                    health_assessment_text=raw_health_assessment_input,
                    lab_report_section_placeholder=lab_report_section
                ) + JSON_STRUCTURE_DEFINITION

                response = model.generate_content(prompt)

                if response and response.text is not None:
                    cleaned_json_string = clean_json_string(response.text)
                    if cleaned_json_string.strip():
                        try:
                            analysis_data = json.loads(cleaned_json_string)
                            # Validate biomarker counts
                            detailed_biomarkers = analysis_data.get("lab_analysis", {}).get("detailed_biomarkers", [])
                            biomarkers_count = len(detailed_biomarkers)
                            reported_count = analysis_data.get("lab_analysis", {}).get("biomarkers_tested_count", 0)
                            if biomarkers_count != reported_count:
                                st.warning(f"Discrepancy: Reported biomarkers_tested_count ({reported_count}) does not match actual count ({biomarkers_count}). Correcting counts.")
                                analysis_data["lab_analysis"]["biomarkers_tested_count"] = biomarkers_count
                                optimal_count = sum(1 for bm in detailed_biomarkers if bm.get("status") == "optimal")
                                keep_in_mind_count = sum(1 for bm in detailed_biomarkers if bm.get("status") == "keep_in_mind")
                                attention_needed_count = sum(1 for bm in detailed_biomarkers if bm.get("status") == "attention_needed")
                                analysis_data["lab_analysis"]["biomarker_categories_summary"] = {
                                    "description_text": f"Out of **C1[{biomarkers_count}]C1** biomarkers analyzed, **C2[{optimal_count}]C2** are Optimal, **C2[{keep_in_mind_count}]C2** require Keep in Mind, and **C2[{attention_needed_count}]C2** need Attention Needed.",
                                    "optimal_count": optimal_count,
                                    "keep_in_mind_count": keep_in_mind_count,
                                    "attention_needed_count": attention_needed_count
                                }
                            # Validate non-empty arrays in additional_guidance
                            for pillar in analysis_data.get("four_pillars", {}).get("pillars", []):
                                guidance = pillar.get("additional_guidance", {}).get("structure", {})
                                for key in ["recommended_foods", "cautious_foods", "recommended_workouts", "avoid_habits_move", "recommended_recovery_tips", "avoid_habits_rest_recover"]:
                                    if not guidance.get(key):
                                        st.warning(f"Empty array detected: {key} in pillar {pillar.get('name')}. Populating with general recommendations.")
                                        guidance[key] = [
                                            {"name": f"General {key.replace('_', ' ').title()} 1", "description": "General recommendation for women’s health due to limited specific data."},
                                            {"name": f"General {key.replace('_', ' ').title()} 2", "description": "General recommendation for women’s health due to limited specific data."}
                                        ]
                            st.header("Your Personalized Bewell Analysis:")
                            st.json(analysis_data)
                        except json.JSONDecodeError as e:
                            st.error(f"Error: Failed to parse AI response as JSON: {e}")
                            st.write("The model returned text, but it was not valid JSON.")
                        # Always display cleaned raw response
                        with st.expander("Show Cleaned Raw Model Response (for debugging)"):
                            st.subheader("Cleaned Raw Model Response")
                            st.text(cleaned_json_string)
                    else:
                        st.error("Error: Model returned an empty response.")
                        with st.expander("Show Cleaned Raw Model Response (for debugging)"):
                            st.subheader("Cleaned Raw Model Response")
                            st.text("Empty response received from the model.")
                elif response and response.prompt_feedback:
                    st.error(f"Analysis blocked. Reason: {response.prompt_feedback.block_reason}")
                    if response.prompt_feedback.safety_ratings:
                        st.write(f"Safety Ratings: {response.prompt_feedback.safety_ratings}")
                    st.write("Please review your input and try again.")
                else:
                    st.error("Error: Failed to generate a valid response.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.write("Check API key, network, and input files.")

        with st.expander("Show Raw Input Sent to AI (for debugging)"):
            st.subheader("Raw Input Sent to AI")
            if raw_lab_report_input:
                st.write("Raw Lab Report Text:")
                st.text(raw_lab_report_input)
            else:
                st.write("No Lab Report data provided.")
            if raw_health_assessment_input:
                st.write("Raw Health Assessment Text:")
                st.text(raw_health_assessment_input)
            else:
                st.write("No Health Assessment data provided.")

if __name__ == "__main__":
    main()