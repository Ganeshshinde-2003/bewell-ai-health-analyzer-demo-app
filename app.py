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
Tone: Approachable, professional, empathetic, supportive, clear, accessible. Use simple, beginner-friendly language as if explaining to someone with no health knowledge. Avoid technical jargon, define terms simply, and use relatable examples (e.g., "Think of cortisol like a stress alarm in your body"). Avoid casual language. Do not use the pronoun "I". Avoid personifying the analysis tone. Focus on empowering the user with clear, actionable, personalized insights.

Input: User's health assessment text and lab report text. Analyze *only* the information provided. Base analysis, recommendations, and rationale *solely* on the specific biomarkers and symptoms reported.

---

User Data:

Health Assessment text:
{health_assessment_text}

{lab_report_section_placeholder}

---

Instructions for Output:

Generate ONE complete response as a JSON object adhering strictly to the JSON_STRUCTURE_DEFINITION below. Do NOT include introductory or concluding text outside the JSON. Do NOT wrap in markdown code blocks (```json ... ``` or ``` ... ```). Populate all fields based *only* on provided user data, adhering to persona, tone, and constraints. Use simple language to explain health concepts, as if speaking to someone new to health topics (e.g., "Estrogen is a hormone that helps your body with energy and mood").

**CRITICAL ACCURACY REQUIREMENT: BIOMARKER COUNTING**
1. **Process All Biomarkers**: Extract all biomarkers with valid results from the lab report, identifying names, results, ranges, and statuses ("optimal", "keep_in_mind", "attention_needed"). Exclude "Not Performed" or "Request Problem" biomarkers from 'detailed_biomarkers'; list these in 'crucial_biomarkers_to_measure' with symptom-linked explanations in simple terms (e.g., "Test **C2[DHEA]C2** to check stress levels because of **C1[tiredness]C1**").
2. **Count Verification**: After generating 'detailed_biomarkers', count the total objects. Assign this to 'lab_analysis.biomarkers_tested_count'. Count biomarkers by status for 'biomarker_categories_summary'. Verify that 'optimal_count' + 'keep_in_mind_count' + 'attention_needed_count' equals 'biomarkers_tested_count'. Reprocess if counts mismatch.
3. **Log Parsing Issues**: If lab report text is ambiguous (e.g., scanned PDFs, tables), note in 'lab_analysis.overall_summary': "Some lab results could not be read due to file issues. Recommendations use available data and health details."

**CRITICAL NON-EMPTY ARRAY REQUIREMENT**
- For 'additional_guidance.structure' arrays ('recommended_foods', 'cautious_foods', 'recommended_workouts', 'avoid_habits_move', 'recommended_recovery_tips', 'avoid_habits_rest_recover'), ensure ALL are populated with at least 2 relevant items. Use user data (symptoms, diagnoses, biomarkers) first. If insufficient, include general women’s health recommendations in simple terms, stating they are general (e.g., "No food details provided, so try eating vegetables for energy").
- **Symptom-to-Recommendation Mappings** (use these to populate arrays, explained simply):
  - **Constipation**: 'recommended_foods': fiber-rich foods (e.g., vegetables, chia seeds; "Fiber helps food move through your body"); 'cautious_foods': low-fiber processed foods (e.g., chips; "These can slow digestion").
  - **Lactose sensitivity**: 'cautious_foods': dairy products (e.g., milk, cheese; "These can upset your stomach").
  - **Stress**: 'recommended_recovery_tips': meditation, deep breathing (e.g., "Breathing slowly calms your mind"); 'avoid_habits_rest_recover': too much screen time before bed (e.g., "Screens can make it hard to relax").
  - **Fatigue**: 'recommended_workouts': gentle exercises (e.g., yoga, walking; "Light movement boosts energy"); 'recommended_recovery_tips': regular sleep times (e.g., "Sleeping at the same time helps your body rest").
  - **Sedentary job**: 'recommended_workouts': daily movement (e.g., walking, stretching; "Moving keeps your body active"); 'avoid_habits_move': sitting too long (e.g., "Long sitting can make you feel stiff").
  - **High cortisol**: 'recommended_recovery_tips': calming activities (e.g., mindfulness; "This lowers your stress alarm"); 'avoid_habits_rest_recover': caffeine late in the day (e.g., "Coffee at night can keep you awake").
  - **Low estrogen**: 'recommended_foods': phytoestrogen-rich foods (e.g., flaxseeds, soy; "These support hormone balance").
- Verify all arrays are non-empty before finalizing JSON. If user data is missing, include general examples in simple terms (e.g., "No specific symptoms given; eating vegetables supports overall health").

**CRITICAL TEXT MARKING FOR HIGHLIGHTING**
- Use **C1[text]C1** for important information and key action items: critical symptoms (e.g., **C1[fatigue]C1**), diagnoses, key biomarker names, or main action steps (e.g., **C1[eat more fiber]C1**).
- Use **C2[text]C2** for things users need to be aware of: specific biomarker values (e.g., **C2[4.310 uIU/mL]C2**), ranges, "not performed" tests, alerts, or changes (e.g., **C2[high TSH alert]C2**).
- Apply sparingly to key terms for clarity (e.g., **C1[constipation]C1**, **C2[low DHEA]C2**).

**KEY PERSONALIZATION AND SCIENTIFIC LINKING**
- Anchor every recommendation to user input (symptom, diagnosis, biomarker) in simple terms (e.g., "For **C1[constipation]C1**, eat **C1[fiber-rich foods]C1** to help digestion").
- Explicitly reference symptoms/diagnoses (e.g., "Given **C1[stomach pain]C1**, avoid **C2[dairy]C2**").
- For "Not Performed" biomarkers, list in 'crucial_biomarkers_to_measure' with simple explanations (e.g., "Test **C2[DHEA]C2** to check stress because of **C1[feeling tired]C1**").
- Provide women-specific, science-backed rationale in simple language (e.g., "Fiber helps your body clear **C2[extra estrogen]C2** to ease **C1[bloating]C1**, like cleaning out clutter").
- Use precise, clear terms (e.g., "Avoid **C2[milk]C2** for **C1[lactose issues]C1**"). Avoid vague phrases like "some people."
- Explain hormone interactions simply (e.g., "High **C2[cortisol]C2**, your stress alarm, can cause **C1[bloating]C1**").

**Conditional Analysis**
- **With Lab Report**: Analyze all valid biomarkers, categorize by status, link to symptoms/diagnoses in simple terms (e.g., "High **C2[thyroid hormone]C2** may mean your thyroid is slow, causing **C1[tiredness]C1**").
- **Without Lab Report**: Use health assessment only. Recommend biomarkers to test in simple terms (e.g., "For **C1[tiredness]C1**, test **C2[thyroid hormones]C2** to check your energy engine").
- Populate arrays with relevant items. If data is insufficient, use general women’s health guidance, noting it’s general (e.g., "No symptoms given, so try walking for health").

**Four Pillars Scoring**
- Score each pillar (Eat Well, Sleep Well, Move Well, Recover Well) from 1 (needs improvement) to 10 (optimal) based on user data.
- Link scores to inputs in simple terms (e.g., "**C1[skipping meals]C1** gives Eat Well a low score of **C2[4]C2** because regular meals fuel your body").
- For each pillar, provide a 'score_rationale' array with at least 2 sentences in simple language explaining why the score was given, tied to user data (e.g., "Eat Well got **C2[4]C2** because **C1[skipping meals]C1** means missing energy. Eating regularly helps your body stay strong.").
"""

# --- JSON Structure Definition ---
JSON_STRUCTURE_DEFINITION = """
{
  "lab_analysis": {
    "overall_summary": "string - Summarize health status using lab results and health details in simple terms, as if explaining to someone new to health topics. Mention key issues and actions (e.g., 'Your **C1[tiredness]C1** and **C2[high thyroid hormone]C2** suggest checking your thyroid'). If no lab report, note analysis uses only health details. For unreadable lab data, state: 'Some lab results could not be read due to file issues. Advice uses available data.'",
    "biomarkers_tested_count": "integer - Count of 'detailed_biomarkers' objects. Verify it matches the array length. Set to 0 if no lab report.",
    "biomarker_categories_summary": {
      "description_text": "string - Summarize biomarker categories in simple terms (e.g., 'Out of **C1[{biomarkers_tested_count}]C1** tests, **C2[{optimal_count}]C2** are good, **C2[{keep_in_mind_count}]C2** need watching, and **C2[{attention_needed_count}]C2** need action'). Exclude 'Not Performed' tests.",
      "optimal_count": "integer - Count of 'optimal' biomarkers. Set to 0 if no lab report.",
      "keep_in_mind_count": "integer - Count of 'keep_in_mind' biomarkers. Set to 0 if no lab report.",
      "attention_needed_count": "integer - Count of 'attention_needed' biomarkers. Set to 0 if no lab report."
    },
    "detailed_biomarkers": [
      {
        "name": "string - Full biomarker name (e.g., '**C1[Estradiol]C1**').",
        "status": "string - 'optimal', 'keep_in_mind', or 'attention_needed'.",
        "status_label": "string - Simple label (e.g., 'Good (Green)' for optimal).",
        "result": "string - Result with units (e.g., '**C2[4.310 uIU/mL]C2**').",
        "range": "string - Normal range (e.g., '**C2[0.450-4.500]C2**').",
        "cycle_impact": "string - Explain menstrual cycle impact in simple terms or state 'Does not usually change with cycle' (e.g., 'Estradiol changes during your cycle and may affect **C1[cramps]C1**').",
        "why_it_matters": "string - Explain biomarker's role in women’s health, linked to user data, in simple terms (e.g., 'High **C2[thyroid hormone]C2** may cause **C1[tiredness]C1**, like a slow energy engine')."
      }
    ],
    "crucial_biomarkers_to_measure": [
      {
        "name": "string - Biomarker name (e.g., '**C2[DHEA, Serum]C2**').",
        "importance": "string - Simple explanation of why testing is needed (e.g., 'Test **C2[DHEA, Serum]C2** to check stress levels because of **C1[tiredness]C1**')."
      }
    ],
    "health_recommendation_summary": "array of strings - Simple, actionable steps (e.g., 'Retest **C2[DHEA, Serum]C2** to understand **C1[stress]C1**')."
  },
  "four_pillars": {
    "introduction": "string - Summarize health status and lab findings in simple terms for the four areas (eating, sleeping, moving, recovering).",
    "pillars": [
      {
        "name": "string - Pillar name (e.g., '**C1[Eat Well]C1**').",
        "score": "integer - 1 (needs improvement) to 10 (great), based on user data (e.g., '**C2[4]C2** for **C1[skipping meals]C1**').",
        "score_rationale": "array of strings - At least 2 sentences in simple language explaining why the score was given, tied to user data (e.g., ['Eat Well got **C2[4]C2** because **C1[skipping meals]C1** means missing energy.', 'Eating regularly helps your body stay strong.']).",
        "why_it_matters": "string - Explain relevance to user data in simple terms (e.g., 'Good food helps balance **C2[hormones]C2** for **C1[bloating]C1**, like fueling a car').",
        "personalized_recommendations": "array of strings - Simple advice (e.g., 'Eat **C1[fiber-rich vegetables]C1** for **C1[constipation]C1**').",
        "root_cause_correlation": "string - Link to root causes in simple terms (e.g., 'Fiber helps **C1[constipation]C1** caused by **C2[low estrogen]C2**').",
        "science_based_explanation": "string - Simple scientific basis (e.g., 'Fiber clears **C2[extra hormones]C2** to ease **C1[mood swings]C1**, like cleaning out clutter').",
        "additional_guidance": {
          "description": "string - Explain guidance in simple terms, note if general due to limited data (e.g., 'No specific data given, so try these general tips for women’s health').",
          "structure": {
            "recommended_foods": [{"name": "string", "description": "string|null - Simple explanation (e.g., 'Vegetables help digestion')"}],
            "cautious_foods": [{"name": "string", "description": "string|null - Simple explanation (e.g., 'Avoid milk if it upsets your stomach')"}],
            "recommended_workouts": [{"name": "string", "description": "string|null - Simple explanation (e.g., 'Walking boosts energy')"}],
            "avoid_habits_move": [{"name": "string", "description": "string|null - Simple explanation (e.g., 'Don’t sit too long to avoid stiffness')"}],
            "recommended_recovery_tips": [{"name": "string", "description": "string|null - Simple explanation (e.g., 'Deep breathing calms stress')"}],
            "avoid_habits_rest_recover": [{"name": "string", "description": "string|null - Simple explanation (e.g., 'Avoid screens at night for better sleep')"}]
          }
        }
      }
    ]
  },
  "supplements": {
    "description": "string - Explain supplement advice in simple terms based on lab or health data (e.g., 'These supplements may help with **C1[tiredness]C1**').",
    "structure": {
      "recommendations": [
        {
          "name": "string - Supplement name (e.g., '**C1[Magnesium]C1**').",
          "rationale": "string - Simple reason linked to data (e.g., 'For **C2[low Estradiol]C2** and **C1[mood swings]C1**, helps calm your body').",
          "expected_outcomes": "string - Simple benefits (e.g., 'Better **C1[sleep]C1**, like a restful night').",
          "dosage_and_timing": "string - Simple dosage (e.g., '**C2[200 mg daily, evening]C2**').",
          "situational_cyclical_considerations": "string - Simple cycle-specific advice (e.g., 'Use in **C1[second half of cycle]C1** for **C1[cramps]C1**')."
        }
      ],
      "conclusion": "string - Encourage sticking to advice and checking with a doctor, in simple terms."
    }
  },
  "action_plan": {
    "description": "string - Summarize actionable steps in simple terms (e.g., 'Here’s how to improve **C1[energy]C1** and reduce **C1[stress]C1**').",
    "structure": {
      "foods_to_enjoy": "array of strings - Simple food advice (e.g., 'Eat **C1[vegetables]C1** for **C1[constipation]C1**').",
      "foods_to_limit": "array of strings - Simple foods to avoid (e.g., 'Limit **C2[milk]C2** for **C1[stomach issues]C1**').",
      "daily_habits": "array of strings - Simple habits (e.g., 'Sleep **C2[7-9 hours]C2** to balance **C1[stress]C1**').",
      "rest_and_recovery": "array of strings - Simple recovery tips (e.g., 'Try **C1[meditation]C1** for **C1[stress]C1**').",
      "movement": "array of strings - Simple workouts (e.g., '**C1[Yoga]C1** for **C1[stress]C1**')."
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