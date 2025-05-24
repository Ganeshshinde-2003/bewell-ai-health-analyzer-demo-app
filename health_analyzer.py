import argparse
import google.generativeai as genai
import os
import fitz  # PyMuPDF - for PDF
import io  # To handle file bytes
import pandas as pd  # for Excel and CSV
from docx import Document  # for .docx files
import re

# --- Text Extraction Function (Handles multiple file types) ---
def extract_text_from_file(file_path):
    """
    Extracts text content from various file types (PDF, DOCX, XLSX, XLS, TXT, CSV).
    Returns extracted text as a string or an error marker.
    """
    if not file_path or not os.path.exists(file_path):
        return ""

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
                print(f"Error processing PDF file: {e}")
                return "[Error Processing PDF File]"

        elif file_extension == ".docx":
            try:
                doc = Document(io.BytesIO(file_bytes))
                for para in doc.paragraphs:
                    text += para.text + "\n"
            except Exception as e:
                print(f"Error processing DOCX file: {e}")
                return "[Error Processing DOCX File]"

        elif file_extension in [".xlsx", ".xls"]:
            try:
                excel_data = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
                for sheet_name, df in excel_data.items():
                    text += f"--- Sheet: {sheet_name} ---\n"
                    text += df.to_string(index=False) + "\n\n"
            except Exception as excel_e:
                print(
                    f"Could not read Excel file '{file_path}'. Ensure 'openpyxl' (for .xlsx) or 'xlrd' (for .xls) is installed. Error: {excel_e}")
                return f"[Could not automatically process Excel file {file_path} - Please try saving as PDF/TXT.]"

        elif file_extension in [".txt", ".csv"]:
            text = file_bytes.decode('utf-8', errors='ignore')

        else:
            print(f"Unsupported file type: {file_extension} for file '{file_path}'")
            return "[Unsupported File Type]"

    except Exception as e:
        print(f"An error occurred while processing '{file_path}': {e}")
        return "[Error Processing File]"

    if not text.strip() and file_extension not in [".txt", ".csv"]:
        print(
            f"No readable text extracted from '{file_path}'. The file might be scanned, empty, or have complex formatting. Consider using a text-based format.")
    return text

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

def main():
    # --- Parse Command-Line Arguments ---
    parser = argparse.ArgumentParser(description="Bewell AI Health Analyzer: Analyze health assessment and lab report files.")
    parser.add_argument("--health-assessment", required=True, help="Path to the health assessment file (PDF, DOCX, XLSX, XLS, TXT, CSV)")
    parser.add_argument("--lab-report", help="Path to the lab report file (PDF, DOCX, XLSX, XLS, TXT, CSV, optional)")
    parser.add_argument("--api-key", help="Google/Gemini API key (optional, defaults to GOOGLE_API_KEY env variable)")
    args = parser.parse_args()

    # --- API Key Handling ---
    api_key = "AIzaSyArQ9zeya1SO-IwsMappkLStXYT0W7WXfk"
    if not api_key:
        print("Error: Please provide a Google/Gemini API key via --api-key or set GOOGLE_API_KEY environment variable.")
        return

    # --- Process Files ---
    raw_health_assessment_input = extract_text_from_file(args.health_assessment)
    raw_lab_report_input = extract_text_from_file(args.lab_report) if args.lab_report else ""

    if raw_health_assessment_input.startswith("[Error"):
        print(f"Error processing Health Assessment: {raw_health_assessment_input}")
        return
    if raw_lab_report_input.startswith("[Error"):
        print(f"Warning: Problem with Lab Report: {raw_lab_report_input}")
        print("Continuing analysis without lab report data.")
        raw_lab_report_input = ""

    # --- Generate Analysis ---
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

        print("Generating analysis...")
        response = model.generate_content(prompt)

        if response and response.text is not None:
            cleaned_json_string = clean_json_string(response.text)
            if cleaned_json_string.strip():
                print("\n=== Cleaned Raw Model Response ===")
                print(cleaned_json_string)
            else:
                print("Error: Model returned an empty response.")
                print("\n=== Cleaned Raw Model Response ===")
                print("Empty response received from the model.")
        elif response and response.prompt_feedback:
            print(f"Error: Analysis blocked. Reason: {response.prompt_feedback.block_reason}")
            if response.prompt_feedback.safety_ratings:
                print(f"Safety Ratings: {response.prompt_feedback.safety_ratings}")
            print("Please review your input and try again.")
        else:
            print("Error: Failed to generate a valid response.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Check API key, network, and input files.")

if __name__ == "__main__":
    main()