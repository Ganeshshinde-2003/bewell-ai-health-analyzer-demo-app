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

Generate ONE complete JSON object following the JSON_STRUCTURE_DEFINITION below. Do NOT include text outside the JSON. Do NOT wrap in markdown code blocks (```json ... ``` or ``` ... ```). Fill all descriptive fields with 5-7 sentences using simple, everyday words and vivid analogies. Ensure all arrays have 2+ items with detailed, beginner-friendly explanations tied to user data (symptoms, diagnoses, biomarkers). If data is missing, use general women’s health tips and note they’re general (e.g., "Since no symptoms were shared, eating colorful fruits can boost your energy like charging a battery"). Make every explanation clear, relatable, and empowering.

**CRITICAL ACCURACY REQUIREMENT: BIOMARKER COUNTING**
1. **Process All Biomarkers**: Extract all biomarkers with valid results from the lab report, identifying names, results, ranges, and statuses ("optimal", "keep_in_mind", "attention_needed"). Exclude "Not Performed" or "Request Problem" biomarkers from 'detailed_biomarkers'; list these in 'crucial_biomarkers_to_measure' with symptom-linked explanations in simple terms (e.g., "Test DHEA to check stress levels because of tiredness").
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
- Use **C1[text]C1** for important information and key action items in descriptive text fields only: critical symptoms (e.g., **C1[fatigue]C1**), diagnoses, or main action steps (e.g., **C1[eat more fiber]C1**) within sentences. Do NOT apply to single-value fields like 'name', 'result', or 'range', or to names (e.g., do NOT use for 'Estradiol', 'Magnesium', 'DHEA, Serum'). For example, in 'detailed_biomarkers.name', use plain 'Estradiol', not '**C1[Estradiol]C1**'.
- Use **C2[text]C2** for things users need to be aware of in descriptive text fields only: specific values, ranges, "not performed" tests, alerts, or changes (e.g., **C2[high thyroid hormone alert]C2**) within sentences. Do NOT apply to single-value fields like 'name', 'result', or 'range', or to names (e.g., do NOT use for 'DHEA, Serum'). For example, in 'crucial_biomarkers_to_measure.importance', use plain 'DHEA, Serum' in the sentence, like "Test DHEA, Serum to check stress levels because of **C1[feeling tired]C1**".
- Apply sparingly to key terms in descriptive text fields (e.g., 'overall_summary', 'why_it_matters', 'score_rationale') for clarity (e.g., **C1[constipation]C1**, **C2[low DHEA alert]C2**).

**KEY PERSONALIZATION AND SCIENTIFIC LINKING**
- Anchor every recommendation to user input (symptom, diagnosis, biomarker) in simple terms (e.g., "For **C1[constipation]C1**, eat fiber-rich foods to help digestion").
- Explicitly reference symptoms/diagnoses in descriptive fields (e.g., "Given **C1[stomach pain]C1**, avoid dairy").
- For "Not Performed" biomarkers, list in 'crucial_biomarkers_to_measure' with simple explanations (e.g., "Test DHEA, Serum to check stress levels because of **C1[feeling tired]C1**").
- Provide women-specific, science-backed rationale in simple language (e.g., "Fiber helps your body clear extra estrogen to ease **C1[bloating]C1**, like cleaning out clutter").
- Use precise, clear terms (e.g., "Avoid dairy for **C1[lactose issues]C1**"). Avoid vague phrases like "some people."
- Explain hormone interactions simply (e.g., "High cortisol, your stress alarm, can cause **C1[bloating]C1**").

**Conditional Analysis**
- **With Lab Report**: Analyze all valid biomarkers, categorize by status, link to symptoms/diagnoses in simple terms (e.g., "High thyroid hormone may mean your thyroid is slow, causing **C1[tiredness]C1**").
- **Without Lab Report**: Use health assessment only. Recommend biomarkers to test in simple terms (e.g., "For **C1[tiredness]C1**, test thyroid hormones to check your energy engine").
- Populate arrays with relevant items. If data is insufficient, use general women’s health guidance, noting it’s general (e.g., "No symptoms given, so try walking for health").

**Four Pillars Scoring**
- Score each pillar (Eat Well, Sleep Well, Move Well, Recover Well) from 1 (needs improvement) to 10 (optimal) based on user data.
- Link scores to inputs in simple terms (e.g., "**C1[skipping meals]C1** gives Eat Well a low score of 4 because regular meals fuel your body").
- For each pillar, provide a 'score_rationale' array with at least 2 sentences in simple language explaining why the score was given, tied to user data (e.g., "Eat Well got 4 because **C1[skipping meals]C1** means missing energy. Eating regularly helps your body stay strong.")
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
        "name": "string - Full biomarker name (e.g., 'Estradiol').",
        "status": "string - 'optimal', 'keep_in_mind', or 'attention_needed'.",
        "status_label": "string - Simple label (e.g., 'Good (Green)' for optimal).",
        "result": "string - Result with units (e.g., '4.310 uIU/mL').",
        "range": "string - Normal range (e.g., '0.450-4.500').",
        "cycle_impact": "string - Explain menstrual cycle impact in simple terms in 1 sentence (e.g., 'Estradiol changes during your cycle and may affect **C1[cramps]C1**').",
        "why_it_matters": "string - Explain biomarker's role in women’s health, linked to user data, in simple terms (e.g., 'High thyroid hormone may cause **C1[tiredness]C1**, like a slow energy engine')."
      }
    ],
    "crucial_biomarkers_to_measure": [
      {
        "name": "string - Biomarker name (e.g., 'DHEA, Serum').",
        "importance": "string - Simple explanation of why testing is needed (e.g., 'Test DHEA, Serum to check stress levels because of **C1[tiredness]C1**')."
      }
    ],
    "health_recommendation_summary": "array of strings - Simple, actionable steps (e.g., 'Retest DHEA, Serum to understand **C1[stress]C1**')."
  },
  "four_pillars": {
    "introduction": "string - Summarize health status and lab findings in simple terms for the four areas (eating, sleeping, moving, recovering).",
    "pillars": [
      {
        "name": "string - Pillar name (e.g., 'Eat Well').",
        "score": "integer - 1 (needs improvement) to 10 (great), based on user data (e.g., '4' for skipping meals).",
        "score_rationale": "array of strings - At least 2 sentences in simple language explaining why the score was given, tied to user data (e.g., ['Eat Well got 4 because **C1[skipping meals]C1** means missing energy.', 'Eating regularly helps your body stay strong.']).",
        "why_it_matters": "string - Explain relevance to user data in simple terms (e.g., 'Good food helps balance hormones for **C1[bloating]C1**, like fueling a car').",
        "personalized_recommendations": "array of strings - Simple advice (e.g., 'Eat fiber-rich vegetables for **C1[constipation]C1**').",
        "root_cause_correlation": "string - Link to root causes in simple terms (e.g., 'Fiber helps **C1[constipation]C1** caused by low estrogen').",
        "science_based_explanation": "string - Simple scientific basis (e.g., 'Fiber clears extra hormones to ease **C1[mood swings]C1**, like cleaning out clutter').",
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
          "name": "string - Supplement name (e.g., 'Magnesium').",
          "rationale": "string - Simple reason linked to data (e.g., 'For low Estradiol and **C1[mood swings]C1**, helps calm your body').",
          "expected_outcomes": "string - Simple benefits (e.g., 'Better **C1[sleep]C1**, like a restful night').",
          "dosage_and_timing": "string - Simple dosage (e.g., '200 mg daily, evening').",
          "situational_cyclical_considerations": "string - Simple cycle-specific advice (e.g., 'Use in the second half of your cycle for **C1[cramps]C1**')."
        }
      ],
      "conclusion": "string - Encourage sticking to advice and checking with a doctor, in simple terms."
    }
  },
  "action_plan": {
    "description": "string - Summarize actionable steps in simple terms (e.g., 'Here’s how to improve **C1[energy]C1** and reduce **C1[stress]C1**').",
    "structure": {
      "foods_to_enjoy": "array of strings - Simple food advice (e.g., 'Eat vegetables for **C1[constipation]C1**').",
      "foods_to_limit": "array of strings - Simple foods to avoid (e.g., 'Limit milk for **C1[stomach issues]C1**').",
      "daily_habits": "array of strings - Simple habits (e.g., 'Sleep 7-9 hours to balance **C1[stress]C1**').",
      "rest_and_recovery": "array of strings - Simple recovery tips (e.g., 'Try meditation for **C1[stress]C1**').",
      "movement": "array of strings - Simple workouts (e.g., 'Yoga for **C1[stress]C1**')."
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