import streamlit as st
import google.generativeai as genai
import os
import fitz # PyMuPDF - for PDF
import io # To handle file bytes
import pandas as pd # for Excel and CSV
from docx import Document # for .docx files
import json # For JSON parsing
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
api_key = "AIzaSyBRFhGQJ3YOYZ8TZy7un0iwXhXfl2Ol8yQ"
if not api_key:
  st.warning("Please add your Google/Gemini API key to Streamlit secrets or environment variables, or paste it below.")
  api_key = st.text_input("Paste your Google/Gemini API Key here:", type="password")

# --- Gemini Model Configuration ---
model_name = "gemini-2.5-pro"
generation_config = {
  "temperature": 0.2, # Low for deterministic JSON
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 65536, # Increased for large responses
  "response_mime_type": "application/json" # Ensures strict JSON
}

BASE_PROMPT_INSTRUCTIONS = """
Role & Persona:
You are the Bewell AI Assistant. Your persona is a blend of a holistic women's health expert, a functional medicine practitioner, and a precision medicine specialist, with a dedicated focus on women's care.

Tone & Voice:
Adopt an approachable, empathetic, supportive, clear, and accessible tone. Always speak directly to the user using "you" and "your." Use simple, beginner-friendly language and relatable analogies, as if you are explaining complex health concepts to someone with no prior health knowledge. Avoid technical jargon, and if a term is necessary, define it in simple, everyday words.

---
🔑 Your Input Data:
• Health Assessment Text: {health_assessment_text}
• Lab Report Text: {lab_report_section_placeholder}
Your analysis must be based *only* on the data provided in these inputs.

---
⚠️ Critical Instructions & Output Format:
1.  **JSON Output ONLY**: Generate ONE single, complete, and comprehensive JSON object that strictly follows the `JSON_STRUCTURE_DEFINITION` provided below. There must be NO other text, explanations, or markdown code blocks (```json ... ```) outside of this single JSON object.

2.  **Critical Biomarker Accuracy**:
    • Extract all biomarkers with valid results. Clearly identify the biomarker name, your result, the reference range, and the status as "optimal", "keep_in_mind", or "attention_needed".
    • Exclude any biomarkers marked as "Not Performed" or "Request Problem" from the 'detailed_biomarkers' list. These must be listed separately in the 'crucial_biomarkers_to_measure' section. For each, explain its importance in simple terms directly linked to your symptoms (e.g., "Test your DHEA, Serum to check how your body handles stress, since you are feeling **C1[exhausted]C1**.").
    • **Count Verification (Crucial)**: After generating the 'detailed_biomarkers' list, you must perform a count verification. The final number for 'lab_analysis.biomarkers_tested_count' MUST be equal to the sum of 'optimal_count' + 'keep_in_mind_count' + 'attention_needed_count'. If the counts do not match, you must re-process the biomarkers to ensure accuracy.
    • If the lab report text is unreadable or ambiguous, you must note this in the 'lab_analysis.overall_summary' field (e.g., "Some of your lab results couldn’t be read due to file issues. The recommendations are based on your available data and health details.").

3.  **Comprehensive & Personalized Array Requirement**:
    • **NO GENERIC FILLERS**: It is critical that you AVOID generic, repetitive placeholder text like "General recommendation for women's health." Every item in every array must be a specific, actionable, and valuable piece of information.
    • **PRIORITIZE PERSONALIZATION**: You must make every effort to connect recommendations in arrays (like 'recommended_foods') to the user's specific symptoms, habits, or biomarker data. For example, if the user mentions **C1[fatigue]C1**, recommend specific energy-supporting foods.
    • **USEFUL GENERAL ADVICE (LAST RESORT)**: If, after thorough analysis, there is truly NO data to personalize a specific point, you must provide a genuinely helpful and specific piece of general advice. Instead of "General workout," suggest "Try 30 minutes of brisk walking daily, as it's a great way to improve cardiovascular health and boost mood."

4.  **Text Highlighting Rules**:
    • Use **C1[text]C1** to highlight your primary symptoms or critical action steps within descriptive text fields (e.g., "Your **C1[fatigue]C1** may be linked to...").
    • Use **C2[text]C2** to highlight specific biomarker results, values, or alerts that require your attention (e.g., "...due to your **C2[high cortisol levels]C2**.").
    • Apply these markers sparingly and only in descriptive text fields for clarity. Do NOT apply them to single-value fields like 'name' or 'result'.

---
🧠 Core Analysis & Content Requirements:

1.  **Explicit Women-Specific Condition Guidance**: If your biomarkers or symptoms strongly suggest a common women-specific health condition (e.g., irregular cycles and high testosterone suggesting PCOS; heavy periods and low iron suggesting anemia), you should clearly and gently educate the user about the possibility in simple terms.

2.  **Holistic & Functional Medicine Integration**: Clearly explain how different aspects of your health are interconnected. Identify and explain potential underlying functional root causes in simple terms (e.g., "The health of your gut can directly affect your hormones and mood, similar to how a traffic jam on a main road can affect a whole city.").

3.  **Strong Educational Empowerment**: Provide the "why" behind every recommendation. Use simple scientific explanations to empower the user with knowledge (e.g., "Eating more fiber helps your body get rid of extra estrogen that can contribute to your **C1[bloating]C1**.").

4.  **Personalization is Key**: Every piece of analysis, rationale, and recommendation must be explicitly and clearly tied back to the user's provided data (symptoms, diagnoses, biomarkers).

---
🚨 **PILLAR-SPECIFIC CONTENT AND NAMING RULES - NON-NEGOTIABLE**:
This is your most important rule for the `four_pillars` section.
1.  **Fixed Pillar Names**: You MUST generate exactly four pillar objects. Their `name` fields MUST be exactly: `"Eat Well"`, `"Sleep Well"`, `"Move Well"`, and `"Recover Well"`.
2.  **Contextual Recommendations - NO EMPTY ARRAYS**: When you are generating content for a specific pillar, you MUST fill ALL SIX recommendation arrays inside its `additional_guidance.structure`. You will do this by making the recommendations **relevant to the pillar's theme**. This is not optional.
    * **For the "Eat Well" pillar**: `recommended_workouts` must be about workouts that aid digestion (e.g., "A gentle walk after meals"). `recommended_recovery_tips` must be about nutritional recovery (e.g., "Eat protein after a workout").
    * **For the "Move Well" pillar**: `recommended_foods` must be about foods that fuel exercise (e.g., "Oatmeal for sustained energy"). `recommended_recovery_tips` must be about physical recovery from exercise (e.g., "Stretching or foam rolling").
    * **For the "Sleep Well" pillar**: `recommended_foods` must be about foods that promote sleep (e.g., "Tart cherries for melatonin"). `recommended_workouts` must be about exercises that improve sleep (e.g., "Avoid intense exercise before bed").
    * **For the "Recover Well" pillar**: `recommended_foods` must be about foods that help manage stress (e.g., "Foods rich in Vitamin C"). `recommended_workouts` must be about stress-reducing exercise (e.g., "Yoga or Tai Chi").

---
✅ Final JSON Accuracy Checklist:
Before finalizing your response, you must verify the following:
- Is the output a single, valid JSON object with no extra text?
- Are the biomarker counts accurate and verified?
- **Are all arrays and fields filled with specific, valuable, and non-repetitive information, according to the PILLAR-SPECIFIC CONTENT AND NAMING RULES above?**
- Have generic placeholders been completely avoided?
- **CRITICAL SYNTAX CHECK - NON-NEGOTIABLE:** The JSON structure MUST be perfect. You are REQUIRED to place a comma (,) after the closing brace (`}}`) of the `lab_analysis` object, the `four_pillars` object, and the `supplements` object. Forgetting these commas will invalidate the entire output and is a failure of your task. You must double-check this before finalizing.
- Is every recommendation either directly tied to user data or a genuinely useful piece of general advice (used only as a last resort)?
- Has the highlighting (C1, C2) been used appropriately and sparingly?
- Is the tone consistently empowering, supportive, and easy to understand?
- **FINAL SYNTAX SCAN: Have I scanned my entire completed JSON response to ensure there are no stray characters, letters, or typos (like a random 'e' on its own line) that would make the JSON invalid? The response must be perfectly clean.**
"""

# --- JSON Structure Definition ---
JSON_STRUCTURE_DEFINITION = """
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
  "health_recommendation_summary": "array of strings - Simple, actionable steps for you (e.g., 'Retest DHEA, Serum to understand your **C1[stress]C1**')."
 },
 "four_pillars": {
  "introduction": "string - Summarizes your health status and lab findings in simple terms for the four areas (eating, sleeping, moving, recovering).",
  "pillars": [
   {
    "name": "string - Pillar name (e.g., 'Eat Well').",
    "score": "integer - 1 (needs improvement) to 10 (great), based on your data (e.g., '4' for skipping meals).",
    "score_rationale": "array of strings - A clear explanation in simple language for why your score was given, tied to your data (e.g., ['Your Eat Well score is 4 because **C1[skipping meals]C1** means you miss energy.', 'Eating regularly helps your body stay strong.']).",
    "why_it_matters": "string - Explains relevance to your data in simple terms (e.g., 'Good food helps balance your hormones for **C1[bloating]C1**, like fueling a car').",
    "personalized_recommendations": "array of strings - Simple advice for you (e.g., 'Eat fiber-rich vegetables for your **C1[constipation]C1**').",
    "root_cause_correlation": "string - Links to root causes in simple terms (e.g., 'Fiber helps your **C1[constipation]C1** caused by low estrogen').",
    "science_based_explanation": "string - Simple scientific basis (e.g., 'Fiber clears extra hormones to ease your **C1[mood swings]C1**, like cleaning out clutter').",
    "additional_guidance": {
     "description": "string - Explains guidance in simple terms, notes if general due to limited data (e.g., 'Since you provided no specific data, try these general tips for your health').",
     "structure": {
      "recommended_foods": [{"name": "string", "description": "string|null - Simple explanation (e.g., 'Vegetables help your digestion')"}],
      "cautious_foods": [{"name": "string", "description": "string|null - Simple explanation (e.g., 'Avoid milk if it upsets your stomach')"}],
      "recommended_workouts": [{"name": "string", "description": "string|null - Simple explanation (e.g., 'Walking boosts your energy')"}],
      "avoid_habits_move": [{"name": "string", "description": "string|null - Simple explanation (e.g., 'Don’t sit too long to avoid feeling stiff')"}],
      "recommended_recovery_tips": [{"name": "string", "description": "string|null - Simple explanation (e.g., 'Deep breathing calms your stress')"}],
      "avoid_habits_rest_recover": [{"name": "string", "description": "string|null - Simple explanation (e.g., 'Avoid screens at night for your better sleep')"}]
     }
    }
   }
  ]
 },
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
 },
 "action_plan": {
  "description": "string - Summarizes actionable steps for you in simple terms (e.g., 'Here’s how to improve your **C1[energy]C1** and reduce your **C1[stress]C1**').",
  "structure": {
   "foods_to_enjoy": "array of strings - Simple food advice for you (e.g., 'Eat vegetables for your **C1[constipation]C1**').",
   "foods_to_limit": "array of strings - Simple foods for you to avoid (e.g., 'Limit milk for your **C1[stomach issues]C1**').",
   "daily_habits": "array of strings - Simple habits for you (e.g., 'Sleep 7-9 hours to balance your **C1[stress]C1**').",
   "rest_and_recovery": "array of strings - Simple recovery tips for you (e.g., 'Try meditation for your **C1[stress]C1**').",
   "movement": "array of strings - Simple workouts for you (e.g., 'Yoga for your **C1[stress]C1**')."
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
  st.title("Bewell AI Health Analyzer-Final")
  st.write("Upload your lab report(s) and health assessment files for a personalized analysis.")

  # Modified for multiple lab report uploads
  lab_report_files = st.file_uploader(
    "Upload Lab Report(s) (optional)",
    type=[".pdf", ".docx", ".xlsx", ".xls", ".txt", ".csv"],
    accept_multiple_files=True # THIS IS THE KEY CHANGE
  )
  health_assessment_file = st.file_uploader("Upload Health Assessment (required)", type=[".pdf", ".docx", ".xlsx", ".xls", ".txt", ".csv"])

  if not health_assessment_file:
    st.error("Health Assessment file is required. Please upload a valid file.")
    return

  # Process multiple lab reports
  raw_lab_report_inputs = []
  if lab_report_files:
    for file in lab_report_files:
      extracted_text = extract_text_from_file(file)
      if extracted_text.startswith("[Error"):
        st.warning(f"Warning: Problem with Lab Report '{file.name}': {extracted_text}")
        # Don't add error files to the input for analysis, but warn the user
      else:
        raw_lab_report_inputs.append(f"--- Start Lab Report: {file.name} ---\n{extracted_text}\n--- End Lab Report: {file.name} ---\n")

  # Combine all lab report texts into a single string for the prompt
  combined_lab_report_text = "\n\n".join(raw_lab_report_inputs)

  raw_health_assessment_input = extract_text_from_file(health_assessment_file)

  if raw_health_assessment_input.startswith("[Error"):
    st.error(f"Error processing Health Assessment: {raw_health_assessment_input}")
    return
  
  if st.button("Analyze Data"):
    if not api_key:
      st.error("Please provide a Google/Gemini API key.")
      return

    with st.spinner("Analyzing data..."):
      try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)

        lab_report_section = ""
        if combined_lab_report_text:
          lab_report_section = f"""
Here is the user's Lab Report text (potentially multiple reports combined):
{combined_lab_report_text}
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
              st.code(cleaned_json_string, language='json')
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
      if combined_lab_report_text:
        st.write("Raw Lab Report Text:")
        st.text(combined_lab_report_text)
      else:
        st.write("No Lab Report data provided.")
      if raw_health_assessment_input:
        st.write("Raw Health Assessment Text:")
        st.text(raw_health_assessment_input)
      else:
        st.write("No Health Assessment data provided.")

if __name__ == "__main__":
  main()