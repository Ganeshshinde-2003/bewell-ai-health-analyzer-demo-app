import streamlit as st
import os
import json
import fitz
import io
import pandas as pd
from docx import Document
import re
import time

# --- Vertex AI imports ---
import vertexai
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
from google.oauth2 import service_account

# Page config
st.set_page_config(
    page_title="Bewell AI Health Analyzer - HIPAA by Vertex AI - VP1",
    page_icon="🌿",
    layout="wide"
)

PROJECT_ID = st.secrets.get("PROJECT_ID")
LOCATION = st.secrets.get("LOCATION", "us-central1")

try:
    credentials_json_string = st.secrets.get("google_credentials")

    if not credentials_json_string:
        st.error("Google Cloud credentials not found in Streamlit secrets. Please configure 'google_credentials'.")
        st.stop()

    credentials_dict = json.loads(credentials_json_string)
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)

    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
    st.success("✅ Vertex AI initialized successfully!")

except Exception as e:
    st.error(f"❌ Failed to initialize Vertex AI. Please check your Streamlit secrets and project settings.\n\nError: {e}")
    st.stop()

MODEL_NAME = "gemini-2.5-flash-lite"

# Safety settings for production
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

RESPONSE_SCHEMA_BIOMARKERS = {
    "type":"object",
    "properties":{
        "lab_analysis":{
            "type":"object",
            "properties":{
                "overall_summary":{"type":"string"},
                "biomarkers_tested_count":{"type":"integer"},
                "biomarker_categories_summary":{
                    "type":"object",
                    "properties":{
                        "description_text":{"type":"string"},
                        "optimal_count":{"type":"integer"},
                        "keep_in_mind_count":{"type":"integer"},
                        "attention_needed_count":{"type":"integer"}
                    },
                    "required":["description_text","optimal_count","keep_in_mind_count","attention_needed_count"],
                    "additionalProperties": False
                },
                "detailed_biomarkers":{
                    "type":"array",
                    "items":{
                        "type":"object",
                        "properties":{
                            "name":{"type":"string"},
                            "status":{"type":"string"},
                            "status_label":{"type":"string"},
                            "result":{"type":"string"},
                            "range":{"type":"string"},
                            "cycle_impact":{"type":"string"},
                            "why_it_matters":{"type":"string"}
                        },
                        "required":["name","status","status_label","result","range","cycle_impact","why_it_matters"],
                        "additionalProperties": False
                    }
                },
                "crucial_biomarkers_to_measure":{
                    "type":"array",
                    "items":{
                        "type":"object",
                        "properties":{
                            "name":{"type":"string"},
                            "importance":{"type":"string"}
                        },
                        "required":["name","importance"],
                        "additionalProperties": False
                    }
                },
                "health_recommendation_summary":{
                    "type":"array",
                    "items":{"type":"string"}
                }
            },
            "required":["overall_summary","biomarkers_tested_count","biomarker_categories_summary","detailed_biomarkers","crucial_biomarkers_to_measure","health_recommendation_summary"],
            "additionalProperties": False
        }
    },
    "required":["lab_analysis"],
    "additionalProperties": False
}

RESPONSE_SCHEMA_4PILLARS = {
    "type":"object",
    "properties":{
        "four_pillars":{
            "type":"object",
            "properties":{
                "introduction":{"type":"string"},
                "pillars":{
                    "type":"array",
                    "items":{
                        "type":"object",
                        "properties":{
                            "name":{"type":"string"},
                            "score":{"type":"integer"},
                            "score_rationale":{
                                "type":"array",
                                "items":{"type":"string"}
                            },
                            "why_it_matters":{"type":"string"},
                            "root_cause_correlation":{"type":"string"},
                            "science_based_explanation":{"type":"string"},
                            "additional_guidance":{
                                "type":"object",
                                "properties":{
                                    "description":{"type":"string"},
                                    "structure":{"type":"object"}
                                },
                                "required":["description","structure"],
                                "additionalProperties": True
                            }
                        },
                        "required":["name","score","score_rationale","why_it_matters","root_cause_correlation","science_based_explanation","additional_guidance"],
                        "additionalProperties": False
                    }
                }
            },
            "required":["introduction","pillars"],
            "additionalProperties": False
        }
    },
    "required":["four_pillars"],
    "additionalProperties": False
}

RESPONSE_SCHEMA_SUPPLEMENTS = {
    "type":"object",
    "properties":{
        "supplements":{
            "type":"object",
            "properties":{
                "description":{"type":"string"},
                "structure":{
                    "type":"object",
                    "properties":{
                        "recommendations":{
                            "type":"array",
                            "items":{
                                "type":"object",
                                "properties":{
                                    "name":{"type":"string"},
                                    "rationale":{"type":"string"},
                                    "expected_outcomes":{"type":"string"},
                                    "dosage_and_timing":{"type":"string"},
                                    "situational_cyclical_considerations":{"type":"string"}
                                },
                                "required":["name","rationale","expected_outcomes","dosage_and_timing","situational_cyclical_considerations"],
                                "additionalProperties": False
                            }
                        },
                        "conclusion":{"type":"string"}
                    },
                    "required":["recommendations","conclusion"],
                    "additionalProperties": False
                }
            },
            "required":["description","structure"],
            "additionalProperties": False
        }
    },
    "required":["supplements"],
    "additionalProperties": False
}

def build_generation_config(schema_dict):
    return {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
        "response_schema": schema_dict
    }

@st.cache_data(show_spinner=False)
def extract_text_from_file(uploaded_file):
    if uploaded_file is None:
        return ""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    text = ""
    try:
        file_bytes = uploaded_file.getvalue()
        if file_extension == ".pdf":
            pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                text += page.get_text("text") + "\n"
            pdf_document.close()
        elif file_extension == ".docx":
            doc = Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file_extension in [".xlsx", ".xls"]:
            excel_data = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
            for sheet_name, df in excel_data.items():
                text += f"--- Sheet: {sheet_name} ---\n"
                text += df.to_string(index=False) + "\n\n"
        elif file_extension in [".txt", ".csv"]:
            text = file_bytes.decode('utf-8', errors='ignore')
        else:
            st.warning(f"Unsupported file type: {file_extension} for file '{uploaded_file.name}'")
            return "[Unsupported File Type Uploaded]"
    except Exception as e:
        st.error(f"An error occurred while processing '{uploaded_file.name}': {e}")
        return "[Error Processing File]"
    if not text.strip() and file_extension not in [".txt", ".csv"]:
        st.warning(f"No readable text extracted from '{uploaded_file.name}'. The file might be scanned, empty, or have complex formatting. Consider pasting the text manually.")
    return text

def clean_json_string(json_string):
    if not isinstance(json_string, str):
        return json_string
    stripped_string = re.sub(r'^``````\s*$', '', json_string.strip(), flags=re.MULTILINE)
    stripped_string = re.sub(r',\s*}', '}', stripped_string)
    stripped_string = re.sub(r',\s*]', ']', stripped_string)
    return stripped_string

def call_gemini_with_retry(prompt, schema_dict, max_retries=3):
    raw_response = ""
    for attempt in range(max_retries):
        if attempt > 0:
            st.info(f"Attempt {attempt + 1} is ongoing...")
            time.sleep(1)
        try:
            generation_config = build_generation_config(schema_dict)
            model = GenerativeModel(MODEL_NAME, generation_config=generation_config, safety_settings=safety_settings)
            response = model.generate_content(prompt)
            raw_response = response.text

            if raw_response:
                cleaned_str = clean_json_string(raw_response)
                return json.loads(cleaned_str), raw_response
            else:
                raise ValueError("Model returned an empty response.")
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed (Invalid JSON). Retrying...")
                time.sleep(2) 
            else:
                st.error(f"Attempt {attempt + 1} failed. JSON decode Error: {e}")
                st.code(raw_response, language="json", label="Raw response causing JSON error")
                raise Exception(f"Failed JSON decode after {max_retries} attempts: {e}")
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed (Unexpected error: {e}). Retrying...")
                time.sleep(2)
            else:
                st.error(f"Attempt {attempt + 1} failed. Unexpected error: {e}")
                raise
    raise Exception("Maximum retries reached without success.")

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

JSON_STRUCTURE_BIOMARKERS = """
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

JSON_STRUCTURE_4PILLARS = """
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

JSON_STRUCTURE_SUPPLEMENTS_ACTIONS = """
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


def main():
    st.title("🌿 Your Personal AI Health Assistant – HIPAA Secure by Bewell + Vertex AI (PV+2)")
    st.write("Upload your lab report(s) and health assessment files for a personalized analysis.")

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
        raw_lab_texts = []
        if lab_report_files:
            for file in lab_report_files:
                extracted_text = extract_text_from_file(file)
                if "[Error" in extracted_text or "[Unsupported" in extracted_text:
                    st.warning(f"Problem with Lab Report '{file.name}': {extracted_text}")
                else:
                    raw_lab_texts.append(f"--- Start Lab Report: {file.name} ---\n{extracted_text}\n--- End Lab Report: {file.name} ---")

        combined_lab_report_text = "\n\n".join(raw_lab_texts)

        raw_health_assessment = extract_text_from_file(health_assessment_file)
        if "[Error" in raw_health_assessment:
            st.error(f"Critical Error processing Health Assessment: {raw_health_assessment}")
            st.stop()

        lab_report_section = f"""
Here is the user's Lab Report text (potentially multiple reports combined):
{combined_lab_report_text}
""" if combined_lab_report_text else "No Lab Report text was provided. Analysis will be based solely on Health Assessment data."

        # Biomarkers
        biomarker_prompt = BASE_PROMPT_COMMON.format(
            health_assessment_text=raw_health_assessment,
            lab_report_section_placeholder=lab_report_section
        ) + "--- Specific Instructions for Biomarker Analysis ---\n" + JSON_STRUCTURE_BIOMARKERS
        biomarker_data, biomarker_raw = call_gemini_with_retry(biomarker_prompt, RESPONSE_SCHEMA_BIOMARKERS)

        # Four Pillars
        four_pillars_prompt = BASE_PROMPT_COMMON.format(
            health_assessment_text=raw_health_assessment,
            lab_report_section_placeholder=lab_report_section
        ) + "--- Specific Instructions for Four Pillars Analysis ---\n" + JSON_STRUCTURE_4PILLARS
        four_pillars_data, four_pillars_raw = call_gemini_with_retry(four_pillars_prompt, RESPONSE_SCHEMA_4PILLARS)

        # Supplements
        supplements_prompt = BASE_PROMPT_COMMON.format(
            health_assessment_text=raw_health_assessment,
            lab_report_section_placeholder=lab_report_section
        ) + "--- Specific Instructions for Supplements and Action Items Analysis ---\n" + JSON_STRUCTURE_SUPPLEMENTS_ACTIONS
        supplements_data, supplements_raw = call_gemini_with_retry(supplements_prompt, RESPONSE_SCHEMA_SUPPLEMENTS)

        final_output = {}
        final_output.update(biomarker_data)
        final_output.update(four_pillars_data)
        final_output.update(supplements_data)

        st.header("🔬 Your Personalized Bewell Analysis (Interactive JSON):")
        st.json(final_output, expanded=True)

        st.markdown("---")
        st.header("📋 Copy Full JSON Output (Plain Text):")
        st.text_area("Select the text below and copy it to your clipboard:",
                     json.dumps(final_output, indent=2),
                     height=400, disabled=True)

        with st.expander("Show All Debug Information (Raw Responses & Prompts)"):
            combined_debug_output = []
            combined_debug_output.append(f"--- Biomarkers Raw ---\n{biomarker_raw}\n--- End Biomarkers Raw ---\n\n")
            combined_debug_output.append(f"--- Four Pillars Raw ---\n{four_pillars_raw}\n--- End Four Pillars Raw ---\n\n")
            combined_debug_output.append(f"--- Supplements Raw ---\n{supplements_raw}\n--- End Supplements Raw ---\n\n")
            st.code("".join(combined_debug_output), language='json')


if __name__ == "__main__":
    main()
