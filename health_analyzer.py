import google.generativeai as genai
import os
import fitz  # PyMuPDF for PDF
import io
import pandas as pd  # for Excel and CSV
from docx import Document  # for .docx files
from pptx import Presentation  # for .ppt and .pptx files
import argparse
import sys
import logging

# --- Set up logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Text Extraction Function ---
def extract_text_from_file(file_path):
    logger.info(f"Attempting to extract text from file: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return f"[Error: File {file_path} does not exist]"

    file_extension = os.path.splitext(file_path)[1].lower()
    text = ""

    try:
        with open(file_path, "rb") as file:
            file_bytes = file.read()

        if file_extension == ".pdf":
            logger.info("Processing PDF file")
            pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                text += page.get_text("text") + "\n"
            pdf_document.close()

        elif file_extension == ".docx":
            logger.info("Processing DOCX file")
            doc = Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                text += para.text + "\n"

        elif file_extension in [".ppt", ".pptx"]:
            logger.info("Processing PowerPoint file")
            prs = Presentation(io.BytesIO(file_bytes))
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"

        elif file_extension in [".xlsx", ".xls"]:
            logger.info("Processing Excel file")
            excel_data = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
            for sheet_name, df in excel_data.items():
                text += f"--- Sheet: {sheet_name} ---\n"
                text += df.to_csv(index=False) + "\n\n"

        elif file_extension in [".txt", ".csv"]:
            logger.info("Processing TXT or CSV file")
            text = file_bytes.decode('utf-8', errors='ignore')

        else:
            logger.warning(f"Unsupported file type: {file_extension}. Attempting to read as text")
            try:
                text = file_bytes.decode('utf-8', errors='ignore')
            except Exception as e:
                logger.error(f"Failed to process {file_path} as text: {e}")
                return f"[Error: Unsupported file type {file_extension}. Could not extract text: {e}]"

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return f"[Error processing {file_path}: {e}]"

    if not text.strip() and file_extension not in [".txt", ".csv"]:
        logger.warning(f"No readable text extracted from {file_path}. File might be scanned, empty, or have complex formatting.")
        return f"[Warning: No readable text extracted from {file_path}. File might be scanned, empty, or have complex formatting.]"

    logger.info(f"Successfully extracted text from {file_path}")
    return text

# --- Master Prompt Template ---
BEWELL_MASTER_PROMPT_TEMPLATE = """
Role: Bewell AI Assistant.
Persona: Holistic women's health expert, precision medicine, functional doctor, Women focused care.
Tone: Approachable, professional, empathetic, supportive, clear, accessible. Avoid casual language. Do not use the pronoun "I". Avoid personifying the analysis tone and manner. Focus on empowering the user with clear, accurate, and personalized insights from Bewell.

Input: User's health assessment text and lab report text. Analyze *only* the information provided in these texts. Base the analysis, recommendations, and rationale *solely* on the specific biomarkers and symptoms reported by the user in the provided data.

Output Structure: Generate ONE complete response containing THREE distinct sections as follows, using Markdown for clear formatting (headers, bolding, lists):

# Section 1: Lab Analysis

## Overall Health Summary:
- Synthesize user's current health status *based solely on provided lab results and health assessment*.
- Highlight significant areas of concern or strengths in simple, personalized language.

## Detailed Biomarker Analysis:
- For *each* biomarker listed in the provided lab report:
    - Clearly label the biomarker status as: **Optimal (Green)**, **Keep in Mind (Yellow)**, or **Attention Needed (Orange-Red)** based on provided result and range.
    - Provide Result and Range: Clearly state user's result and the provided reference range.
    - Provide Cycle Impact: Detail any known fluctuations or impacts specific menstrual cycle phases have on the biomarker relevant to women's health. State 'Not typically impacted by cycle' or 'Cycle impact not well-established' if applicable.
    - Provide Why It Matters: Explain the biomarker's primary function, its importance specifically to women's health, and the potential practical implications if *this user's specific level* is abnormal or borderline. Use clear, science-backed explanations without medical jargon, assuming the user has no medical background.

## Crucial Biomarkers to Measure:
- Provide a list of essential biomarkers women should measure regularly, categorized clearly and simply.
- Briefly explain the importance of *each* biomarker in accessible language.

## Health Recommendation Summary:
- Provide clear, concise, *actionable* steps tailored *specifically* to the user's *provided* lab results and health assessment findings. Presented in accessible language.

# Section 2: Four Pillars Analysis (Eat Well, Sleep Well, Move Well, Recover Well)

## Introduction:
- Briefly summarize the user's overall health *based on findings from Section 1* in clear, accessible language.

## Four Pillars Analysis:
- For *each* pillar (Eat Well, Sleep Well, Move Well, Recover Well):
    - ### Why This Pillar Matters:
        Explain how this pillar is *specifically relevant to this user's unique health assessment details and lab findings*. Use reachable language.
    - ### Bewell's Personalized Recommendations:
        Provide *actionable, personalized* advice tailored to *this user's specific status and lifestyle*. Recommendations must be achievable.
    - ### Root Cause Correlation:
        Clearly explain in accessible language how *each recommendation* connects directly to the *root causes or contributing factors* identified *in this user's lab results and health assessment*.
    - ### Science-Based Explanation:
        For *each recommendation*, provide a clear, simple scientific basis focused on practical user benefits, without medical jargon.
    - ### Additional Guidance for Pillars:
        - **Eat Well:** Include lists of recommended top foods to consume regularly and foods to approach cautiously based on general women's health principles.
        - **Move Well:** Provide a list of the top 5 recommended workouts and habits to avoid, focusing on general health benefits.
        - **Recover Well:** Include top 5 recommended recovery tips and habits to avoid, focusing on stress management and rest.

# Section 3: Personalized Supplement Recommendations

- Review the provided blood test biomarkers and detailed health assessment responses.
- Generate personalized supplement recommendations tailored *specifically to the user's unique needs based *only* on the provided data*.
- For *each* recommended supplement:
    - **Supplement Name:**
    - **Personalized Rationale:** Clearly explain *why* recommended based on user's biomarkers and reported health assessment symptoms. Explain how it addresses *this user's specific issues reported*. Use simple, accessible language.
    - **Expected Outcomes:** Describe tangible, *personalized* benefits the user can realistically notice.
    - **Recommended Dosage & Timing:** Clearly outline precise dosage instructions and optimal timing, based on general guidelines or evidence where applicable.
    - **Situational/Cyclical Considerations:** Clearly identify if beneficial during specific menstrual cycle phases or particular life circumstances *if applicable and relevant to the supplement and user's provided profile*. Explain *why* this is the case simply.

- **Conclude This Section:**
    - Provide concise, reassuring guidance to encourage adherence.
    - Clearly state that supplements are adjunctive and medical consultation is necessary.

Ensure the entire analysis maintains clarity, professional tone, personalization, and accessibility, empowering users to actively manage their health effectively and confidently.

---

**User Data:**

Here is the user's Lab Report text:
{lab_report_text}

Here is the user's Health Assessment text:
{health_assessment_text}

---

**Instructions for Output:**

Generate ONE complete response containing the THREE distinct sections as detailed above. Ensure each section is clearly labeled with the specified Markdown headers and follows all formatting rules specified for its content. Use markdown for formatting within each section.
"""

def main():
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Bewell AI Health Analyzer: Generate health analysis from health assessment (required) and lab report (optional) files.")
    parser.add_argument("--lab-report", type=str, help="Path to lab report file (any format) [optional]")
    parser.add_argument("--health-assessment", type=str, required=True, help="Path to health assessment file (any format) [required]")
    args = parser.parse_args()

    logger.info("Starting health analysis process")
    logger.info(f"Health assessment file: {args.health_assessment}")
    logger.info(f"Lab report file: {args.lab_report if args.lab_report else 'Not provided'}")

    # --- Extract Text from Files ---
    lab_report_text = extract_text_from_file(args.lab_report) if args.lab_report else ""
    health_assessment_text = extract_text_from_file(args.health_assessment)

    # --- Validate Health Assessment ---
    if not health_assessment_text.strip():
        logger.error(f"No valid text extracted from health assessment file: {args.health_assessment}")
        print(f"Error: No valid text extracted from health assessment file: {args.health_assessment}")
        sys.exit(1)

    # --- API Key and Model Configuration ---
    api_key = "AIzaSyArQ9zeya1SO-IwsMappkLStXYT0W7WXfk"  # Replace with your API key or use environment variable
    if not api_key:
        logger.error("Google/Gemini API key not found")
        print("Error: Google/Gemini API key not found. Set it in the script or as an environment variable (GOOGLE_API_KEY).")
        sys.exit(1)

    model_name = "gemini-2.0-flash"
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    # --- Configure Gemini API ---
    try:
        logger.info("Configuring Gemini API")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
        )

        # --- Format Prompt ---
        logger.info("Formatting prompt with user data")
        prompt = BEWELL_MASTER_PROMPT_TEMPLATE.format(
            lab_report_text=lab_report_text,
            health_assessment_text=health_assessment_text
        )

        print("Generating health analysis... This may take a few moments.")
        logger.info("Generating health analysis")

        # --- Generate Analysis ---
        response = model.generate_content(prompt)

        # --- Output Results ---
        print("\n=== Your Personalized Bewell Analysis ===\n")
        if response and response.text:
            print(response.text)
            logger.info("Analysis generated successfully")
        elif response and response.prompt_feedback:
            logger.error(f"Analysis request blocked. Reason: {response.prompt_feedback.block_reason}")
            print(f"Error: Analysis request was blocked. Reason: {response.prompt_feedback.block_reason}")
            if response.prompt_feedback.safety_ratings:
                print(f"Safety Ratings: {response.prompt_feedback.safety_ratings}")
            print("Please review your input files for sensitive content and try again.")
            sys.exit(1)
        else:
            logger.error("Failed to generate a valid response from the model")
            print("Error: Failed to generate a valid response from the model.")
            sys.exit(1)

        # --- Output Raw Extracted Text ---
        print("\n=== Raw Input Sent to AI ===\n")
        if lab_report_text.strip():
            print("Lab Report Text:")
            print(lab_report_text)
            print("\n")
        else:
            print("No Lab Report text provided.")

        print("Health Assessment Text:")
        print(health_assessment_text)
        logger.info("Raw input text output completed")

    except Exception as e:
        logger.error(f"Error during API interaction or processing: {e}")
        print(f"Error during API interaction or processing: {e}")
        sys.exit(1)

    print("\n=== Footer ===")
    print("This analysis is AI-generated for informational purposes only. Always consult a medical professional for diagnosis and treatment.")
    logger.info("Health analysis process completed")

if __name__ == "__main__":
    main()