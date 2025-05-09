import streamlit as st
import google.generativeai as genai
import os
import fitz  # PyMuPDF - for PDF
import io  # To handle file bytes\
import pandas as pd # for Excel and CSV
from docx import Document # for .docx files

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
    Returns extracted text as a string.
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
                # Use BytesIO for reading from bytes
                excel_data = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
                for sheet_name, df in excel_data.items():
                    text += f"--- Sheet: {sheet_name} ---\n"
                    # Convert DataFrame to a simple text representation (like CSV)
                    text += df.to_csv(index=False) + "\n\n"
            except Exception as excel_e:
                 # Catch pandas specific errors or missing dependencies (xlrd, openpyxl)
                 st.warning(f"Could not read Excel file {uploaded_file.name} with pandas. Ensure 'openpyxl' (for .xlsx) or 'xlrd' (for .xls) is installed. Error: {excel_e}")
                 text = f"[Could not automatically process Excel file {uploaded_file.name} - Please try pasting the text or saving as PDF/TXT.]" # Indicate failure

        elif file_extension in [".txt", ".csv"]:
            # Text or CSV extraction (simple read)
            # Read bytes and decode as UTF-8
            text = file_bytes.decode('utf-8')

        else:
            # Handle unsupported file types explicitly
            st.warning(f"Unsupported file type uploaded: {file_extension}")
            return "[Unsupported File Type Uploaded]" # Return a marker for unsupported type

    except Exception as e:
        # Catch any other errors during processing
        st.error(f"An error occurred while processing {uploaded_file.name}: {e}")
        return "[Error Processing File]" # Return a marker for processing error

    # Basic check if text was extracted (ignore for txt/csv as they might be empty)
    if not text.strip() and file_extension not in [".txt", ".csv"]:
         st.warning(f"No readable text extracted from {uploaded_file.name}. The file might be scanned, empty, or have complex formatting. Consider pasting the text manually.")

    return text

# --- API Key Handling ---
# It's recommended to use st.secrets for secure deployment on Streamlit Cloud
# For local testing, use environment variables (like in a .env file) or paste it
api_key = "AIzaSyArQ9zeya1SO-IwsMappkLStXYT0W7WXfk" # Check .env first
if not api_key:
    api_key = "AIzaSyArQ9zeya1SO-IwsMappkLStXYT0W7WXfk" # Check Streamlit secrets (recommended for Streamlit Cloud)
    if not api_key:
        st.warning("Please add your Google/Gemini API key to your environment variables (e.g., in a .env file as GOOGLE_API_KEY), Streamlit secrets, or paste it below.")
        # Fallback to text input in the app if key is not found elsewhere
        api_key = st.text_input("Or paste your Google/Gemini API Key here:", type="password")


# --- Gemini Model Configuration ---
# Using 1.5-pro is strongly recommended for its larger context window and
# better ability to follow complex, multi-part instructions and simplify concepts.
model_name = "gemini-2.0-flash" # Changed to 1.5-pro for better results with complex prompt

generation_config = {
    "temperature": 0.7, # Adjust for creativity vs. predictability
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192, # Generous output size for detailed reports
    "response_mime_type": "text/plain",
}

# --- App Title and Description ---
st.title("🌿 Bewell AI Health Analyzer")
st.markdown("""
Welcome to Bewell! Upload your lab report and health assessment files or paste the text to receive a personalized health analysis designed to be easy to understand, even without a medical background. We'll analyze your data based on the Four Pillars of wellness (Eat Well, Sleep Well, Move Well, Recover Well) and provide tailored supplement recommendations.

**Note:** This is an AI analysis tool. The insights provided are for informational purposes only and should not replace professional medical advice. Always consult with a qualified healthcare provider for any health concerns or before making decisions about your health or treatment. The quality of the analysis depends heavily on the clarity and readability of the uploaded files or pasted text.
""")

# --- Input Fields ---
st.header("Your Health Data")

st.subheader("Lab Report")
# Update the accepted file types
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
# Update the accepted file types
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


# --- The Main Prompt Construction ---
# UPDATED: Using the new detailed prompt template
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
- Briefly explain the importance of *each* biomarker in accessible language. (Note: Generate a general list of common and important biomarkers for women's health).

## Health Recommendation Summary:
- Provide clear, concise, *actionable* steps tailored *specifically* to the user's *provided* lab results and health assessment findings (symbols, history). Presented in accessible language.

# Section 2: Four Pillars Analysis (Eat Well, Sleep Well, Move Well, Recover Well)

## Introduction:
- Briefly summarize the user's overall health *based on findings from Section 1* in clear, accessible language.

## Four Pillars Analysis:
- For *each* pillar (Eat Well, Sleep Well, Move Well, Recover Well):
    - ### Why This Pillar Matters:
        Explain how this pillar is *specifically relevant to this user's unique health assessment details and lab findings*. Use accessible language.
    - ### Bewell's Personalized Recommendations:
        Provide *actionable, personalized* advice tailored to *this user's specific status and lifestyle* (e.g., addressing inconsistent eating, stress, sedentary job, specific symptoms reported). Recommendations must be achievable.
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
    - **Personalized Rationale:** Clearly explain *why* recommended based on user's biomarkers (mention status: Optimal, Keep in Mind, Attention Needed) and reported health assessment symptoms. Explain how it addresses *this user's specific issues reported*. Use simple, accessible language.
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

Generate ONE complete response containing the THREE distinct sections as detailed above ('# Section 1: Lab Analysis', '# Section 2: Four Pillars Analysis', '# Section 3: Personalized Supplement Recommendations'). Ensure each section is clearly labeled with the specified Markdown headers and follows all formatting rules specified for its content. Use markdown for formatting (bolding, lists, headers) within each section.
"""

# --- Analysis Button and Logic ---
if st.button("Generate Health Analysis"):
    if not api_key:
        st.error("Please provide your Google/Gemini API key.")
    else:
        # --- Get Input Data ---
        # Prioritize uploaded file text, fall back to text area if no file
        raw_lab_report_input = extract_text_from_file(lab_report_file) if lab_report_file else lab_report_text_area
        raw_health_assessment_input = extract_text_from_file(health_assessment_file) if health_assessment_file else health_assessment_text_area

        # Check if at least one source of input is available
        if not raw_lab_report_input and not raw_health_assessment_input:
            st.error("Please upload at least one document (Lab Report or Health Assessment) or paste text into the fields.")
        else:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                    # Add safety settings if desired
                    # safety_settings=...
                )

                # Format the prompt with the extracted/pasted user data
                prompt = BEWELL_MASTER_PROMPT_TEMPLATE.format(
                    lab_report_text=raw_lab_report_input,
                    health_assessment_text=raw_health_assessment_input
                )

                st.info("Extracting text from uploaded files (if any) and generating analysis. This may take a few moments...")

                # Generate the content using the Gemini API
                response = model.generate_content(prompt)

                # --- Display the Results ---
                st.header("Your Personalized Bewell Analysis")

                if response and response.text:
                    st.markdown(response.text)
                elif response and response.prompt_feedback:
                     # Handle cases where the prompt was blocked
                     st.warning(f"Analysis request was blocked. Reason: {response.prompt_feedback.block_reason}")
                     if response.prompt_feedback.safety_ratings:
                          st.warning(f"Safety Ratings: {response.prompt_feedback.safety_ratings}")
                     st.info("Please review your input for any sensitive or policy-violating content and try again.")
                     st.info("Note: Health information can sometimes be flagged. If you believe this was an error, try simplifying potentially sensitive descriptions in the health assessment.")

                else:
                     # Handle cases with no text response or unexpected format
                     st.error("Failed to generate a valid response from the model. The model might have returned an empty response or the response format was unexpected.")
                     # Optional: Print the full response object for debugging if needed
                     # st.write(response)

                # --- Display Raw Extracted Input ---
                st.markdown("---") # Separator
                st.subheader("Review Raw Input Sent to AI")
                st.info("This section shows the text extracted from your uploaded files or pasted into the text areas. This is what the AI processed.")

                if raw_lab_report_input:
                    with st.expander("Show Raw Lab Report Text"):
                        st.text(raw_lab_report_input) # Use st.text for raw text output
                else:
                    st.info("No Lab Report file uploaded or text pasted.")

                if raw_health_assessment_input:
                    with st.expander("Show Raw Health Assessment Text"):
                         st.text(raw_health_assessment_input) # Use st.text for raw text output
                else:
                    st.info("No Health Assessment file uploaded or text pasted.")


            except Exception as e:
                st.error(f"An error occurred during API interaction or response processing: {e}")
                st.error("Please check your API key, ensure the model name ('gemini-1.5-pro' recommended) is correct, and verify your internet connection.")

# --- Footer ---
st.markdown("---")
st.markdown("Built for Outlier Model Playground research. This analysis is AI-generated and for informational purposes only. Always consult a medical professional for diagnosis and treatment.")