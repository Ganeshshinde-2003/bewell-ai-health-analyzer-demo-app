import streamlit as st
import google.generativeai as genai
import os
import fitz  # PyMuPDF - for PDF
import io
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
                # You might experiment with "html", "xml", "json", "textwords", etc. for different structures
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
                    # Adding sheet name might help the model understand context
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
         # Optionally, you could set text to a marker here too if you want the model to know extraction failed silently
         # text = "[No readable text extracted]" # Uncomment this line if you want the model to know extraction failed

    return text

# --- API Key Handling ---
# It's recommended to use st.secrets for secure deployment on Streamlit Cloud
# For local testing, use environment variables or paste it
api_key = "AIzaSyArQ9zeya1SO-IwsMappkLStXYT0W7WXfk" # Check .env first
if not api_key:
    api_key = "AIzaSyArQ9zeya1SO-IwsMappkLStXYT0W7WXfk" # Check Streamlit secrets
    if not api_key:
        st.warning("Please add your Google/Gemini API key to your environment variables (e.g., in a .env file as GOOGLE_API_KEY), Streamlit secrets, or paste it below.")
        # Fallback to text input in the app if key is not found elsewhere
        api_key = st.text_input("Or paste your Google/Gemini API Key here:", type="password")


# --- Gemini Model Configuration ---
# Using 1.5-pro is strongly recommended for its larger context window and
# better ability to follow complex, multi-part instructions and simplify concepts.
model_name = "gemini-2.0-flash" # Using 1.5-pro for better performance with text extraction output

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
# The prompt template remains the same, as we are only changing how the input is displayed
# NOT changing the instructions to the AI.
BEWELL_MASTER_PROMPT_TEMPLATE = """
You are Bewell's AI assistant and holistic women's health expert. Your primary goal is to provide a health analysis that is not only accurate but also **easy for someone with no medical background to understand**. Explain all concepts clearly, avoid jargon where possible, and explain any necessary medical terms simply. Your task is to analyze the provided user data (lab report text and health assessment) and generate a comprehensive, multi-part health analysis structured into THREE distinct sections:

1.  **Lab Analysis**
2.  **Four Pillars Analysis**
3.  **Personalized Supplement Recommendations**

Follow the specific instructions for the content and formatting of each section as described below. Ensure the tone is professional, clear, empathetic, supportive, and focused on women's unique health needs, simplifying medical information throughout.

---

**Detailed Instructions for Section 1: Lab Analysis**

Analyze the provided lab report text. For each biomarker listed that you can identify:
- Clearly indicate the biomarker status by labeling as:
    - **Optimal (Green)** if the value is clearly within the healthy reference range.
    - **Keep in Mind (Yellow)** if the value is technically within the normal range but close enough to the limits that it requires monitoring or consideration in the context of symptoms.
    - **Attention Needed (Orange-Red)** if the value is outside the healthy range and indicates potential health concerns.
- For each biomarker, provide:
    - **Result and Range:** Clearly state the user's result alongside the optimal reference range you used for assessment.
    - **Why It Matters:** Provide a **detailed, thorough, and easy-to-understand** explanation. **Assume the user has no medical background.** Explain the biomarker's *function in the body* (what it does), why healthy levels are important *specifically for women's health*, and the potential *practical implications* (how it might affect the user's body, health, energy levels, or symptoms) if levels are outside or borderline the range. Break down any medical terms simply.
Additionally, at the beginning of this 'Lab Analysis' section:
- Provide a brief but informative **Overall Health Summary** that synthesizes key insights from both the lab report and health assessment. **Explain the overall picture in simple terms**, highlighting any critical areas of concern or notable strengths.
At the conclusion of this 'Lab Analysis' section:
- Provide a concise yet actionable **Health Recommendation Summary**, offering practical steps or suggestions tailored specifically to this user's results. **Ensure these steps are easy to understand and actionable** for someone without medical knowledge, focusing on how they can help optimize her health, prevent potential issues, and manage any existing symptoms or conditions effectively.

---

**Detailed Instructions for Section 2: Four Pillars Analysis**

Based on the user's health assessment and lab report, generate a detailed and personalized analysis structured under Bewell's Four Pillars framework (Eat Well, Sleep Well, Move Well, Recover Well).
- **Introduction:** Briefly summarize the user's current overall health status based on the provided data. **Explain this summary in simple language.**
- **Four Pillars Analysis:** For each of the pillars (Eat Well, Sleep Well, Move Well, Recover Well):
    - **Why This Pillar is Important:** Explain specifically why this pillar matters *for the user's situation*, taking into account their personal health conditions, concerns, symptoms, and relevant lab results. **Explain this connection clearly and simply.**
    - **Bewell's Recommendations:** Provide personalized, actionable recommendations tailored explicitly to the user's health status and lifestyle. **Ensure these recommendations are clear, achievable, and easy to understand and implement.**
    - **Root Cause Correlation:** Explain clearly how each recommendation connects directly to the root causes identified or suggested in the user's lab results and health assessment (e.g., addressing potential low-grade inflammation, supporting immune function). **Make this connection simple and logical for a non-medical audience.**
- **Science-Based Explanation:** For *each specific recommendation* provided under the pillars, include a brief explanation of why this recommendation is scientifically valid and relevant to the user's specific biomarkers or reported symptoms. **Crucially, explain this in simple terms, focusing on the practical impact for the user, without using complex medical jargon.** Ground the explanation in fundamental scientific principles relevant to the user's situation.
- **Conclusion:** Summarize the key recommendations and anticipated benefits. Provide an encouraging note. **Ensure the language is simple and motivating.**

---

**Detailed Instructions for Section 3: Personalized Supplement Recommendations**

Act as a holistic clinician. Carefully review the blood test results (biomarkers) and detailed health assessment responses. Generate personalized supplement recommendations specifically tailored to the user's unique needs.
Your recommendations must include:
- **Supplement Name**
- **Personalized Rationale (Why It Matters):** Explain *why* you selected each supplement based on the user's lab results (mentioning relevant biomarkers and their status - Optimal, Keep in Mind, Attention Needed) and health assessment findings (e.g., energy levels, mood, menstrual symptoms, digestive health, recurring symptoms). **Clearly connect it to how it relates to their reported symptoms or potentially impacts the function indicated by the lab results in a way a layperson can grasp. Use simple language.**
- **Expected Outcomes:** Outline specific, personalized health benefits and improvements the user can realistically expect to feel or notice from taking the supplement. **Focus on tangible, understandable outcomes.**
- **Recommended Dosage & Timing:** Provide clear, evidence-based dosage instructions and optimal timing (e.g., morning, evening, with meals, cycle phases).
- **Situational/Cyclical Considerations:** Identify if any recommendation is particularly beneficial during specific menstrual cycle phases or certain life situations (stressful periods, intense physical activity, pregnancy, etc.). Explain *why* this is the case simply.
Clearly structure each recommendation with the headings listed above for each supplement. End with a concise, reassuring summary to encourage adherence and clear next steps for the user. **Maintain simple, encouraging language.** Do not recommend supplements for conditions clearly requiring medical diagnosis and treatment unless as *adjunct* support discussed with a doctor.

---

**User Data:**

Here is the user's Lab Report text:
{lab_report_text}

Here is the user's Health Assessment text:
{health_assessment_text}

---

**Instructions for Output:**

Generate ONE complete response containing the THREE distinct sections as detailed above ('Lab Analysis', 'Four Pillars Analysis', 'Personalized Supplement Recommendations'). Ensure each section is clearly labeled and follows the formatting rules specified for its content. Use markdown for formatting (bolding, lists, headers).
"""

# --- Analysis Button and Logic ---
if st.button("Generate Health Analysis"):
    if not api_key:
        st.error("Please provide your Google/Gemini API key.")
    else:
        # --- Get Input Data ---
        # Store the raw extracted/pasted text for later display
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
                st.error("Please check your API key, ensure the model name ('gemini-2.0-flash' recommended) is correct, and verify your internet connection.")

# --- Footer ---
st.markdown("---")
st.markdown("Built for Outlier Model Playground research. This analysis is AI-generated and for informational purposes only. Always consult a medical professional for diagnosis and treatment.")