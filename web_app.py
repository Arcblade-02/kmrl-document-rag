import streamlit as st
import os
from google import genai
from google.genai.errors import APIError

# Read the key from secrets.toml (Streamlit's recommended way)
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# --- RAG CONTEXT (The KMRL Knowledge Base) ---
KMRL_DOCUMENTS_CONTEXT = """
[KMRL_DOCUMENTS_CONTEXT_START]

**DOCUMENT ID: SOP-MAINT-401 (Track Maintenance Procedure)**
- Track inspection frequency: Daily for all primary running lines (Line 1).
- Inspection frequency for sidings, yards, and non-primary tracks is Weekly.
- Team requirement: All track inspections must be conducted by a minimum of two (2) certified technicians.
- Mandatory tool: The required and standard tool-kit is TK-45A.
- Safety Protocol: Technicians must notify the Operations Control Centre (OCC) via digital form (Form OCC-A) before any personnel access the track area.

**DOCUMENT ID: POLICY-HR-32B (Human Resources Policy Manual)**
- Sick Leave: All full-time employees are entitled to 15 days of sick leave per financial year, which accrues annually.
- Overtime Rate: Overtime is compensated at 1.5 times (1.5x) the base hourly rate for any hours worked in excess of 40 hours in a standard work week.
- Weekend Pay: All work performed on Saturday or Sunday is considered overtime, regardless of the weekly hour count.

**DOCUMENT ID: PROC-SIGNAL-005 (Signaling Procurement Contract)**
- Primary Signaling Vendor: Siemens (Contract ID SC-1002).
- Secondary Vendor: Alstom.
- Secondary Vendor Use: Utilization of the secondary vendor (Alstom) requires management approval documented on Requisition Form R-3, including a written justification memo.

[KMRL_DOCUMENTS_CONTEXT_END]
"""

# --- SYSTEM INSTRUCTION (The AI's Persona) ---
SYSTEM_INSTRUCTION = (
    "You are the KMRL Document Intelligence System. Your primary role is to act as an expert "
    "on all internal Kochi Metro Rail Limited policies, procedures, and contracts. "
    "Your function is to answer employee queries instantly and accurately by synthesizing "
    "information from the provided documents. "
    "CRITICAL RULE: You MUST cite the source Document ID for every piece of factual "
    "information you provide. The tone must be professional, authoritative, and helpful. "
    "If the information is not found in the documents, state it explicitly."
)

# --- CORE RAG FUNCTION (Updated for Streamlit Secrets) ---
def get_cited_answer(user_query: str):
    """
    Sends the user query and the full context to the Gemini API, using Streamlit secrets for the key.
    """
    try:
        # Use st.secrets to securely retrieve the API key
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except KeyError:
        return "[ERROR] API Key is missing from Streamlit Secrets configuration."
    except Exception as e:
        # Catch other API initialization errors
        return f"[ERROR] Could not initialize Gemini Client: {e}"

    # Construct the full prompt for the model
    full_prompt = (
        f"KMRL Documents Context:\n{KMRL_DOCUMENTS_CONTEXT}\n\n"
        f"User Query:\n{user_query}\n\n"
        "Please provide a comprehensive and fully cited answer based ONLY on the context above."
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION
            )
        )
        return response.text
    except APIError as e:
        return f"An API error occurred: 503 UNAVAILABLE. Please try again later. Details: {e}"
    except Exception as e:
        return f"An unexpected error occurred during content generation: {e}"


# --- STREAMLIT INTERFACE CODE (Main App Logic) ---

st.set_page_config(page_title="KMRL Document Intelligence Prototype", layout="centered")
st.title("🚇 KMRL Document Intelligence Prototype")
st.subheader("RAG System for Kochi Metro Rail Policies")

# Check for API Key setup in Streamlit Secrets
if "GEMINI_API_KEY" not in st.secrets:
    st.error("🚨 **Configuration Error:** The `GEMINI_API_KEY` is missing from Streamlit Secrets. Please configure it to run the app.")
else:
    st.info("System Ready. Ask a question below to see cross-document synthesis.")
    
    # --- Input and Conversation Logic ---
    user_input = st.chat_input("Ask a question (e.g., How many sick leave days do I have?)")

    if user_input:
        # 1. Display user query
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # 2. Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Searching KMRL documents and synthesizing answer..."):
                response_text = get_cited_answer(user_input)
                
            # 3. Display cited response
            st.markdown(response_text)
        
        # 4. Show context used (Optional but great for demo)
        with st.expander("🔍 Show Document Context Used"):
            st.code(KMRL_DOCUMENTS_CONTEXT, language='markdown')