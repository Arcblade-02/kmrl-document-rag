import streamlit as st
from google import genai
from google.genai.errors import APIError

# --- RAG CONTEXT (The KMRL Knowledge Base - Hardcoded for this script) ---
# NOTE: In a real RAG system, this context would come from a dynamic vector search.
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

# --- SYSTEM INSTRUCTION (The AI's Conversational Persona) ---
SYSTEM_INSTRUCTION = (
    "You are the **KMRL Document Intelligence Assistant**, a professional and helpful expert on all internal "
    "Kochi Metro Rail Limited policies, procedures, and contracts. Your tone must be friendly, authoritative, "
    "and conversational. "
    "Your function is to answer employee queries instantly and accurately by synthesizing "
    "information from the provided documents. **CRITICAL RULE**: You MUST cite the source "
    "Document ID (e.g., [DOCUMENT ID: POLICY-HR-32B]) for every piece of factual "
    "information you provide. If the information is not found in the documents, state it explicitly."
)

# --- CORE RAG FUNCTION (Now accepting and using full chat history) ---
def get_cited_answer(user_query: str, history: list):
    """
    Constructs the full prompt including context, history, and the new query, 
    and sends it to the Gemini API using st.secrets for the key.
    """
    try:
        # 1. Initialize Client and System Instruction
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        
        # 2. Construct the RAG Payload Prompt
        # This is the prompt that contains all the information the model needs:
        # Context (the documents) + Conversation History + Current Question.
        full_rag_payload = (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"--- KMRL DOCUMENTS CONTEXT ---\n{KMRL_DOCUMENTS_CONTEXT}\n\n"
            f"--- CONVERSATION HISTORY ---\n"
            # Format the history into a simple, readable string for the model
            + "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
            + f"\n\n--- CURRENT USER QUERY ---\n{user_query}\n\n"
            "Based ONLY on the CONTEXT and considering the HISTORY, please answer the CURRENT USER QUERY. "
            "You MUST cite the Document ID for every fact."
        )

        # 3. Generate Content
        # We send the entire RAG payload as a single prompt for controlled, stateful RAG.
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_rag_payload
        )
        
        return response.text
        
    except KeyError:
        return "🚨 **Configuration Error:** The `GEMINI_API_KEY` is missing from Streamlit Secrets. Please configure it to run the app."
    except APIError as e:
        return f"❌ An API error occurred: {e}. Please check your key and permissions."
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# ==============================================================================
# --- STREAMLIT INTERFACE CODE (Main App Logic) ---
# ==============================================================================

st.set_page_config(page_title="KMRL Document Intelligence Prototype", layout="centered")
st.title("🚇 KMRL Document Intelligence Prototype")
st.subheader("RAG System for Kochi Metro Rail Policies")

# --- INITIALIZATION ---

# Initialize chat history if it doesn't exist
# This is the key to conversational memory.
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- MAIN CHAT INTERFACE ---

# 1. Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Accept user input and start the RAG process
user_input = st.chat_input("Ask a question (e.g., What is the overtime rate?)")

if user_input:
    # 2a. Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2b. Generate AI response (Now using the history)
    with st.chat_message("assistant"):
        with st.spinner("Searching KMRL documents and synthesizing answer..."):
            # Pass the current input AND the full conversation history to the RAG function
            response_text = get_cited_answer(user_input, st.session_state.messages)
        
        st.markdown(response_text)
    
    # 2c. Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # Optional: Add an expander to show the context used
    with st.expander("🔍 Show Document Context Used"):
        st.code(KMRL_DOCUMENTS_CONTEXT, language='markdown')