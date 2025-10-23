import os
from google import genai
from google.genai.errors import APIError
import time # Adding a small delay for presentation flow

# --- 1. CONTEXT INJECTION (Simulated KMRL Knowledge Base) ---
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

# --- 2. SYSTEM INSTRUCTION (The AI's Persona and Rules) ---
SYSTEM_INSTRUCTION = (
    "You are the KMRL Document Intelligence System. Your primary role is to act as an expert "
    "on all internal Kochi Metro Rail Limited policies, procedures, and contracts. "
    "Your function is to answer employee queries instantly and accurately by synthesizing "
    "information from the provided documents (the KMRL_DOCUMENTS_CONTEXT). "
    "CRITICAL RULE: You MUST cite the source Document ID for every piece of factual "
    "information you provide. The tone must be professional, authoritative, and helpful. "
    "If the information is not found in the documents provided, you must explicitly state: "
    "'The required information is not available in the current document set.'"
)


# --- 3. CORE RAG FUNCTION ---
def get_cited_answer(user_query: str):
    """
    Combines the context and the query, then calls the Gemini API.
    """
    print(f"\nQUERY: {user_query}")
    time.sleep(1) # Pause for 1 second for presentation effect
    try:
        client = genai.Client()
    except ValueError:
        print("\n[ERROR] GEMINI_API_KEY environment variable not set. Please set it and retry.")
        return

    # Combine all parts into the final prompt for the model
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
        print("\n--- KMRL SYSTEM RESPONSE (CITED) ---")
        print(response.text)
        print("------------------------------------\n")
    except APIError as e:
        print(f"An API error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- 4. EXECUTION OF TESTS ---
if __name__ == '__main__':
    print("--- KMRL DOCUMENT INTELLIGENCE SYSTEM STARTUP ---")

    # TEST 1: Cross-Document Synthesis (High Value Demo)
    TEST_QUERY_1 = (
        "I have a technician who worked 42 hours this week, including a 4-hour track inspection on a Saturday. "
        "Based on the policies, how much of that work is paid at the overtime rate, and what are the document "
        "IDs that define the overtime rate and the required track team size?"
    )
    get_cited_answer(TEST_QUERY_1)
    
    # TEST 2: Specific Exception Handling
    TEST_QUERY_2 = (
        "What is the primary vendor for signaling equipment and what special requirements are in place "
        "if I need to use the secondary vendor, including the required form ID?"
    )
    get_cited_answer(TEST_QUERY_2)
    
    # TEST 3: Negative Test (Handling unavailability)
    TEST_QUERY_3 = (
        "What is the policy for booking emergency travel tickets for maintenance staff?"
    )
    get_cited_answer(TEST_QUERY_3)