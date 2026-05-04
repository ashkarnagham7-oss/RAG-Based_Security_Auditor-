from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from config import OPENAI_MODEL
from dotenv import load_dotenv
load_dotenv()

def build_context(retrieved_docs: List[Tuple[Document, float]]) -> str:
    chunks = []

    for idx, (doc, score) in enumerate(retrieved_docs, start=1):
        metadata = doc.metadata
        chunk = f"""
[Retrieved Example {idx}]
Similarity Score: {score:.4f}
Example ID: {metadata.get("example_id")}
CWE: {metadata.get("cwe_id")}
OWASP: {metadata.get("owasp_category")}
Language: {metadata.get("language")}
CVE: {metadata.get("cve_id")}
CVSS: {metadata.get("cvss")}
Incident: {metadata.get("incident_description")}

Code:
{doc.page_content}
"""
        chunks.append(chunk.strip())

    return "\n\n".join(chunks)


def explain_vulnerability(query_code: str, retrieved_docs: List[Tuple[Document, float]]) -> str:
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

    context = build_context(retrieved_docs)

    prompt = f"""
You are a security code auditor.

Task:
1. Analyze the input code.
2. you are going to determine if it is vulnerable or safe.
2. Use the retrieved SecureCode examples as supporting evidence(if the top score of similarity is less than 0.40, reason(safe or vulnerable and why) upon your information).
3. Identify likely vulnerability type(s).
4. Mention any likely CWE and CVE alignment if appropriate.
5. If code is vulnerable : 
- Explain why the code is dangerous.
- Suggest a secure remediation.
6. Be precise and concise.

Vulnerable input code:
{query_code}

Retrieved examples:
{context}

Return your answer in this structure:
- Summary
- The final result (vulnerable or safe code)
- If vulnerable, extract the evidence from retreived examples:
    1-If there is a detected CVE, show the cve_id, description, and the corresponding OWASP category
    2-Risk explanation
    3-Remediation
"""

    response = llm.invoke(prompt)
    return response.content