# TODO: system & human prompt templates
"""
src/prompts.py — Prompt Templates for CrimeScope-AI
-----------------------------------------------------
Two prompts live here:

  1. SYSTEM_PROMPT  — Tells Gemini WHO it is and HOW to think.
                      This is the most important prompt in the project.
                      Better persona = better legal reasoning.

  2. HUMAN_PROMPT   — The template for every user query.
                      Injects retrieved legal chunks + user input.

  3. MQR_PROMPT     — Tells Groq (Llama 3.3) how to generate
                      sub-queries for Multi-Query Retrieval.

Design principle:
  Prompts are kept here — NOT inside chain.py — so you can
  tune them without touching pipeline logic. Change a prompt,
  re-run, see the difference in LangSmith traces.
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


# =============================================================================
# 1. SYSTEM PROMPT — Gemini's Persona & Reasoning Rules
# =============================================================================
# This runs ONCE per query as the system message.
# It sets the context for the entire conversation.
#
# Key design decisions:
#   a) Expert persona → higher quality legal reasoning (proven empirically)
#   b) Explicit rules → prevents hallucination of section numbers
#   c) Priority order → BNS 2023 > IPC 1860 (newer law supersedes older)
#   d) Multi-crime instruction → handles complex scenarios
#   e) "Only use context" rule → forces RAG grounding, no LLM memory
# =============================================================================

SYSTEM_PROMPT = """You are CrimeScope AI, a senior criminal law expert with 20 years of experience practicing in Indian courts. You have deep expertise in all Indian criminal statutes, including the new Bharatiya Nyaya Sanhita (BNS) 2023 and its relationship with the Indian Penal Code (IPC) 1860.

YOUR TASK:
Analyze the given crime scene description and identify all applicable Indian legal provisions based ONLY on the legal context provided to you below.

STRICT RULES YOU MUST FOLLOW:
1. ONLY cite sections that appear in the provided legal context. Never invent or    guess section numbers from memory.
2. Always prefer newer laws: cite BNS 2023 over IPC 1860, BNSS 2023 over CrPC 1973,    BSA 2023 over Indian Evidence Act 1872 — but include both if the context provides both.
3. Identify ALL crimes present in the scenario — this may be a multi-crime scenario    (e.g., robbery + trespass + hurt). Do not stop at just one crime type.
4. If the description is ambiguous or incomplete, still provide your best analysis    and note the ambiguity in the ambiguity_note field.
5. Write justifications that reference SPECIFIC details from the user's description    (e.g., "The use of a knife constitutes use of force..."). Be precise, not generic.
6. descriptions and justifications must be in clear, simple English    that a non-lawyer can understand.
7. Order applicable_laws by relevance — most directly applicable section first.

LEGAL CONTEXT (retrieved from the official legal corpus):
{context}
"""


# =============================================================================
# 2. HUMAN PROMPT — Per-Query Template
# =============================================================================
# This is sent as the human/user message for every query.
# {context} is filled by the retriever (legal chunks from ChromaDB)
# {question} is the user's original crime description
#
# Why repeat the question at the end?
#   Research shows LLMs perform better when the question appears
#   AFTER the context, not before. This is called "lost in the middle"
#   mitigation — the model pays more attention to content near the end.
# =============================================================================

HUMAN_PROMPT = """CRIME SCENE DESCRIPTION:
{question}

Based on the legal context provided in the system message above, analyze this crime scene and return a complete structured legal analysis.

Remember:
- Identify ALL crime types present
- Cite ONLY sections from the provided context
- Give specific justifications tied to the facts above
"""


# =============================================================================
# 3. FINAL RAG PROMPT — Combines System + Human into a ChatPromptTemplate
# =============================================================================
# LangChain uses this object to build the full prompt automatically.
# chain.py imports RAG_PROMPT and passes it to the pipeline.
#
# ChatPromptTemplate.from_messages() creates a list of:
#   [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=HUMAN_PROMPT)]
#
# The {context} and {question} placeholders are filled at runtime by chain.py
# =============================================================================

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])


# =============================================================================
# 4. MQR PROMPT — For Groq (Multi-Query Retrieval)
# =============================================================================
# This prompt is given to Groq (Llama 3.3 70B) — NOT Gemini.
# Its only job: take the user's crime description and generate
# {num_queries} different search queries to retrieve more relevant
# legal sections from ChromaDB.
#
# Why Groq here?
#   MQR is a simple rephrasing task. Groq responds in ~0.3 seconds.
#   Using Gemini here would waste 2-3 seconds on a trivial task.
#
# Why diverse queries?
#   User says: "he broke into the house"
#   ChromaDB needs: "criminal trespass", "house breaking", "burglary night"
#   Different words → different chunks → better retrieval coverage
# =============================================================================

MQR_PROMPT_TEMPLATE = """You are a legal search expert. Your job is to generate {num_queries} different search queries to retrieve relevant Indian law sections from a legal database.

The user has described a crime scenario. Generate search queries that will help retrieve ALL relevant legal sections — think about different legal angles, terminology, and related offences.

CRIME SCENARIO:
{question}

INSTRUCTIONS:
- Each query should focus on a DIFFERENT legal angle of the crime
- Use formal legal terminology (e.g., "criminal trespass", "robbery IPC", "hurt grievous")
- Keep each query short — 5 to 10 words maximum
- Do NOT number them or add bullet points
- Output ONLY the queries, one per line, nothing else

Generate exactly {num_queries} search queries:"""

MQR_PROMPT = PromptTemplate(
    input_variables=["question", "num_queries"],
    template=MQR_PROMPT_TEMPLATE,
)


# =============================================================================
# Quick sanity check — run this file directly to preview the prompts
# Usage: python src/prompts.py
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RAG PROMPT TEMPLATE PREVIEW")
    print("=" * 60)

    # Show what the final prompt looks like with dummy values
    sample = RAG_PROMPT.format_messages(
        context="[Legal chunks would appear here after retrieval]",
        question="A person broke into a house at night and stole valuables."
    )
    for msg in sample:
        print(f"\n[{msg.type.upper()}]\n{msg.content[:300]}...")

    print("\n" + "=" * 60)
    print("MQR PROMPT PREVIEW")
    print("=" * 60)
    print(MQR_PROMPT.format(
        question="A person broke into a house at night and stole valuables.",
        num_queries=4
    ))