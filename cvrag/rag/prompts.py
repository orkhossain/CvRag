from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM = """You are a helpful assistant. Answer the user's question based ONLY on the provided context and any tool outputs.

INSTRUCTIONS:
- Use only the provided context and tool outputs; do not add outside knowledge
- If the context does not contain the answer, say "I don't have that information in the context"
- Keep the response concise and directly relevant to the question
- If a question is ambiguous, ask a brief clarification question
- You may translate the context as needed, but do not introduce new information
- Respond in {language}"""

PROMPTS = {
    "general_qa": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM
                + "\n\nRELEVANT CONTEXT:\n{context}\n\nProvide a direct, informative answer using only the context provided.",
            ),
            MessagesPlaceholder("chat_history"),
            ("user", "Question: {question}"),
        ]
    ),
    "role_targeting": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM
                + "\n\nRELEVANT CONTEXT:\n{context}\n\nCreate a targeted response highlighting the most relevant qualifications for the specific role mentioned. Focus on alignment between experience and role requirements.",
            ),
            MessagesPlaceholder("chat_history"),
            (
                "user",
                "Query: {question}\n\nProvide a role-targeted summary (120-150 words) with bullet points emphasizing relevant experience, skills, and quantified achievements.",
            ),
        ]
    ),
    "cover_letter": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM
                + "\n\nRELEVANT CONTEXT:\n{context}\n\nWrite a compelling cover letter using specific examples and quantifiable achievements from the context.",
            ),
            MessagesPlaceholder("chat_history"),
            (
                "user",
                "Request: {question}\n\nCreate a professional cover letter (~180 words) that:\n1. Opens with strong alignment\n2. Highlights 2-3 relevant achievements with metrics\n3. Shows enthusiasm and understanding\n4. Closes with next steps",
            ),
        ]
    ),
    "star_examples": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM
                + "\n\nRELEVANT CONTEXT:\n{context}\n\nProvide STAR format examples (Situation, Task, Action, Result) with specific details and quantified outcomes.",
            ),
            MessagesPlaceholder("chat_history"),
            (
                "user",
                "Request: {question}\n\nGenerate compelling STAR examples that demonstrate:\n- Technical leadership and problem-solving\n- Specific actions taken\n- Quantified business impact\n\nFormat: **Situation:** [context] **Task:** [challenge] **Action:** [steps] **Result:** [outcome]",
            ),
        ]
    ),
    "technical_deepdive": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM
                + "\n\nRELEVANT CONTEXT:\n{context}\n\nProvide detailed technical explanations focusing on implementation details, architecture decisions, and technologies used.",
            ),
            MessagesPlaceholder("chat_history"),
            (
                "user",
                "Technical Question: {question}\n\nProvide a comprehensive technical explanation including:\n- Specific technologies and tools used\n- Architecture and implementation approach\n- Challenges faced and solutions\n- Technical outcomes and metrics",
            ),
        ]
    ),
    "interview_prep": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM
                + "\n\nRELEVANT CONTEXT:\n{context}\n\nProvide comprehensive interview preparation focusing on relevant experience, specific examples, and potential follow-up questions.",
            ),
            MessagesPlaceholder("chat_history"),
            (
                "user",
                "Interview Prep: {question}\n\nProvide interview-ready responses including:\n- Key talking points with specific examples\n- Quantified achievements and impact\n- Technical details where relevant\n- Potential follow-up questions to prepare for",
            ),
        ]
    ),
    "role_fit_matcher": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM
                + "\n\nRELEVANT CONTEXT:\n{context}\n\nCompare the CV context to the job description and return JSON only.",
            ),
            (
                "user",
                "Job description:\n{job_description}\n\nRole: {role}\nCompany: {company}\n\nReturn JSON with keys: summary, top_matches, gaps, suggested_questions. Use arrays for list fields.",
            ),
        ]
    ),
    "quick_summary": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM
                + "\n\nRELEVANT CONTEXT:\n{context}\n\nProvide a concise recruiter-ready summary.",
            ),
            (
                "user",
                "Role level: {role_level}\nFocus: {focus}\n\nWrite a 30-60 second summary (80-120 words) using only the context.",
            ),
        ]
    ),
    "project_deep_dives": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM
                + "\n\nRELEVANT CONTEXT:\n{context}\n\nCreate project deep-dive cards and return JSON only.",
            ),
            (
                "user",
                "Projects data:\n{projects}\n\nReturn an array of cards with keys: name, overview, architecture, stack, impact, challenges.",
            ),
        ]
    ),
    "star_bank": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM
                + "\n\nRELEVANT CONTEXT:\n{context}\n\nGenerate STAR examples tagged by competency and return JSON only.",
            ),
            (
                "user",
                "Competencies: {competencies}\nCount: {count}\n\nReturn an array with keys: competency, situation, task, action, result, metrics.",
            ),
        ]
    ),
    "export_linkedin": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM
                + "\n\nRELEVANT CONTEXT:\n{context}\n\nWrite a LinkedIn About section using only the context.",
            ),
            (
                "user",
                "Focus: {focus}\n\nWrite 150-220 words in a friendly, professional tone.",
            ),
        ]
    ),
    "export_ats": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM
                + "\n\nRELEVANT CONTEXT:\n{context}\n\nCreate ATS-friendly resume text using only the context.",
            ),
            (
                "user",
                "Format: plain text with clear section headers: Summary, Skills, Experience, Projects, Education, Certifications. Avoid tables and columns.",
            ),
        ]
    ),
    "recruiter_pitch": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM
                + "\n\nRELEVANT CONTEXT:\n{context}\n\nWrite a concise recruiter-facing pitch that explains why the candidate is a strong option. Use only the context.",
            ),
            (
                "user",
                "Question: {question}\n\nRespond in 4-6 sentences. Mention role-relevant strengths, impact, and any soft skills explicitly present in the context.",
            ),
        ]
    ),
}
