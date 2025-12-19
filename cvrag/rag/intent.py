def detect_query_intent(query: str) -> str:
    query_lower = query.lower()

    if any(
        phrase in query_lower
        for phrase in ["for the role", "position at", "applying to", "job at", "role of"]
    ):
        return "role_targeting"

    if any(
        phrase in query_lower
        for phrase in ["cover letter", "application letter", "letter for"]
    ):
        return "cover_letter"

    if any(
        phrase in query_lower
        for phrase in [
            "give me an example",
            "tell me about a time",
            "describe a situation",
            "star example",
            "star examples",
            "star format",
            "behavioral",
        ]
    ):
        return "star_examples"

    if any(
        phrase in query_lower
        for phrase in [
            "how did you implement",
            "technical details",
            "architecture",
            "deep dive",
            "explain the technical",
        ]
    ):
        return "technical_deepdive"

    if any(
        phrase in query_lower
        for phrase in ["interview", "prepare for", "questions about", "what would you say"]
    ):
        return "interview_prep"

    return "general_qa"


def enhance_query(query: str, intent: str | None = None) -> str:
    if not intent:
        intent = detect_query_intent(query)

    query_lower = query.strip().lower()

    if intent == "role_targeting":
        return f"relevant experience skills achievements for role: {query}"
    if intent == "cover_letter":
        return f"achievements leadership impact quantified results: {query}"
    if intent == "star_examples":
        return f"specific examples achievements leadership problem solving: {query}"
    if intent == "technical_deepdive":
        return f"technical implementation architecture details: {query}"
    if intent == "interview_prep":
        return f"experience skills achievements examples: {query}"

    if any(word in query_lower for word in ["aws", "cloud", "kubernetes", "docker"]):
        return f"cloud infrastructure devops: {query}"
    if any(word in query_lower for word in ["experience", "worked", "job", "role"]):
        return f"professional experience work history: {query}"
    if any(word in query_lower for word in ["skill", "technology", "tech", "programming"]):
        return f"technical skills technologies: {query}"
    if any(word in query_lower for word in ["project", "built", "created", "developed"]):
        return f"projects development work: {query}"
    if any(word in query_lower for word in ["education", "degree", "university", "college"]):
        return f"education academic background: {query}"

    return query
