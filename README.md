---
title: Enhanced CV RAG - Unified Ask
emoji: ⚡
colorFrom: gray
colorTo: purple
sdk: docker
pinned: false
short_description: Cv Rag
---

# Enhanced CV RAG System - Unified Ask Endpoint

An intelligent Retrieval-Augmented Generation (RAG) system that automatically detects query intent and provides specialized responses for CV/resume content.

## 🎯 One Endpoint, Multiple Capabilities

The `/ask` endpoint automatically detects what you need and responds accordingly:

### 📝 **General Q&A**
```
"What experience do I have with Kubernetes?"
"List my Python projects"
"What are my cloud certifications?"
```

### 🎯 **Role-Targeted Summaries**
```
"Create a summary for Senior DevOps Engineer at AWS"
"Summarize my qualifications for Full Stack Developer role"
"What makes me suitable for Cloud Architect position?"
```

### 💌 **Cover Letters**
```
"Write a cover letter for Software Engineer at Google"
"Create an application letter for DevOps role emphasizing automation"
"Draft a cover letter highlighting my cloud experience"
```

### ⭐ **STAR Examples**
```
"Give me STAR examples of leadership"
"Tell me about a time I solved a technical problem"
"Describe situations where I improved performance"
```

### 🔧 **Technical Deep-Dives**
```
"Explain the technical details of the Kubernetes migration"
"How did I implement the microservices architecture?"
"Describe the CI/CD pipeline I built"
```

### 🎤 **Interview Preparation**
```
"How should I answer questions about cloud experience?"
"Prepare me for DevOps interview questions"
"What examples should I use for behavioral interviews?"
```

## 🚀 Key Features

- **Intent Auto-Detection**: Automatically understands what type of response you need
- **Advanced Retrieval**: MMR algorithm ensures diverse, relevant context
- **Smart Context**: Multi-strategy retrieval based on query type
- **Professional Formatting**: Responses tailored to specific use cases
- **Technical Focus**: Optimized for software engineering and technical roles

## 📋 API Endpoints

- `POST /ask` - **Primary endpoint** - handles all query types with automatic intent detection
- `POST /set-cv` - Upload CV data (JSON format)
- `GET /` - API documentation and examples
- `POST /recruiter/role-fit` - Role-fit matcher against a job description
- `POST /recruiter/quick-summary` - 30-60s recruiter-ready summary
- `GET /recruiter/skills-matrix` - Structured skills matrix
- `POST /recruiter/project-deep-dives` - Project deep-dive cards
- `POST /recruiter/star-bank` - STAR examples tagged by competency
- `GET /recruiter/certifications` - Certification verification data
- `GET /recruiter/availability` - Availability and location preferences
- `POST /recruiter/export` - LinkedIn or ATS-friendly export text
- `GET /recruiter/export/pdf` - Download PDF CV when available
- `GET /recruiter/contact` - Contact and scheduling links
- `GET /recruiter/references` - References or endorsements

## 🔧 Technical Stack

- **Framework**: FastAPI with CORS support
- **Embeddings**: HuggingFace `all-mpnet-base-v2` (superior for technical content)
- **Vector Store**: FAISS with Maximum Marginal Relevance
- **LLM**: Groq Llama 3.1 8B Instant
- **Processing**: Enhanced chunking with semantic awareness

## 💡 Usage Examples

**Simple Query:**
```json
POST /ask
{
  "query": "What AWS experience do I have?"
}
```

**Role-Targeted:**
```json
POST /ask
{
  "query": "Create a summary for Cloud Engineer role at Netflix focusing on Kubernetes and automation"
}
```

**Cover Letter:**
```json
POST /ask
{
  "query": "Write a cover letter for Senior Full Stack Developer emphasizing React and Node.js experience"
}
```

**Role Fit Matcher:**
```json
POST /recruiter/role-fit
{
  "job_description": "Senior DevOps Engineer with AWS, Terraform, and Kubernetes experience",
  "role": "Senior DevOps Engineer",
  "company": "ExampleCo"
}
```

**Recruiter Summary:**
```json
POST /recruiter/quick-summary
{
  "role_level": "Senior",
  "focus": "cloud infrastructure and automation"
}
```

**Skills Matrix:**
```json
GET /recruiter/skills-matrix
```

## 📊 Response Format

```json
{
  "answer": "Detailed response based on detected intent",
  "intent": "role_targeting|cover_letter|star_examples|technical_deepdive|interview_prep|general_qa",
  "context_used": 12,
  "query_enhanced": true
}
```

The system intelligently adapts its response style, context retrieval, and formatting based on what you're asking for - all through a single, natural language interface.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
