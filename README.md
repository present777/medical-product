# AI Clinical Communication Training System

> An AI-powered clinical communication training platform based on Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG), designed to simulate standardized patients (SPs) and provide evidence-based clinical interview assessment.

---

# Overview

Clinical communication is one of the most important skills for medical students and junior healthcare professionals. However, opportunities to practice with standardized patients (SPs) are often limited due to the high cost of teaching resources and instructor availability.

This project explores the application of Large Language Models (LLMs) in clinical communication training. By integrating Retrieval-Augmented Generation (RAG) with authoritative medical textbooks, the system provides realistic AI standardized patients together with evidence-based evaluation, enabling repeated practice and continuous improvement of clinical interviewing skills.

The platform supports a complete workflow including:

- AI Standardized Patient Generation
- Clinical Interview
- Diagnosis Submission
- Intelligent Evaluation
- Personalized Feedback Report

---

# Features

##  AI Standardized Patient Simulation

- Dynamically generates diverse clinical cases.
- Supports natural multi-turn conversations.
- Simulates realistic patient behaviors instead of scripted dialogue.

---

##  Retrieval-Augmented Generation (RAG)

- Retrieves knowledge from authoritative medical textbooks.
- Reduces hallucinations of Large Language Models.
- Grounds patient responses and evaluations with reliable medical evidence.

---

##  Intelligent Evaluation Framework

The evaluation framework assesses users from three perspectives:

- Diagnostic Accuracy
- Completeness of Information Collection
- Clinical Communication Quality

The system automatically generates personalized feedback after each interview.

---

##  Evidence-based Feedback

Instead of only providing scores, the evaluation report cites supporting knowledge retrieved from the medical knowledge base, improving transparency and interpretability.

---

##  Voice Interaction

Supports speech input to simulate real-world doctor-patient communication and improve user experience.

---

##  Continuous Practice

The system generates different standardized patient cases, allowing users to repeatedly practice clinical interviews and gradually improve diagnostic thinking.

---

# System Architecture

```text
                     +----------------------+
                     |        User          |
                     +----------------------+
                                |
                                v
                    Clinical Interview
                                |
                                v
                     +------------------+
                     | Prompt Templates |
                     +------------------+
                                |
                +---------------+---------------+
                |                               |
                v                               v
      Medical Knowledge Base              Large Language Model
      (medical_knowledge_db)                    (LLM)
                |                               |
                +---------------+---------------+
                                |
                                v
                 AI Standardized Patient (SP)
                                |
                                v
                   Diagnosis Submission
                                |
                                v
                   Evaluation Framework
                                |
                                v
                  Personalized Feedback
```

---

# Workflow

```text
Generate Clinical Case
          │
          ▼
Clinical Interview
          │
          ▼
Submit Diagnosis
          │
          ▼
Retrieve Medical Knowledge
          │
          ▼
Evaluate Interview
          │
          ▼
Generate Feedback Report
```

---

# Project Structure

```text
.
├── .devcontainer/              # Development environment configuration
├── medical_knowledge_db/       # Vector database generated from medical textbooks
├── web+voice.py                # Main application entry
├── requirements.txt            # Project dependencies
├── .gitignore
└── README.md
```

---

# Technology Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Web Framework | Streamlit |
| LLM | DeepSeek API |
| Prompt Design | Custom Prompt Templates |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Embedding Model | DeepSeek Embeddings |
| Speech Input | streamlit-mic-recorder |

---

# My Contribution

As a member of the project team, my primary contributions included:

- Designed and iteratively refined prompt strategies for AI standardized patient generation.
- Participated in designing and improving the evaluation framework through repeated testing and feedback.
- Conducted functional testing, identified failure cases, and proposed improvements to interaction logic and user experience.
- Assisted in refining the overall workflow of the system from clinical interview to evaluation.

---

# Future Work

Potential future improvements include:

- Expanding the medical knowledge base with additional authoritative textbooks.
- Supporting more medical specialties and disease categories.
- Introducing multimodal assessment using facial expressions and speech analysis.
- Enhancing OSCE-style clinical examination simulation.
- Deploying the platform as an online web service.

---

# Acknowledgements

This project was completed through multidisciplinary collaboration between students from Mathematics and Nursing.

It explores the application of Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) in clinical communication training and medical education.

---

# Disclaimer

This project is intended **for educational and research purposes only**.

It is **not** a medical diagnosis system and should **not** be used for clinical decision-making.
