\# SHANI — System Overview (Master Document)



\## 1. System Definition



SHANI (Scientific Harvesting and Analysis of Networked Information) is a \*\*deterministic, database-driven research pipeline\*\* designed to generate structured scientific literature reviews from academic papers.



The system enforces:

\- strict stage-based execution

\- persistent workflow state

\- programmatic control over reasoning

\- LLM usage only for controlled text generation



\---



\## 2. Core Philosophy



SHANI is NOT an autonomous AI agent.



It is a \*\*controlled research system\*\* where:

\- logic = code

\- memory = database

\- language = LLM



\### Key Principles



\- Deterministic execution

\- Sequential pipeline

\- No hidden reasoning

\- Database-first design

\- Separation of knowledge and writing



\---



\## 3. System Architecture



SHANI consists of five layers:



\### 1. CLI Layer

Entry point for execution



Responsibilities:

\- workflow creation

\- execution control

\- monitoring



Main file:

core/shani.py



\---



\### 2. Workflow Layer

Controls pipeline execution



Responsibilities:

\- stage transitions

\- failure handling

\- resume capability



Main file:

core/orchestrator.py



\---



\### 3. Repository Layer

Handles database interaction



Responsibilities:

\- data persistence

\- transaction control

\- structured storage



Key modules:

\- workflow\_repo

\- stage\_repo

\- paper\_repo

\- paper\_content\_repo

\- research\_knowledge\_repo

\- draft\_section\_repo



\---



\### 4. Tool Layer

Implements research pipeline logic



Each tool follows:

tool(repo, workflow\_id, \*\*kwargs)



Core tools:

\- generate\_queries

\- search\_papers

\- process\_papers

\- extract\_paper\_content

\- extract\_research\_knowledge

\- draft\_sections

\- synthesize\_paper



\---



\### 5. Service Layer

Provides supporting functionality



Includes:

\- llm\_service

\- vector\_db\_service

\- knowledge\_compressor

\- evaluation\_service



\---



\## 4. Research Pipeline (S1–S7)



SHANI executes a fixed 7-stage pipeline:



\### S1 — Generate Queries

Input: research topic  

Output: structured search queries  



\---



\### S2 — Search Papers

Input: queries  

Output: paper metadata  



Sources:

\- OpenAlex

\- Semantic Scholar

\- arXiv



\---



\### S3 — Process Papers

Responsibilities:

\- download PDFs

\- validate files

\- store locally



\---



\### S4 — Extract Paper Content

Tools:

\- GROBID

\- PyMuPDF



Extracted:

\- title

\- abstract

\- sections

\- references



\---



\### S5 — Extract Research Knowledge

Converts content into structured data



Extracted entities:

\- materials

\- synthesis methods

\- characterization techniques

\- simulation methods

\- applications



Important:

Paper ≠ Study (multiple knowledge records per paper)



\---



\### S6 — Draft Sections

Input:

\- structured knowledge

\- paper metadata



Output:

\- section-wise literature drafts



\---



\### S7 — Synthesize Paper

Combines all sections into final document



Output:

results/review\_paper.docx



\---



\## 5. Database Model



Database file:

database/research\_workflow.db



\### Core Tables



\- Workflow

\- Stage

\- ExecutionAttempt

\- Paper

\- PaperContent

\- ResearchKnowledge

\- DraftSection

\- FinalPaperSection

\- FailureLog



\---



\## 6. Data Flow



Pipeline:



Research Topic  

↓  

Queries  

↓  

Paper Metadata  

↓  

PDF Content  

↓  

Structured Knowledge  

↓  

Draft Sections  

↓  

Final Literature Review  



\---



\## 7. LLM Role



LLM is used ONLY for:

\- summarization

\- structured writing

\- controlled transformation



LLM is NOT used for:

\- pipeline control

\- data storage

\- decision making



\---



\## 8. Vector Database Role



Vector DB is used for:

\- semantic retrieval

\- similarity search

\- context compression



Files:

\- vector\_index.faiss

\- vector\_index.faiss.map.npy



\---



\## 9. Execution Model



\- Sequential processing

\- No parallel writes

\- Deterministic stage transitions

\- Persistent workflow state



Typical run:

50–60 papers  

20–30 minutes  



\---



\## 10. Error Handling



\- Each stage tracked in Stage table

\- Failures logged in FailureLog

\- Retry supported via ExecutionAttempt



\---



\## 11. Current System State



Implemented:

\- Full S1–S7 pipeline

\- Knowledge compression (S6)

\- Multi-source retrieval

\- PDF extraction pipeline

\- Database integration

\- Vector DB integration



\---



\## 12. Design Constraints



\- No architectural changes without versioning

\- Tools must follow standard interface

\- Database schema is authoritative

\- Pipeline must remain sequential



\---



\## 13. Future Direction



\- improved knowledge clustering (S6)

\- citation-aware generation

\- vector retrieval optimization

\- SHANI-specific LLM training

\- evaluation metrics integration



\---



\## 14. How to Work With SHANI (Important)



When modifying the system:



Always think in terms of:

workflow → stage → tool → database



Do NOT:

\- introduce randomness

\- bypass stages

\- mix responsibilities across layers



\---



\## 15. Summary



SHANI is a \*\*deterministic research system\*\* that transforms scientific papers into structured knowledge and generates literature reviews through a controlled, stage-based pipeline.



It prioritizes:

\- reliability over creativity

\- structure over improvisation

\- system design over raw LLM usage

