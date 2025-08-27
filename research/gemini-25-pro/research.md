# A Strategic Analysis of PDF Information Extraction Approaches for 2025

## The Evolving Landscape of Document Intelligence in 2025

### Introduction: Beyond Traditional OCR to Comprehensive Document Understanding

The field of automated document processing is undergoing a fundamental paradigm shift. For decades, the primary technology was Optical Character Recognition (OCR), a process focused on the singular task of converting pixels from a document image into a stream of machine-readable characters.1 While foundational, this approach treated documents as flat text files, largely ignoring the rich structural and semantic context embedded in their visual layout. By 2025, this legacy view has been rendered obsolete by the maturation of Intelligent Document Processing (IDP) and Document AI, which have transformed the objective from mere "text recognition" to holistic "document understanding".3

This evolution is not incremental; it represents a qualitative leap in capability, driven by significant advancements in machine learning. The widespread adoption of deep learning architectures, particularly Convolutional Neural Networks (CNNs) and Transformers, has enabled models to interpret visual and textual data with unprecedented accuracy.3 More recently, the emergence of large-scale multimodal foundation models, often referred to as Vision Language Models (VLMs), has introduced the ability to process and reason about visual and linguistic information simultaneously, unlocking new frontiers in contextual interpretation.4

The transition to comprehensive document understanding unlocks substantial business value by automating complex, previously manual workflows. Use cases now span a wide array of industries. In financial services, firms can process loan and mortgage applications in minutes by accurately extracting applicant names, mortgage rates, and invoice totals from diverse forms.6 Healthcare organizations can better serve patients by extracting critical data from intake forms and insurance claims, eliminating manual review and preserving the original context of the information.6 Government agencies can process small business loan applications and tax forms with high accuracy, while logistics and supply chain companies can digitize bills of lading and customs forms to improve traceability and reduce data entry errors.3 This shift reduces operational costs, accelerates business cycles, enhances data accessibility, and fortifies compliance and audit readiness.3

### The Central Dichotomy: Specialized Pipelines vs. General-Purpose Vision Models

As organizations seek to harness these advanced capabilities, the technological landscape in 2025 is defined by two dominant, and at times competing, architectural philosophies for information extraction from PDFs. Understanding this dichotomy is critical for making sound strategic investments in document intelligence technology.

The first approach can be defined as the **Specialized OCR Pipeline**. This is a multi-stage, highly optimized workflow composed of discrete, purpose-built components. A typical pipeline begins with object detection models to locate specific page elements like tables, charts, or text blocks. It then applies specialized OCR and structure-aware models tailored to each element type to perform extraction.8 Exemplars of this methodology, such as the NVIDIA NeMo Retriever PDF Extraction pipeline, are engineered for high throughput, low latency, and fine-grained control over each processing step.8 This architecture excels in environments with predictable document types, such as standardized government forms, where its linear scalability and high precision yield significant operational efficiencies.2

The second approach is the **General-Purpose Vision Language Model (VLM)**. This methodology is more holistic and end-to-end, leveraging a single, large multimodal model to process a document page as an image and directly generate structured output or answer natural language queries about its content.5 Leading models such as Meta's Llama 3.2 Vision, Google's Gemini 2.5 Pro, and Anthropic's Claude 3.7 Sonnet epitomize this approach.5 The core strength of VLMs lies in their profound contextual and semantic understanding. This enables them to handle documents with highly variable or unpredictable layouts and to infer information that is not explicitly stated in the text, such as interpreting the relative heights of bars in a chart that lacks numerical labels.2

These two architectural philosophies reflect fundamentally different conceptions of the document extraction task. The specialized pipeline treats it as an engineering problem to be deconstructed and solved with a suite of optimized, purpose-built tools for each sub-task—layout detection, table parsing, text recognition, and so on. This is a classic divide-and-conquer strategy, exemplified by the microservice-based architecture of NVIDIA's NeMo Retriever, where each component can be independently optimized and evaluated.8 This approach prioritizes deterministic, measurable performance, leading to high throughput and predictable accuracy on known document types.2

In contrast, the VLM approach treats document extraction as a reasoning problem. It posits that a single, powerful intelligence can comprehend the document as a whole, much like a human analyst. Prompting a VLM involves presenting it with the entire page image and issuing a high-level instruction, such as "extract the line items from this invoice" or "summarize the key obligations in this contract".8 This method relies on the model's emergent reasoning capabilities to understand the interplay of text, layout, and visual elements. It prioritizes adaptability and flexibility; while it may be slower and more computationally expensive per page, a VLM can often successfully process a novel document format it has never encountered because it understands the underlying

*concept* of an invoice or a contract.2

The choice between these architectures, therefore, is not merely technical but strategic. Investing in a specialized pipeline is a commitment to building a highly efficient "factory" optimized for specific, high-volume document workflows. Investing in a VLM is a commitment to developing a flexible, generalist "analyst" capable of handling a wider range of ad-hoc and unforeseen tasks, albeit at a different performance and cost profile. The remainder of this report will delve into the trade-offs—speed, cost, control, flexibility, and contextual reasoning—that define this critical decision point.2

## Deep Dive into Methodologies and Leading Solutions

### Specialized OCR and Intelligent Document Processing (IDP) Platforms

The market for specialized document processing is dominated by a mix of large cloud providers offering managed services, powerful open-source engines that serve as foundational components, and enterprise-grade frameworks that orchestrate these components into production-ready pipelines.

#### Cloud Behemoths: The Managed IDP Ecosystems

The major cloud providers have developed mature, scalable, and feature-rich IDP platforms that abstract away much of the underlying complexity of building and maintaining an extraction pipeline.

* **Amazon Textract:** Amazon's offering is positioned as a machine learning service that moves significantly beyond basic OCR.6 Its core capabilities include robust form extraction, which identifies key-value pairs (e.g., "First Name": "Jane") while preserving their relationship, and table extraction, which maintains row and column structures for easy ingestion into databases.13 Textract also provides layout analysis to identify elements like paragraphs, titles, and headers, as well as signature detection.13 A key differentiator is its
  Queries feature, which allows users to specify the data they need using natural language questions (e.g., "What is the customer name?"), receiving a precise answer without needing to define templates or understand the document's structure.13 Crucially, this feature can be customized for domain-specific documents using
  **Adapters**. This mechanism allows users to upload a small set of sample documents (as few as ten), annotate the data, and fine-tune the pre-trained Queries model within hours, representing a streamlined pathway to high accuracy on proprietary document types.13
* **Azure AI Document Intelligence:** Microsoft's platform, formerly known as Azure Form Recognizer, provides a clear and powerful distinction between its model offerings.1 It features a suite of pre-built models for common document types like invoices and receipts. For custom needs, it offers two distinct training paths:
  **custom template models**, which are ideal for documents with a fixed, static visual layout, and **custom neural models**, which leverage deep learning to handle documents with the same information but variable layouts (e.g., W-2 forms from different employers).16 The platform's API is comprehensive, supporting detailed layout analysis that can identify the role of paragraphs (e.g., title, section heading, footer) and offering add-on capabilities for recognizing mathematical formulas, barcodes, and fonts.17 The entire fine-tuning and model management lifecycle is integrated into the Azure AI Foundry, providing a centralized environment for data preparation, training, and deployment.19
* **Google Document AI:** Google's solution is centered around the Document AI Workbench, a platform for building, managing, and deploying **Custom Document Extractors (CDEs)**.21 A significant strategic advantage of Google's approach is its deep integration of powerful foundation models directly into the model development workflow. These models provide strong zero-shot prediction capabilities, meaning they can often extract relevant information from a document with no prior training. This is leveraged to
  **auto-label** new training documents, where the foundation model provides initial annotations that a human reviewer then simply needs to verify.21 This dramatically accelerates the most time-consuming part of custom model development: data labeling. The workflow involves defining a schema of required fields, importing documents, using the auto-labeling feature, verifying the labels, and then training a new, fine-tuned processor version.22

#### High-Performance Open Source Engines

For organizations that require greater control, flexibility, or wish to avoid vendor lock-in, open-source engines provide the core components for building custom pipelines.

* **Tesseract OCR:** As one of the most mature and widely used open-source OCR engines, Tesseract is known for its high degree of flexibility, extensive language support (over 100 languages), and strong community backing.1 Originally developed by Hewlett-Packard and now maintained by Google, it serves as a powerful workhorse for text recognition. However, it is not a complete document understanding solution out of the box. Achieving high accuracy with Tesseract often requires significant image pre-processing, including deskewing, denoising, and binarization.27 Furthermore, Tesseract does not process PDF files directly; it requires a wrapper library like
  pypdfium2 to first render PDF pages as images.1 While highly capable for raw text extraction, it can struggle with complex, multi-column layouts and structured data like tables, positioning it best as a foundational OCR component within a larger, more sophisticated pipeline.1
* **PaddleOCR:** Developed by the PaddlePaddle community, PaddleOCR has emerged as a modern, high-performance open-source alternative that is both lightweight and fast.1 Its key strengths lie in its comprehensive, end-to-end toolkit and exceptional multilingual capabilities, with support for over 80 languages, including complex scripts like Chinese, Japanese, and Korean.1 The toolkit includes a suite of high-quality models such as
  PP-OCR for general recognition, PP-Structure for layout and table analysis, and PP-ChatOCR for key information extraction.1 A major advantage of PaddleOCR is its deployment flexibility; it is designed for efficient integration across various platforms, including servers, mobile applications, and embedded systems, making it suitable for real-time applications where performance is critical.1

#### Enterprise-Grade Frameworks: Orchestrating the Pipeline

Bridging the gap between individual components and a fully managed cloud service are enterprise-grade frameworks that provide a structured, scalable, and production-ready environment for building document intelligence pipelines.

* **NVIDIA NeMo Retriever:** This framework is an enterprise-grade reference implementation for building high-performance information retrieval and extraction pipelines, particularly for Retrieval-Augmented Generation (RAG) applications.8 Its architecture is built upon
  **NVIDIA Inference Microservices (NIMs)**, a collection of containerized, optimized models for specific tasks.9 The extraction pipeline is a prime example of the specialized, multi-stage approach. It includes NIMs for object detection (to locate tables and charts), multiple OCR options (including a VLM-based
  Parse service and a highly optimized PaddleOCR NIM), embedding generation, and result reranking.12 Performance benchmarks demonstrate the power of this specialized architecture, showing that the NeMo Retriever OCR pipeline can achieve an end-to-end latency of just 0.118 seconds per page on an NVIDIA A100 GPU—a 32.3x throughput advantage over a much larger, general-purpose VLM attempting the same extraction task.8
* **Docling:** Docling is an open-source framework focused on a critical challenge: simplifying the conversion of diverse document formats (including PDF, DOCX, and PPTX) into a unified, richly structured intermediate representation.32 It excels at understanding complex document layouts by leveraging state-of-the-art models like
  DocLayNet for layout parsing and TableFormer for table structure recognition.32 The output is a
  DoclingDocument object that preserves not just text, but also tables, images, code blocks, and layout information. This structured output can then be easily exported to formats like Markdown or JSON, making it ideal for feeding downstream LLM pipelines.32 A key feature is its pluggable architecture, which allows users to select the most appropriate OCR backend for their needs, with support for EasyOCR and Tesseract, among others.35 This makes Docling a powerful and flexible "front-end" for any document processing workflow, with native integrations for popular ecosystems like LangChain and LlamaIndex.32

### Vision Language Models (VLMs) for Document Understanding

VLMs represent a fundamentally different approach to document extraction, relying on holistic visual-semantic reasoning rather than a sequence of discrete processing steps.

#### State-of-the-Art Models and Capabilities

The latest generation of VLMs has demonstrated remarkable capabilities in understanding and extracting information from complex documents.

* **Llama 3.2 Vision (11B & 90B):** Meta's latest multimodal models showcase strong capabilities for document-level understanding. They can interpret not only text but also complex visual elements like charts and graphs.11 A significant advancement is their ability to perform direct table extraction by treating the PDF page as a single image input and generating a structured representation (e.g., Markdown) of the table, a task that has traditionally required specialized table recognition models.36 The potential for domain specialization is high, as demonstrated by models like
  Cephalo-Llama-3.2, which has been fine-tuned on a large corpus of scientific papers to improve its performance on technical documents.37
* **Gemini 2.5 Pro & Claude 3.7 Sonnet:** These leading commercial VLMs have set new benchmarks for accuracy across a wide range of challenging document types. In comparative studies, they have shown exceptional performance on tasks that are notoriously difficult for traditional OCR, such as extracting information from handwritten physician's notes, where Claude 3.7 Sonnet achieved 92% accuracy.10 Gemini 2.5 Pro is noted for its consistently high accuracy across many languages, including low-resource languages, making it a strong choice for global applications.10 Claude's large context window gives it an advantage in maintaining consistency and understanding relationships across long, multi-page documents.38

#### Architectural Advantages and Limitations

The primary architectural advantage of VLMs is their capacity for **visual-semantic reasoning**. They do not just see text; they understand its context based on its position, style, and proximity to other elements like images, captions, and tables. This allows them to succeed in scenarios where traditional OCR systems fail. For instance, a VLM can answer a question like "What is the trend shown in the bar chart?" by visually interpreting the chart's structure, even if the specific data points are not explicitly listed as text.8 This makes them invaluable for extracting information from infographics, complex reports, and documents with unpredictable layouts.2

However, this powerful reasoning capability comes with significant trade-offs. Compared to specialized OCR pipelines, VLMs exhibit:

* **Higher Latency and Lower Throughput:** Processing a full-page image through a large VLM is computationally intensive, resulting in slower response times that may not be suitable for real-time applications.8
* **Increased Cost:** VLMs tend to produce more verbose, narrative-style outputs, which increases the number of tokens generated. In a pay-per-token pricing model, this directly translates to higher inference costs.8
* **Scalability and Reliability Concerns:** When relying on third-party VLM APIs, organizations may face rate limiting, concurrency restrictions, or unpredictable performance during peak usage periods, which can pose a significant business risk for mission-critical, high-volume workflows.2

### The Power of Synthesis: Hybrid and Composable Systems

The analysis of specialized pipelines versus general-purpose VLMs reveals that the "best" approach is rarely a binary choice. Instead, the most robust and efficient solutions in 2025 are increasingly hybrid, combining the strengths of both methodologies. This trend points toward a broader shift in enterprise AI strategy: the move away from seeking a single, monolithic solution and toward building **composable systems**. Sophisticated organizations are effectively creating a "Document Operating System" by assembling a portfolio of best-in-class components—often delivered as microservices—for each specific task in the document intelligence lifecycle.

This architectural philosophy acknowledges that different tools are superior for different jobs. Specialized engines like Tesseract or PaddleOCR are excellent at high-speed, accurate text recognition 1; layout analysis models like

DocLayNet (used by Docling) excel at identifying document structure 32; and VLMs are unparalleled at high-level semantic reasoning and exception handling.2 The future of document processing, therefore, lies not in finding the "one best tool" but in creating a flexible, orchestrated workflow that can dynamically leverage the unique strengths of different models and services. This composable approach maximizes architectural flexibility, prevents vendor lock-in, and allows the system to evolve as new, more powerful components become available.

#### Architectural Patterns for Hybrid Systems

Several effective patterns for building these hybrid systems have emerged:

* **Pattern 1: OCR for Broad Extraction, VLM for Targeted Reasoning:** This is a highly cost-effective pattern. A fast, low-cost specialized pipeline (e.g., using Tesseract or Amazon Textract's basic OCR) is first used to extract all raw text, tables, and layout information from a document. Then, for specific, challenging tasks—such as parsing complex, multi-line invoice items or interpreting a nuanced clause in a legal contract—only that small, relevant snippet of text (or the corresponding image crop) is sent to a powerful VLM for deep parsing and structuring. This approach balances the speed and scale of specialized OCR with the deep contextual understanding of a VLM, optimizing for both performance and cost.2
* **Pattern 2: VLM for Classification, Specialized Models for Extraction:** In workflows that handle a variety of document types, a VLM can be used as an intelligent "receptionist." The VLM's first task is to perform initial document classification (e.g., "This is an invoice from Vendor A," "This is a bill of lading," "This is a patient intake form"). Based on this classification, an orchestration layer then routes the document to the appropriate downstream extraction model, which could be a custom-trained Azure neural model, a fine-tuned Docling pipeline, or a specific template-based extractor. This ensures that each document is processed by the model best suited for its specific layout and content, maximizing accuracy.3

#### A Practical Decision Matrix

To guide the selection of the optimal approach, the following matrix maps common document types to the most suitable extraction methodology, providing the underlying rationale for each recommendation.

| Document Type | Best Approach | Reasoning |
| --- | --- | --- |
| **Standard Forms (e.g., W-9, 1099)** | OCR | The layout is consistent and unchanging. High accuracy (>99%) is achievable with template-based or specialized neural models. Control and predictability are paramount.2 |
| **ID Documents (e.g., Passports)** | OCR | Formats are highly standardized. Security and data privacy are major concerns, favoring specialized, often on-premise, OCR solutions with specific security features.2 |
| **Invoices** | Hybrid | Semi-structured but highly variable between vendors. A hybrid approach is optimal: use a specialized OCR pipeline to extract header data (invoice number, date, vendor name) and table structures, then use a VLM to parse complex or non-standard line items.2 |
| **Financial Statements** | Hybrid | Contains highly structured tables that are best handled by specialized table extraction models (like those in Docling or NeMo Retriever). However, interpreting footnotes and management discussion sections requires the contextual understanding of a VLM.2 |
| **Resumes** | Hybrid | The overall structure (contact info, experience, education) is conceptually consistent, but the visual layout varies wildly. A hybrid approach can use layout-aware OCR to identify sections and a VLM to parse the content within those sections.2 |
| **Receipts** | LLMs / VLMs | Formats are extremely variable. Key information often requires contextual understanding (e.g., distinguishing the total amount from other figures). VLMs excel at this task.2 |
| **Handwritten Notes** | LLMs / VLMs | Text is irregular and often requires contextual clues to be deciphered correctly. Modern VLMs like Claude and Gemini have shown strong performance on this challenging task.2 |
| **Legal Contracts** | LLMs / VLMs | Extraction requires deep semantic understanding of complex language, nested clauses, and inter-document references. This is a core strength of large language models.2 |
| **Medical Records** | LLMs / VLMs | Documents often contain a mix of structured data, unstructured physician's notes, and complex relationships between entities. VLMs are needed to interpret this complexity.2 |

## Empirical Analysis: Performance, Benchmarks, and Cost

A strategic decision requires a rigorous, data-driven evaluation of solution performance and economic viability. This section moves beyond marketing claims to analyze the empirical evidence from technical benchmarks and provide a clear-eyed view of the total cost of ownership.

### A Multi-Dimensional View of Accuracy

Evaluating the performance of a document intelligence system requires a nuanced understanding of accuracy that goes far beyond a single percentage. Different metrics are critical for different aspects of the extraction task, and a comprehensive evaluation must consider them all.39

* **Character Error Rate (CER) and Word Error Rate (WER):** These are the most fundamental metrics for assessing the quality of raw text transcription. CER measures the percentage of characters that were incorrectly substituted, inserted, or deleted, calculated using the Levenshtein distance. WER operates at the word level and is calculated as WER=(S+D+I)/N, where S is substitutions, D is deletions, I is insertions, and N is the total number of words in the ground truth. These metrics are essential for evaluating the core OCR engine's performance.39
* **Exact Match Rate (EMR):** For structured data extraction, EMR is paramount. It measures the percentage of fields where the extracted value is a perfect, character-for-character match with the ground truth. For critical data points like an invoice number, a social security number, or a total amount, a near-miss is a complete failure. EMR provides a strict, unforgiving measure of this performance.39
* **F1 Score (Precision and Recall):** The F1 score is the harmonic mean of precision and recall and is the most important metric for evaluating named entity extraction. **Precision** measures the accuracy of the extracted values (i.e., of all the fields the model extracted, what proportion were correct?). **Recall** measures the completeness of the extraction (i.e., of all the fields that *should* have been extracted, what proportion did the model find?). The F1 score provides a single, balanced measure of a model's ability to find the right information without extracting incorrect information.39
* **Table Edit Distance Score (TEDS):** Extracting tables is a notoriously difficult task because it requires recognizing not just the text within cells but also the correct row and column structure. TEDS is a specialized metric designed to evaluate this structural accuracy. It measures the "edit distance" or the number of operations (e.g., splitting cells, merging rows) required to transform the extracted table structure into the ground truth structure. A high TEDS score is indicative of a system that can reliably parse complex tables.30

It is crucial to note that the performance on all these metrics is highly dependent on the quality of the input document. The industry standard minimum resolution for optimal OCR results is 300 DPI. Documents with low resolution, poor contrast, background noise, or significant skew will degrade the performance of any system.10

### Comparative Performance Analysis

Synthesizing data from multiple independent benchmarks provides a clear picture of the current performance landscape. The OmniDocBench, for instance, offers a rigorous, end-to-end evaluation of various tools and models across diverse document types, including academic papers, financial reports, and handwritten notes.41

Key performance trends observed in 2025 include:

* **Printed Text Accuracy:** For clean, printed documents, top-tier commercial services like Google Cloud Vision and AWS Textract, along with leading open-source engines like PaddleOCR, consistently achieve character accuracy rates exceeding 99%, approaching the theoretical limit.41
* **Handwriting Recognition:** This remains a challenging domain where VLMs currently hold a significant advantage. Benchmarks show leading models like Claude 3.7 Sonnet achieving up to 92% accuracy on handwritten physician notes, while Gemini 2.5 Pro and HandwritingOCR achieve rates around 84-95%.10 Traditional enterprise OCR platforms often struggle with cursive or messy handwriting, with accuracy dropping significantly.38
* **Table and Structure Extraction:** In structured tasks, specialized pipeline tools often outperform generalist VLMs. On the OmniDocBench, tools like PaddleOCR's PP-StructureV3 and MinerU demonstrate top-tier performance in table recognition (TEDS) and overall document structure preservation, as measured by edit distance.41 Similarly, a focused benchmark on sustainability reports showed the Docling framework achieving 97.9% cell accuracy on complex tables, significantly outperforming a competitor like Unstructured, which only managed 75% on the same task.44
* **Multilingual Performance:** VLMs like Gemini 2.5 Pro have demonstrated exceptional consistency across a wide range of languages, including those with fewer digital resources, making them a strong choice for global enterprises.10 Open-source engines like PaddleOCR and Tesseract also offer extensive language support.1

The following table provides a strategic, at-a-glance comparison of leading solutions, synthesizing data from across the research.

**Comprehensive Solution Benchmark (2025)**

| Solution | Primary Approach | Accuracy (Structured) | Accuracy (Unstructured) | Handwriting Accuracy | Table Extraction Fidelity | Latency / Throughput | Cost Model | Customization |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Amazon Textract** | Specialized Pipeline | High (95-99%) | Medium | Medium (~48-70%) | High | Low / High | Pay-per-page API | Medium (Adapters) |
| **Azure Doc. Intel.** | Specialized Pipeline | High (96-98%) | Medium-High | Medium (~70-85%) | High | Low / High | Pay-per-page API | High (Custom Models) |
| **Google Doc. AI** | Specialized Pipeline | High (96-98%) | Medium-High | High (~84-94%) | High | Low / High | Pay-per-page API | High (Workbench) |
| **NVIDIA NeMo Ret.** | Specialized Pipeline | High | High | N/A | High | Very Low / Very High | Infrastructure | High (Full Pipeline) |
| **Tesseract** | OCR Component | Medium-High | Low | Low | Low | Low / High | Infrastructure (Open Source) | High (Fine-tuning) |
| **PaddleOCR** | Specialized Pipeline | High | Medium | Medium | Very High (TEDS ~84%) | Low / High | Infrastructure (Open Source) | High (Fine-tuning) |
| **Docling** | Hybrid Framework | High | High | N/A | Very High (97.9% cell acc.) | Medium / Medium | Infrastructure (Open Source) | High (Pluggable) |
| **Llama 3.2 Vision** | VLM | Medium | High | Medium-High | Medium-High | High / Low | Token-based / Infra. | High (Fine-tuning) |
| **Gemini 2.5 Pro** | VLM | Medium-High | High | High (~84%) | High | High / Low | Token-based API | Medium (API) |
| **Claude 3.7 Sonnet** | VLM | Medium-High | Very High | Very High (~92%) | High | High / Low | Token-based API | Medium (API) |

Note: Accuracy figures are synthesized estimates from multiple sources.10 Latency/Throughput are relative comparisons; NeMo Retriever's 0.118s/page provides a quantitative baseline.8

### Economic Considerations: Total Cost of Ownership (TCO)

A comprehensive financial analysis must extend beyond the advertised price per page to consider the total cost of ownership, which includes development, infrastructure, maintenance, and the business cost of uncorrected errors.

* **API-based Pricing (Operating Expense - OpEx):** The major cloud providers (AWS, Azure, Google) operate on a tiered, pay-as-you-go model.45 For basic text detection (OCR), prices are competitive, typically around $1.50 per 1,000 pages for the first million pages, dropping to $0.60 thereafter.45 However, costs escalate significantly as more advanced features are used. For example, using Amazon Textract for form extraction costs $50 per 1,000 pages, and adding table and query extraction can bring the total to $70 per 1,000 pages.45 This model offers low upfront cost and scalability but can become expensive for high-volume, complex extraction workflows.
* **Infrastructure-based Pricing (Capital/Operating Expense - CapEx/OpEx):** Deploying open-source or self-hosted enterprise frameworks shifts the cost from API calls to infrastructure and personnel. This includes the cost of high-performance computing resources (e.g., NVIDIA A100 GPUs for running NeMo Retriever 8), storage, and networking. More importantly, it includes the "hidden cost" of the high development and maintenance effort required to build, deploy, and manage a resilient, production-grade pipeline.2 While potentially cheaper at massive scale, this approach requires a significant upfront investment and a skilled MLOps team.
* **The Accuracy-Cost Curve:** A critical strategic consideration is the non-linear relationship between accuracy and cost. Analysis shows this relationship follows a logarithmic curve: improving accuracy from 80% to 90% incurs a moderate cost increase, moving from 90% to 95% is substantial, and pushing from 95% to 99% can be exponentially expensive.10 This has profound implications for return on investment (ROI). For many business processes, achieving 99.9% straight-through processing may be prohibitively expensive. A more cost-effective strategy is often to engineer a system that achieves 95% accuracy and integrates a robust, efficient human-in-the-loop validation workflow to handle the remaining 5% of exceptions. A 4% improvement in OCR accuracy (from 95% to 99%) can lead to an 80% reduction in the volume of documents requiring manual verification, a tangible operational saving that must be weighed against the cost of achieving that accuracy gain.43

## Implementation in Practice: From Pilot to Production

Deploying a successful document intelligence solution requires a well-architected pipeline, a clear understanding of operational processing paradigms, and a strategic approach to model customization.

### Architecting the Modern Document Processing Pipeline

A production-grade pipeline is a multi-stage workflow designed for robustness, accuracy, and integration with core business systems. The following blueprint synthesizes best practices for building such a system.3

1. **Ingestion:** The pipeline begins by capturing documents from all relevant sources, which may include physical scanners, email inboxes, mobile application uploads, or cloud storage buckets (e.g., Amazon S3).3
2. **Preprocessing:** This is a critical and often underestimated stage. Raw document images are cleaned and standardized to optimize them for the OCR engine. Common techniques include **deskewing** to correct rotational alignment, **denoising** to remove artifacts like speckles or stains, **binarization** to convert the image to black and white, and **contrast enhancement**. Implementing a robust preprocessing module often delivers the single largest improvement in overall accuracy.3
3. **Classification:** For workflows handling multiple document types, it is highly efficient to classify the document *before* attempting to extract data. An accurate classification step (e.g., identifying a document as an "invoice" versus a "bill of lading") allows the system to route the document to the most appropriate specialized extraction model or template, significantly boosting accuracy.3
4. **Extraction:** This is the core of the pipeline, where the chosen methodology—be it a specialized OCR pipeline, a VLM, or a hybrid system—is applied to the preprocessed image to extract text, layout, and structured data.
5. **Post-processing and Validation:** Raw extracted data is rarely ready for business use. This stage transforms it into validated, structured information. Key steps include:
   * **Validation Rules:** Applying rules to check the format and validity of extracted fields, such as using regular expressions (regex) for dates and phone numbers, or applying checksum validation for tax IDs and IBANs.3
   * **Confidence Scoring:** Using the model's confidence scores to flag low-confidence extractions for review.7
   * **Human-in-the-Loop (HITL):** Surfacing all flagged exceptions or documents below a certain confidence threshold to a human operator via a review user interface (UI). This step is crucial for ensuring near-perfect effective accuracy for critical data.3
6. **Integration:** The final stage involves pushing the validated, structured data into downstream business systems. This is typically achieved via API calls to Enterprise Resource Planning (ERP) systems, Customer Relationship Management (CRM) platforms, or by exporting the data to a database or data warehouse for analytics.3

### Processing Paradigms: Real-Time vs. Batch

The operational requirements of the business use case will dictate the choice between two primary processing paradigms.50

* **Real-Time Processing:** This paradigm is characterized by the immediate, low-latency processing of documents as they are generated or received. It prioritizes immediacy and responsiveness over raw efficiency.50 Real-time processing is essential for interactive, customer-facing workflows such as Know Your Customer (KYC) onboarding in financial services (where an ID must be verified instantly), point-of-sale receipt analysis for loyalty programs, or initial processing of an insurance claim submitted via a mobile app.3 Architecting a real-time system requires significant technical expertise to handle high data throughput and potential spikes in volume while maintaining millisecond-level latencies. It often involves more complex and costly infrastructure.50
* **Batch Processing:** In contrast, batch processing involves accumulating documents over a defined period and processing them in large, scheduled jobs. This approach prioritizes computational efficiency, thoroughness, and cost-effectiveness.50 It is the ideal choice for non-time-sensitive, high-volume tasks such as digitizing historical archives, performing end-of-day financial reporting, or processing large volumes of invoices for accounts payable where a 24-hour turnaround is acceptable.5 Batch systems are generally less complex to design and can be optimized to run during off-peak hours to minimize resource costs.52
* **Hybrid Architectures:** For organizations with diverse needs, sophisticated hybrid architectures like the Lambda or Kappa architectures have emerged. These systems combine a "speed layer" that processes incoming data in real-time to provide immediate insights with a "batch layer" that manages the master dataset and performs comprehensive, large-scale analysis. A serving layer then merges the results from both, giving users access to both up-to-the-second and historical data, offering the best of both worlds.51

### The Critical Path to Accuracy: Customization and Fine-Tuning

While pre-trained models offer impressive out-of-the-box performance, achieving the high levels of accuracy required for mission-critical business applications almost always necessitates some form of customization or fine-tuning on domain-specific data. The technical steps for this process are becoming increasingly accessible, but it is crucial to recognize that the ultimate success of any customization effort is overwhelmingly dependent on the quality, quantity, and representativeness of the training data. The principle of "garbage in, garbage out" remains the fundamental law of machine learning. The most significant cost and effort in a custom model project are not the compute cycles for model training but the human-intensive process of curating, cleaning, and accurately labeling the ground truth dataset.14 Therefore, project plans must allocate the majority of resources to data operations (DataOps), and investing in a high-quality data labeling platform or service is likely to yield a higher ROI than simply choosing a more advanced training algorithm.

#### Fine-Tuning Open Source Engines

* **Tesseract:** The Tesseract training process offers several levels of customization. The most common is **fine-tuning**, where an existing language model is further trained on a small set of specific data, such as a new font. A more involved option is to **cut off and retrain the top layers** of the neural network, which is useful if the new task is significantly different from the base model. The most demanding option is **retraining from scratch**, which requires a massive, representative dataset and significant computational resources.53 The general workflow involves creating "ground truth" data consisting of image files (e.g.,
  .png) and corresponding text files (.gt.txt), setting up the tesstrain repository environment, and executing the training command, specifying a base model (e.g., eng) and a fine-tuning type (e.g., Impact).54
* **PaddleOCR:** The fine-tuning workflow for PaddleOCR is well-documented and streamlined. It begins with setting up a dedicated environment (typically using Conda) and installing the paddlepaddle deep learning framework.56 The most critical step is data preparation: training and evaluation datasets must be created with annotation files that map each image to its transcription and bounding box coordinates (specifically, 8-point polygons for text regions).57 The process is controlled via a YAML configuration file where key parameters like the base pre-trained model, learning rate, batch size, and number of epochs are defined. Once configured, the training process is launched with a single command, which will fine-tune the model and save the resulting inference model for deployment.57

#### Training on Cloud Platforms: A Comparative Overview

* **AWS Textract (Adapters):** Amazon provides a user-friendly, console-driven workflow for customizing its Queries feature through **Adapters**. The process involves creating a new adapter, creating a dataset by uploading at least ten sample documents (five for training, five for testing), and defining the natural language queries for the data to be extracted. Textract's **auto-labeling** feature then uses the pre-trained model to generate initial answers and bounding boxes for these queries. The user's primary task is to review and verify these auto-generated labels in the annotation tool. Once the ground truth is verified, a new adapter version is trained. This approach abstracts away the complexities of deep learning, framing customization as a data annotation and verification task.14
* **Azure AI Document Intelligence (Custom Models):** Azure offers a more traditional machine learning workflow within the AI Foundry portal. Users create a project, prepare and upload their training and validation datasets to Azure Blob Storage, and then initiate a training job.20 A key decision is selecting the base model type: a
  **template model** for fixed layouts or a **neural model** for variable layouts.16 The portal allows for the configuration of training parameters before launching the job. Upon completion, the new custom model can be analyzed for performance and deployed as a REST endpoint for inference.20
* **Google Document AI (Workbench):** Google's approach is distinguished by its tight integration with foundation models to accelerate the workflow. The process begins with creating a **Custom Document Extractor** processor and defining the schema of fields to be extracted.21 When documents are imported for training, the user can enable the
  **auto-labeling** feature, which leverages a powerful foundation model to perform zero-shot extraction and populate the labels. The user then enters a labeling UI to review, correct, and confirm these AI-generated annotations. Once a sufficient dataset of verified labels is created (e.g., at least 10 training and 10 test instances per field), a new custom model-based processor version can be trained. This "human-in-the-loop" approach, where AI assists in creating the training data, significantly reduces the manual effort required to build a high-accuracy custom model.21

## Strategic Recommendations and Future Outlook

### Decision Framework for Technology Adoption

The selection of an optimal document intelligence solution is a multi-faceted decision that must align with an organization's specific technical, operational, and financial context. The following framework provides a set of strategic criteria to guide this decision-making process.

* **Document Variability:** This is the primary determinant of the required technology.
  + **Fixed Templates (e.g., W-9s):** Best served by OCR with custom template models (e.g., Azure Template Models) for maximum accuracy and cost-efficiency.
  + **Semi-Structured (e.g., Invoices):** Require hybrid systems or custom neural models (e.g., Google CDE, Azure Neural Models) that can handle layout variations.
  + **Highly Unstructured (e.g., Legal Contracts):** Best suited for VLMs that can perform deep semantic understanding.
* **Accuracy Requirement:** The business tolerance for errors dictates the level of investment.
  + **"Good Enough" (e.g., 95%) + Human Review:** A cost-effective strategy for many back-office processes, achievable with well-configured cloud APIs or open-source pipelines.
  + **Mission-Critical (e.g., >99%):** Requires significant investment in high-quality data labeling and custom model fine-tuning, potentially leveraging advanced models and rigorous validation workflows.
* **Processing Volume & Latency:** Operational needs define the architectural choice.
  + **High Volume / Batch:** Specialized pipelines like NVIDIA NeMo Retriever or self-hosted open-source solutions are optimized for throughput and are most cost-effective at scale.
  + **Low Volume / Real-Time:** Managed cloud APIs provide a scalable, low-latency solution without the overhead of infrastructure management, making them ideal for interactive applications.
* **Technical Expertise & Resources:** The available in-house talent shapes the build-vs-buy decision.
  + **Limited ML/DevOps Team:** Managed cloud APIs (Textract, Document AI) offer the fastest path to production with the lowest maintenance burden.
  + **Strong ML Engineering Team:** Open-source engines (Tesseract, PaddleOCR) and frameworks (Docling, NeMo Retriever) provide maximum control, customizability, and the potential for lower long-term costs.
* **Data Privacy & Security:** Regulatory and compliance requirements can be a deciding factor.
  + **Highly Sensitive Data (e.g., PII, PHI):** May necessitate on-premise or Virtual Private Cloud (VPC) deployments of open-source or enterprise frameworks to ensure data never leaves a secure perimeter.
* **Total Cost of Ownership (TCO):** A holistic financial analysis is essential. This must balance the predictable, recurring costs of API calls against the upfront and ongoing costs of infrastructure, development, and maintenance for self-hosted solutions, while also factoring in the business cost of uncorrected extraction errors.

### Future Trajectories (2026 and Beyond)

The field of document intelligence continues to advance at a rapid pace. Several key trends are poised to reshape the landscape in the coming years.

* **Self-Supervised and Unsupervised Learning:** A major research trend is the development of models that can be pre-trained on vast quantities of *unlabeled* documents.4 Techniques like masked image modeling and contrastive learning allow models to learn the underlying structure and semantics of documents without requiring costly, human-generated annotations. This will dramatically reduce the barrier to entry for creating highly accurate, domain-specific models, democratizing access to state-of-the-art document AI.
* **Agentic Workflows:** The next frontier extends beyond simple information extraction to multi-step, autonomous reasoning and action. AI agents, powered by advanced language models, will be capable of executing complex tasks involving documents. For example, an "accounts payable agent" could be instructed to: "Review all incoming vendor contracts, extract the payment terms and liability clauses, compare them against our corporate policy, flag any non-compliant clauses, draft a response email to the vendor requesting clarification, and schedule a follow-up for legal review." This represents a shift from data extraction to automated knowledge work.
* **The Path to 99.9% Accuracy:** Industry projections indicate that the current trajectory of improvement will lead to significant milestones by 2027. Near-perfect (99.9%) accuracy for printed text is expected to become the standard. Handwriting recognition is projected to reach 90-95% accuracy for most styles, achieving parity with human performance. Furthermore, specialized domain models, extensively trained on data from fields like law, medicine, and finance, will achieve expert-level performance in their narrow applications. This convergence of capabilities will make fully autonomous, end-to-end document processing a practical reality for a wide range of mainstream business functions, fundamentally altering the nature of knowledge work.10

#### Works cited

1. Best OCR Software in 2025 — A Tool Comparison & Evaluation Guide - Unstract, accessed August 26, 2025, <https://unstract.com/blog/best-pdf-ocr-software/>
2. Document Data Extraction in 2025: LLMs vs OCRs - Vellum AI, accessed August 26, 2025, <https://www.vellum.ai/blog/document-data-extraction-in-2025-llms-vs-ocrs>
3. OCR in 2025: How Intelligent OCR Turns Documents into Data (Use Cases, Tools, and Best Practices) - - BIX Tech, accessed August 26, 2025, <https://bix-tech.com/ocr-in-2025-how-intelligent-ocr-turns-documents-into-data-use-cases-tools-and-best-practices/>
4. What Makes OCR Different in 2025? Impact of Multimodal LLMs and AI Trends - Pixno, accessed August 26, 2025, <https://photes.io/blog/posts/ocr-research-trend>
5. The 2025 Guide to Document Data Extraction using AI, accessed August 26, 2025, <https://www.cradl.ai/post/document-data-extraction-using-ai>
6. Intelligently Extract Text & Data with OCR - Amazon Textract, accessed August 26, 2025, <https://aws.amazon.com/textract/>
7. Automated Data Extraction: The Complete Guide for 2025 - Solvexia, accessed August 26, 2025, <https://www.solvexia.com/blog/automated-data-extraction>
8. Approaches to PDF Data Extraction for Information Retrieval | NVIDIA Technical Blog, accessed August 26, 2025, <https://developer.nvidia.com/blog/approaches-to-pdf-data-extraction-for-information-retrieval/>
9. NVIDIA NeMo Retriever - NVIDIA Developer, accessed August 26, 2025, <https://developer.nvidia.com/nemo-retriever>
10. The Definitive Guide to OCR Accuracy: Benchmarks and Best Practices for 2025 - Medium, accessed August 26, 2025, [https://medium.com/@sanjeeva.bora/the-definitive-guide-to-ocr-accuracy-benchmarks-and-best-practices-for-2025-8116609655da](https://medium.com/%40sanjeeva.bora/the-definitive-guide-to-ocr-accuracy-benchmarks-and-best-practices-for-2025-8116609655da)
11. Vision Capabilities | How-to guides - Llama, accessed August 26, 2025, <https://www.llama.com/docs/how-to-guides/vision-capabilities/>
12. NeMo Retriever - NVIDIA Documentation, accessed August 26, 2025, <https://docs.nvidia.com/nemo/retriever/index.html>
13. Amazon Textract features, accessed August 26, 2025, <https://aws.amazon.com/textract/features/>
14. Custom Queries tutorial - Amazon Textract - AWS Documentation, accessed August 26, 2025, <https://docs.aws.amazon.com/textract/latest/dg/textract-adapters-tutorial.html>
15. Best practices for Amazon Textract Custom Queries - AWS Documentation, accessed August 26, 2025, <https://docs.aws.amazon.com/textract/latest/dg/best-practices-adapters.html>
16. Custom document models - Document Intelligence - Azure AI ..., accessed August 26, 2025, <https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/train/custom-model?view=doc-intel-4.0.0>
17. Document layout analysis - Document Intelligence - Azure AI services | Microsoft Learn, accessed August 26, 2025, <https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/layout?view=doc-intel-4.0.0>
18. Document Processing Models - Document Intelligence - Azure AI services | Microsoft Learn, accessed August 26, 2025, <https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/model-overview?view=doc-intel-4.0.0>
19. Fine-tune models with Azure AI Foundry - Microsoft Community, accessed August 26, 2025, <https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/fine-tuning-overview>
20. Customize a model with Azure OpenAI in Azure AI Foundry Models, accessed August 26, 2025, <https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/fine-tuning>
21. Custom Document Extraction with Document AI Workbench | Google Cloud Skills Boost, accessed August 26, 2025, <https://www.cloudskillsboost.google/focuses/67855?parent=catalog>
22. Custom extractor mechanisms | Document AI | Google Cloud, accessed August 26, 2025, <https://cloud.google.com/document-ai/docs/ce-mechanisms>
23. Document AI Workbench - Custom Document Extractor - Google Codelabs, accessed August 26, 2025, <https://codelabs.developers.google.com/codelabs/docai-custom>
24. Create, use, and manage a custom document classifier | Document AI - Google Cloud, accessed August 26, 2025, <https://cloud.google.com/document-ai/docs/custom-classifier>
25. Best OCR Software 2025: 10 Top Tools Compared and Ranked - Medium, accessed August 26, 2025, [https://medium.com/@Klippa/best-ocr-software-2025-10-top-tools-compared-and-ranked-5cbf011a42a2](https://medium.com/%40Klippa/best-ocr-software-2025-10-top-tools-compared-and-ranked-5cbf011a42a2)
26. Top 5 Open-Source OCR Software Picks for 2025 - Wondershare EdrawMind, accessed August 26, 2025, <https://edrawmind.wondershare.com/office-software/top-5-open-source-ocr-software.html>
27. Python OCR Tutorial: Tesseract, Pytesseract, and OpenCV - Nanonets, accessed August 26, 2025, <https://nanonets.com/blog/ocr-with-tesseract/>
28. PaddleOCR 3.0 Technical Report - arXiv, accessed August 26, 2025, <https://arxiv.org/html/2507.05595v1>
29. (PDF) PaddleOCR 3.0 Technical Report - ResearchGate, accessed August 26, 2025, <https://www.researchgate.net/publication/393511573_PaddleOCR_30_Technical_Report>
30. Turn Complex Documents into Usable Data with VLM, NVIDIA NeMo ..., accessed August 26, 2025, <https://developer.nvidia.com/blog/turn-complex-documents-into-usable-data-with-vlm-nvidia-nemo-retriever-parse/>
31. NVIDIA/nv-ingest: NeMo Retriever extraction is a scalable, performance-oriented document content and metadata extraction microservice. NeMo Retriever extraction uses specialized NVIDIA NIM microservices to find, contextualize, and extract text, tables, charts and images that you can use in downstream generative applications. - GitHub, accessed August 26, 2025, <https://github.com/NVIDIA/nv-ingest>
32. Docling — Overview. Docling: Simplified Document Processing ..., accessed August 26, 2025, [https://medium.com/@hari.haran849/docling-overview-b456139f3d04](https://medium.com/%40hari.haran849/docling-overview-b456139f3d04)
33. Docling Project - GitHub, accessed August 26, 2025, <https://github.com/docling-project>
34. Docling: An Efficient Open-Source Toolkit for AI-driven Document Conversion for AAAI 2025, accessed August 26, 2025, <https://research.ibm.com/publications/docling-an-efficient-open-source-toolkit-for-ai-driven-document-conversion>
35. Docling: Make your Documents Gen AI-ready - GeeksforGeeks, accessed August 26, 2025, <https://www.geeksforgeeks.org/data-science/docling-make-your-documents-gen-ai-ready/>
36. PDF to Markdown with VLMs: LLama 3.2 Vision and getomni-ai/zerox - Medium, accessed August 26, 2025, [https://medium.com/@giacomo\_\_95/vlms-for-pdf-to-markdown-table-conversion-llama-3-2-vision-llava-and-getomni-ai-zerox-00b167cbe6a1](https://medium.com/%40giacomo__95/vlms-for-pdf-to-markdown-table-conversion-llama-3-2-vision-llava-and-getomni-ai-zerox-00b167cbe6a1)
37. lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k · Hugging Face, accessed August 26, 2025, <https://huggingface.co/lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k>
38. Updated 2025 Review: My notes on the best OCR for handwriting recognition and text extraction : r/computervision - Reddit, accessed August 26, 2025, <https://www.reddit.com/r/computervision/comments/1mbpab3/updated_2025_review_my_notes_on_the_best_ocr_for/>
39. 2025 Guide to OCR Accuracy: Choosing the Right API for Your Business - Mindee, accessed August 26, 2025, <https://www.mindee.com/blog/ocr-accuracy-choosing-right-api>
40. OCR Evaluation Module | Clarifai Docs, accessed August 26, 2025, <https://docs.clarifai.com/create/modules/examples/ocr-evaluation/>
41. opendatalab/OmniDocBench: [CVPR 2025] A Comprehensive Benchmark for Document Parsing and Evaluation - GitHub, accessed August 26, 2025, <https://github.com/opendatalab/OmniDocBench>
42. OCR Benchmark: Text Extraction / Capture Accuracy [2025] - Research AIMultiple, accessed August 26, 2025, <https://research.aimultiple.com/ocr-accuracy/>
43. OCR Accuracy Benchmarks: The 2025 Digital Transformation Revolution - VAO Labs, accessed August 26, 2025, [https://www.vao.world/blogs/OCR-Accuracy-Benchmarks:-The-2025-Digital-Transformation-Revolution](https://www.vao.world/blogs/OCR-Accuracy-Benchmarks%3A-The-2025-Digital-Transformation-Revolution)
44. PDF Data Extraction Benchmark 2025: Comparing Docling, Unstructured, and LlamaParse for Document Processing Pipelines - Procycons, accessed August 26, 2025, <https://procycons.com/en/blogs/pdf-data-extraction-benchmark/>
45. Amazon Textract pricing - AWS, accessed August 26, 2025, <https://aws.amazon.com/textract/pricing/>
46. Document AI pricing - Google Cloud, accessed August 26, 2025, <https://cloud.google.com/document-ai/pricing>
47. What's the pricing for reading an engineering spec document using Document Intelligence AI - Microsoft Community, accessed August 26, 2025, <https://learn.microsoft.com/en-us/answers/questions/1512278/whats-the-pricing-for-reading-an-engineering-spec>
48. Best data extraction tools for 2025 | Parseur®, accessed August 26, 2025, <https://parseur.com/blog/best-data-extraction-tools>
49. aws-samples/amazon-textract-transformer-pipeline: Post-process Amazon Textract results with Hugging Face transformer models for document understanding - GitHub, accessed August 26, 2025, <https://github.com/aws-samples/amazon-textract-transformer-pipeline>
50. Real-Time vs. Batch Data Processing: When speed matters, accessed August 26, 2025, <https://journalwjarr.com/sites/default/files/fulltext_pdf/WJARR-2025-1213.pdf>
51. (PDF) Real-time vs. Batch Processing - ResearchGate, accessed August 26, 2025, <https://www.researchgate.net/publication/391593711_Real-time_vs_Batch_Processing>
52. Batch OCR Processing for Large Document Collections - DEV Community, accessed August 26, 2025, <https://dev.to/revisepdf/batch-ocr-processing-for-large-document-collections-4h30>
53. How to train Tesseract 4.00 | tessdoc - GitHub Pages, accessed August 26, 2025, <https://tesseract-ocr.github.io/tessdoc/tess4/TrainingTesseract-4.00.html>
54. Fine-tuning Tesseract's OCR (with some help from R) - Andrés Cruz, accessed August 26, 2025, <https://arcruz0.github.io/posts/finetuning-tess/>
55. Matleo/Tesseract\_fine\_tuning\_training - GitHub, accessed August 26, 2025, <https://github.com/Matleo/Tesseract_fine_tuning_training>
56. Fine-Tuning PaddleOCR's Recognition Model For Dummies by A Dummy, accessed August 26, 2025, <https://anushsom.medium.com/finetuning-paddleocrs-recognition-model-for-dummies-by-a-dummy-89ac7d7edcf6>
57. OCR Fine-Tuning: From Raw Data to Custom Paddle OCR Model | HackerNoon, accessed August 26, 2025, <https://hackernoon.com/ocr-fine-tuning-from-raw-data-to-custom-paddle-ocr-model>
58. Fine Tuning Detection on Custom Dataset · PaddlePaddle PaddleOCR · Discussion #14602, accessed August 26, 2025, <https://github.com/PaddlePaddle/PaddleOCR/discussions/14602>
59. fine-tuning Azure OpenAI service models with Weights & Biases - YouTube, accessed August 26, 2025, <https://www.youtube.com/watch?v=N1CI8Ld0-PA>
