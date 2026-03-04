If the `architecture_overview.md` file is empty and you need to fill it in with relevant information about the architecture of the **Document Intelligence Refinery** project, here's a suggested outline and content you can use to get started. The goal is to explain the system's high-level design, its components, and how they interact.

---

# **architecture_overview.md**

## **Document Intelligence Refinery Architecture Overview**

The **Document Intelligence Refinery** is an AI-based document extraction system designed to process and extract meaningful information from a variety of document types. This overview provides a high-level explanation of the system architecture, its components, and how the entire extraction pipeline functions.

---

## **1. System Architecture Components**

The system consists of multiple stages, each dedicated to a specific part of the document processing pipeline:

### **1.1 Triage Agent**

* **Function:** The Triage Agent is responsible for profiling documents at the start. It assesses the characteristics of each document, such as:

  * Character density (`char_density`)
  * Image ratio (`image_area_ratio`)
  * Table count and layout type
* **Tools Used:** `pdfplumber`, `PyMuPDF`
* **Output:** `DocumentProfile` — Contains information about the document's layout, origin type (scanned, mixed, or digital), and other relevant metadata.

### **1.2 Document Profile**

* **Function:** Based on the signals from the Triage Agent, the `DocumentProfile` defines the overall characteristics of a document:

  * **Origin Type:** Determines if the document is scanned, native digital, or mixed.
  * **Layout Complexity:** Identifies whether the document has a single column, multiple columns, or heavy tables.
  * **Extraction Strategy:** Decides which extraction strategy (A, B, or C) should be applied to the document.
* **Output:** This profile drives the classification logic that routes the document to the appropriate extraction strategy.

### **1.3 Extraction Strategies**

The core of the extraction process involves three distinct strategies, each tailored for different types of documents. The strategies are executed based on the document's classification in the `DocumentProfile`.

* **Strategy A — Fast Text (pdfplumber)**

  * Used for **native digital documents** that have a simple layout (single column).
  * **Fast and cheap extraction**.
  * **Fallback:** If the extraction confidence is low, the document escalates to **Strategy B**.

* **Strategy B — Layout-Aware (Docling/MinerU)**

  * Used for **mixed documents** with complex layouts, such as multi-column formats or documents with heavy tables.
  * **Medium cost** and more accurate for complex layouts.
  * **Fallback:** If the confidence score on key pages is low, the document escalates to **Strategy C**.

* **Strategy C — Vision (EasyOCR/Chunkr)**

  * Used for **scanned documents** or documents with a high image-to-text ratio.
  * **High cost** and **requires additional resources** like GPU (for more accurate OCR).
  * **Fallback:** For low-confidence pages, the system can use a **Gemini Flash free tier** or other fallback options.

---

## **2. Data Flow in the Pipeline**

The overall pipeline can be visualized as follows:

```mermaid
flowchart TD
    A[Raw Document] --> B[Triage Agent\n(pdfplumber signals)]
    B --> C{DocumentProfile\n(origin + layout)}

    C -->|scanned_by_density OR ghost_text\nOR high_image_thin| F[Strategy C\nVision\nEasyOCR primary\nGemini Flash fallback]
    C -->|mixed + table_heavy\nOR multi_column| E[Strategy B\nLayout-Aware\nDocling]
    C -->|mixed + single_column\nOR native_digital| D[Strategy A\nFast Text\npdfplumber]

    D -->|page confidence < 0.75| E
    E -->|page confidence < 0.65| F

    D -->|page confidence >= 0.75| G[ExtractedDocument\nnormalized schema]
    E -->|page confidence >= 0.65| G
    F --> G

    G --> H[ChunkValidator\nenforce invariants]
    H --> I[Semantic Chunking Engine\nLDU List]
    I --> J[PageIndex Builder\nLLM section summaries]
    I --> K[Vector Store\nChromaDB]
    I --> L[FactTable\nSQLite]
    J --> M[Query Interface Agent\nLangGraph]
    K --> M
    L --> M
    M --> N[Answer + ProvenanceChain\npage + bbox + hash]

    style D fill:#c8f7c5
    style E fill:#fef9c3
    style F fill:#fde8e8
    style H fill:#e8e8ff
```

### **Explanation of the Data Flow:**

1. **Raw Document Input:** The raw document (PDF) is passed through the **Triage Agent**, which analyzes basic signals such as image ratio and text density.
2. **Document Classification:** The system then categorizes the document into one of three origin types (scanned, mixed, or native digital) and selects the appropriate extraction strategy.
3. **Extraction:** Depending on the document's profile, the extraction strategy is applied:

   * **Strategy A** handles fast text extraction.
   * **Strategy B** is used for complex layouts with tables and multiple columns.
   * **Strategy C** is for scanned documents or those with high image content.
4. **Chunking and Indexing:** After extraction, the content is split into **Logical Document Units (LDUs)**, which are then indexed for search and analysis.
5. **Query Interface:** Finally, a query interface is available to interact with the data and retrieve information based on user queries.

---

## **3. Key Features and Metrics**

### **3.1 Failure Modes**

The system monitors for common failure modes, such as:

* **Structure Collapse:** When the document’s layout is poorly preserved (e.g., multi-column layouts flattened into text).
* **Context Poverty:** When important sections, like tables or captions, are incorrectly extracted or lost.
* **Provenance Blindness:** When extracted content cannot be traced back to its source location in the original document (missing bounding boxes or content hashes).

### **3.2 Signals and Metrics**

Each document is analyzed for key signals that determine which strategy to use:

* **Character Density (char_density):** Measures how much text is in a document compared to its physical size.
* **Image Area Ratio (img_ratio):** Measures how much of the document is made up of images versus text.
* **Table Density (avg_tables/page):** Measures the frequency of tables within the document.
* **X-Jump Ratio (x_jump):** Measures the degree of disruption in the reading order for multi-column documents.

### **3.3 Provenance and Validation**

To ensure that the extraction is reliable, each document's content is tracked with a **Provenance Chain**, which includes:

* **Page Number:** Each extracted chunk is associated with its original page number.
* **Bounding Box (bbox):** The exact location of the extracted content on the page.
* **Content Hash:** A unique identifier for the extracted content to ensure consistency and accuracy across runs.

---

## **4. Escalation and Confidence**

The system uses a tiered escalation process:

1. **Document-Level Routing:** Based on initial signals, the document is routed to the most appropriate extraction strategy.
2. **Page-Level Escalation:** If the confidence score for a page is low, the system escalates to a more powerful extraction strategy (e.g., from **Strategy A** to **Strategy B**, or from **Strategy B** to **Strategy C**).

---

## **5. Cost and Resource Management**

The system is designed to be efficient in terms of cost:

* **Strategy A** is **fast and cheap**, using tools like **pdfplumber**.
* **Strategy B** is **medium cost**, requiring tools like **Docling** and **MinerU**.
* **Strategy C** is **expensive**, relying on **EasyOCR** and **Gemini Flash** for scanned documents or low-confidence extractions.

The **escalation policy** ensures that documents are processed using the least expensive strategy unless higher quality is necessary.

---

## **6. Future Improvements**

As you move forward, you may add new document types or refine your existing thresholds. For instance:

* **Testing with new document types** (e.g., newspapers or academic papers) might reveal the need for a **new signal** to detect multi-column layouts.
* **Additional failure modes** might emerge that require adjustments to the extraction logic or strategy routing.

By continuously refining the thresholds and failure detection mechanisms, the system will become more robust and able to handle a wider variety of documents.

---
