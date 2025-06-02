# Weight-Loss Drug Recommendation Agent Project README

## 1. Project Overview

This project aims to build a weight-loss drug recommendation agent. It collects, processes, and analyzes user reviews from websites like WebMD and official drug prescribing information. By leveraging Natural Language Processing (NLP) techniques, data embedding, and knowledge graphs, it provides users with personalized weight-loss drug consultations and recommendations. The project includes modules for data scraping, data extraction, data standardization, text embedding, and a chatbot based on Retrieval Augmented Generation (RAG).

## 2. Project Goals

* Scrape user reviews and drug information related to weight-loss drugs from the internet.
* Extract key information from unstructured text, such as drug efficacy, side effects, and user satisfaction.
* Standardize the extracted data for subsequent analysis.
* Utilize embedding techniques to convert textual information into vector representations for similarity calculation and model training.
* Build an intelligent chatbot capable of understanding user queries and providing recommendations and answers by combining processed data and the drug knowledge base.

## 3. Directory Structure


Weight-Loss-Drug-Recommendation-Agent/
├── code_chatbot/                   # Chatbot-related code
│   ├── chatbot.py
│   ├── graph_loader.py
│   └── table_loader.py
├── code_embedding/                 # Text embedding related code
│   ├── embedding_ae.ipynb
│   ├── embedding_pi_heavy.py
│   └── embedding_pi_light.py
├── code_extraction/                # Data extraction related code
│   ├── extract_all.py
│   ├── prompt_extraction.py
│   └── schema_extraction.py
├── code_scraping/                  # Data scraping related code
│   ├── batch_scraper.py
│   └── scraper.py
├── code_standardization/           # Data standardization related code
│   ├── environment_standardization.txt
│   ├── standardization.py
│   ├── standardization_raw.ipynb
│   └── standardize_all.py
├── data_embedded/                  # Stores embedded data
│   └── embedded_ae.csv
├── data_extracted/                 # Stores raw extracted data
│   ├── extracted_reviews_all.csv
│   ├── extracted_reviews_raw10.csv
│   └── extracted_reviews_top10.csv
├── data_literature/                # Reference literature
│   ├── ReAct.pdf
│   └── TableRAG.pdf
├── data_prescribing_information/   # Official drug prescribing information
│   ├── Prescribing Information Mounjaro.pdf
│   ├── Prescribing Information Ozempic.pdf
│   ├── Prescribing Information Rybelsus.pdf
│   ├── Prescribing Information Saxenda.pdf
│   ├── Prescribing Information Victoza.pdf
│   ├── Prescribing Information Wegovy.pdf
│   └── Prescribing Information Zepbound.pdf
├── data_standardized/              # Stores standardized data
│   ├── standardized_reviews_all.csv
│   └── standardized_reviews_top10.csv
└── data_webmd/                     # Raw review data scraped from WebMD
├── webmd_all_reviews.csv
├── webmd_mounjaro_reviews.csv
├── webmd_ozempic_reviews.csv
├── webmd_rybelsus_reviews.csv
├── webmd_saxenda_reviews.csv
├── webmd_victoza_reviews.csv
├── webmd_wegovy_reviews.csv
└── webmd_zepbound_reviews.csv


## 4. File Detailed Description

### 4.1. `code_chatbot/` - Chatbot Code

* **`chatbot.py`**: Implements the core logic of the chatbot. May include user intent recognition, dialogue management, and calling the knowledge base for information retrieval and response generation.
* **`graph_loader.py`**: Used to load and query knowledge graph data. The knowledge graph might store relationships between drugs, indications, and side effects.
* **`table_loader.py`**: Used to load and query tabular data, such as reading drug reviews and efficacy data from CSV files for the chatbot to retrieve.

### 4.2. `code_embedding/` - Text Embedding Code

* **`embedding_ae.ipynb`**: Jupyter Notebook file for exploring and implementing methods to embed text data (possibly user reviews or drug descriptions) using an Autoencoder (AE).
* **`embedding_pi_heavy.py`**: Python script for "heavy" embedding of Prescribing Information (PI). This might mean using more complex models or deeper text analysis to generate high-quality embedding vectors.
* **`embedding_pi_light.py`**: Python script for "light" embedding of Prescribing Information. It might use simpler or faster embedding methods, suitable for rapid prototyping or resource-constrained scenarios.

### 4.3. `code_extraction/` - Data Extraction Code

* **`extract_all.py`**: Main execution script that calls other extraction modules to complete all types of data extraction tasks.
* **`prompt_extraction.py`**: Utilizes predefined prompts and Large Language Models (LLMs) to extract structured information from text. For example, extracting drug names, mentioned side effects, and efficacy ratings from user reviews.
* **`schema_extraction.py`**: Defines or infers the schema used for information extraction from data. Specifies which fields to extract and their data types and formats.

### 4.4. `code_scraping/` - Data Scraping Code

* **`batch_scraper.py`**: Implements batch data scraping functionality, allowing data to be scraped from multiple URLs or pages at once, improving efficiency.
* **`scraper.py`**: Contains the core web scraping logic, such as sending HTTP requests, parsing HTML content, and extracting required data. It might be a scraper customized for specific websites (like WebMD).

### 4.5. `code_standardization/` - Data Standardization Code

* **`environment_standardization.txt`**: The project's Python environment dependency list, usually a `requirements.txt` file, recording the libraries and their versions required for the project to run, ensuring consistency in the development environment.
* **`standardization.py`**: Script implementing data standardization processing. May include data cleaning (removing irrelevant characters, handling missing values), format unification (date, numerical formats), and term standardization (mapping synonyms to unified terms, e.g., side effect names).
* **`standardization_raw.ipynb`**: Jupyter Notebook file for exploring and experimenting with methods and processes for standardizing raw data.
* **`standardize_all.py`**: Main execution script that calls functions from `standardization.py` to standardize all scraped and extracted data.

### 4.6. `data_embedded/` - Embedded Data

* **`embedded_ae.csv`**: Stores text embedding vectors generated by `embedding_ae.ipynb` or related scripts. Each row might represent a text segment (like a review) and its corresponding vector representation.

### 4.7. `data_extracted/` - Raw Extracted Data

These CSV files store structured information extracted from raw data sources (like scraped reviews) but have not yet been fully standardized.

* **`extracted_reviews_all.csv`**: Contains all user reviews with extracted information.
* **`extracted_reviews_raw10.csv`**: Possibly a small sample file containing 10 raw extracted reviews, for quick viewing or debugging.
* **`extracted_reviews_top10.csv`**: Possibly the top 10 extracted reviews filtered by some criteria (e.g., rating, usefulness).

### 4.8. `data_literature/` - Reference Literature

* **`ReAct.pdf`**: A paper on the ReAct (Reasoning and Acting) framework. This framework might be used to build more powerful agents (like the chatbot) capable of complex reasoning and executing actions.
* **`TableRAG.pdf`**: A paper on applying Retrieval Augmented Generation (RAG) techniques to tabular data. This suggests the project might use RAG methods to enable the chatbot to query and utilize tabular data (like drug information tables, review statistics tables).

### 4.9. `data_prescribing_information/` - Official Drug Prescribing Information

This directory stores official PDF format prescribing information for various weight-loss drugs. These files are important sources for obtaining accurate drug information (such as indications, contraindications, side effects, dosage).

* `Prescribing Information Mounjaro.pdf`
* `Prescribing Information Ozempic.pdf`
* `Prescribing Information Rybelsus.pdf`
* `Prescribing Information Saxenda.pdf`
* `Prescribing Information Victoza.pdf`
* `Prescribing Information Wegovy.pdf`
* `Prescribing Information Zepbound.pdf`

### 4.10. `data_standardized/` - Standardized Data

These CSV files store data processed by the `code_standardization/` module. The data is cleaner, more organized, and suitable for model training and analysis.

* **`standardized_reviews_all.csv`**: All standardized user review data.
* **`standardized_reviews_top10.csv`**: The top 10 user review data, filtered by some criteria and standardized.

### 4.11. `data_webmd/` - Raw Review Data Scraped from WebMD

This directory stores raw user review data for different weight-loss drugs scraped from the WebMD website. Each CSV file corresponds to reviews for one drug.

* **`webmd_all_reviews.csv`**: A summary file containing all drug reviews scraped from WebMD.
* `webmd_mounjaro_reviews.csv`
* `webmd_ozempic_reviews.csv`
* `webmd_rybelsus_reviews.csv`
* `webmd_saxenda_reviews.csv`
* `webmd_victoza_reviews.csv`
* `webmd_wegovy_reviews.csv`
* `webmd_zepbound_reviews.csv`

## 5. How to Run (Example)

Specific running steps depend on the implementation of each script. Here are some general guidelines:

1.  **Environment Setup**:
    ```bash
    pip install -r code_standardization/environment_standardization.txt
    ```
2.  **Data Scraping**:
    ```bash
    python code_scraping/scraper.py # May need to configure target URLs or drug list
    # OR
    python code_scraping/batch_scraper.py
    ```
3.  **Data Extraction**:
    ```bash
    python code_extraction/extract_all.py # Ensure input file paths are correct
    ```
4.  **Data Standardization**:
    ```bash
    python code_standardization/standardize_all.py # Ensure input file paths are correct
    ```
5.  **Text Embedding**:
    Run the corresponding scripts or Notebooks in `code_embedding/`.
6.  **Chatbot**:
    ```bash
    python code_chatbot/chatbot.py # May need to load embedding files and knowledge base
    ```

**Note**: Please refer to the comments or documentation within each script for more detailed running instructions and parameter configurations.

## 6. Dependencies

Major dependencies are listed in the `code_standardization/environment_standardization.txt` file. Common libraries may include:

* `pandas` for data processing
* `requests`, `beautifulsoup4`, `selenium` for web scraping
* `nltk`, `spacy`, `scikit-learn` for NLP and machine learning
* `pytorch`, `tensorflow`, `transformers` for deep learning and embedding
* `langchain`, `llama-index` (if these frameworks are used) for building RAG and agents
* And other libraries required for specific tasks.

Ensure all dependencies are installed according to the `environment_standardization.txt` file.
