# CSDI-11048557-AM
# README: KGTablePairDataset and DatasetProcessFramework

This README provides a detailed overview of the various files and directories included in the **KGTablePairDataset** and **DatasetProcessFramework** used in this project. Each component plays a vital role in the data processing and analysis workflow, and is outlined as follows:

## 1. KGTablePairDataset

- **KGTablePairDataset/KG_203_csv**:  
  This folder contains the Knowledge Graph portion of the KG-Table Pair Dataset. It holds CSV files representing the KG data for each pair.

- **KGTablePairDataset/Table_203_csv**:  
  This folder includes the Table portion of the KG-Table Pair Dataset, where each file corresponds to the tabular data associated with the KG data.

- **KGTablePairDataset/KG_Construction.py**:  
  This Python script is responsible for processing the KG-Table Pair Dataset. It handles the extraction and transformation of knowledge from the KG and Table components.

## 2. DatasetProcessFramework

- **DatasetProcessFramework/SourceData**:  
  The `SourceData` folder contains the raw dataset used for KG-Table Pair Dataset creation. It serves as the foundation for further data manipulation and analysis.

- **DatasetProcessFramework/NL Database**:  
  The `NL Database` consists of natural language representations generated from the KG and Table data in each pair. These natural language texts are derived from converting the structured KG and Table data into descriptive sentences.

- **DatasetProcessFramework/Answer**:  
  The `Answer` folder holds the answers corresponding to each KG-Table pair. It provides a structured response set for each pair in the dataset.

- **DatasetProcessFramework/DatasetProcessFramework.py**:  
  This Python file serves as the main script for processing the dataset. It includes the logic required to manage and transform the dataset, ensuring it is ready for analysis.

- **DatasetProcessFramework/question_set.csv**:  
  This CSV file contains all the questions related to the dataset. Each question is designed to evaluate knowledge extracted from the KG and Table pairs.

- **DatasetProcessFramework/full_answer_analysis**:  
  This document provides a comprehensive analysis of all answers generated from the dataset. It evaluates the correctness, quality, and coverage of the responses in relation to the dataset questions.

---
