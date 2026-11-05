# INSIGHTPILOT-AI-POWERED-DATA-ANALYTICS-COPILOT
InsightPilot — AI-Powered Data Analytics Copilot
InsightPilot is an interactive Streamlit application that brings dataset exploration, data preparation, machine learning, AI-assisted analysis, and reporting into one workspace.
Features
Data Explorer
- Upload CSV, Excel, or JSON datasets.
- Preview records and inspect dataset structure.
- Review dataset information, column types, missing values, and duplicates.
- Explore data through summaries and visualisations.
Data Preparation
- Inspect and clean data.
- Review missing values and duplicate records.
- Use data-cleaning and feature-engineering workflows before modelling.
Machine Learning Lab
- Train and compare classification or regression models.
- Review model evaluation metrics.
- Explore feature importance where supported by the selected model.
- Select the correct task for the target column:
  - Classification for categories or labels (for example, Titanic Survived).
  - Regression for continuous numerical values.
AI Dataset Copilot
- Ask natural-language questions about the currently uploaded dataset.
- View dataset shape, column information, numerical summaries, missing-value counts, duplicate counts, and sample records.
- Generate dataset explanations using a locally running Ollama model.
Current limitation: The Copilot sends dataset metadata, summaries, and a sample of the first 15 rows to Ollama. It is useful for explanations and questions answerable from that context, but it does not automatically execute arbitrary calculations over every row. Use the analytics and data-analysis tools for exact full-dataset calculations.
Reports & Export
- Review analysis outputs and export supported results from the application.
Tech Stack
- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn (machine-learning workflows)
- Ollama (local LLM for the Dataset Copilot)
Getting Started
Prerequisites
- Python 3.10+ recommended
- pip
- Ollama (only if you want to use the AI Dataset Copilot)
. Clone or download the project
git clone <your-repository-url>
cd <project-folder>
If you downloaded the source files directly, open a terminal in the project folder instead.
2. Create a virtual environment (recommended)
Windows
python -m venv .venv
.venv\\Scripts\\activate
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
If a requirements.txt file is included:
pip install -r requirements.txt
Otherwise, install the packages used by the app (adjust to match your source code):
pip install streamlit pandas numpy matplotlib scikit-learn openpyxl
4. Run InsightPilot
streamlit run app.py
Streamlit will display a local URL in the terminal, usually http://localhost:8501. Open it in your browser.
Using the AI Dataset Copilot (Ollama)
1. Install Ollama from ollama.com.
2. Start Ollama and download a model, for example:
   ollama pull llama3.2
3. Ensure Ollama is running at http://localhost:11434.
4. Start InsightPilot and upload a dataset.
5. Open Dataset Copilot (Ask about data).
6. Enter the model name (for example, llama3.2) and ask a question.
If the Copilot cannot connect, check that Ollama is running and that the model name is installed locally.
Example Dataset: Titanic
Try the Titanic dataset and ask questions such as:
- What does each column represent?
- Which columns contain missing values?
- Summarise the passenger dataset.
- What patterns can be observed in the sample records?
- Which features could be useful for predicting survival?
For Titanic survival prediction, choose Classification and use Survived as the target column.
Suggested Project Structure
InsightPilot/
├── app.py
├── requirements.txt
├── README.md
└── data/                 # Optional: sample datasets
Your actual project may contain additional modules, folders, and assets.
Troubleshooting
- streamlit command not found: Activate your virtual environment and install Streamlit with pip install streamlit.
- Ollama connection error: Start Ollama and verify the endpoint http://localhost:11434.
- Model not found: Run ollama pull <model-name> and use the same model name in the Copilot.
- Unexpected model metrics: Check that the task type and target column are correct. A binary target such as Survived should normally be treated as classification.
Future Improvements
- Execute safe Pandas operations over the full dataset from natural-language requests.
- Add more model evaluation metrics and visual comparisons.
- Improve automated task detection (classification vs regression).
- Add richer report generation and export options.
Author
Rishyender Reddy
MSc Artificial Intelligence | Teesside University
