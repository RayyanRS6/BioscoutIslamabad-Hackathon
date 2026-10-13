# BioScout Islamabad

BioScout Islamabad is an AI-powered web platform for community-driven biodiversity monitoring in Islamabad and surrounding areas. The platform enables users to submit biodiversity observations, view species data, and get AI-assisted information about local biodiversity.

## Features

- **Community Biodiversity Observation Hub**
  - Submit observations with species details, location, and images
  - AI-assisted species identification from uploaded images
  - View observations on an interactive map
  - Filter and search through submitted observations

- **RAG-Enhanced Biodiversity Q&A System**
  - Ask questions about local biodiversity
  - Get AI-powered responses based on knowledge base and community observations
  - View relevant context and sources

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd bioscout_islamabad
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

The application will be available at `http://localhost:8501`

## Project Structure

- `app.py`: Main Streamlit application
- `observations.csv`: Database of biodiversity observations
- `kb_docs/`: Knowledge base documents about local biodiversity
- `uploaded_images/`: Directory for storing uploaded observation images
- `requirements.txt`: Python dependencies

## Knowledge Base

The knowledge base includes information about:
- Margalla Hills National Park biodiversity
- Rawal Lake ecosystem
- Urban biodiversity in Islamabad

## Contributing

1. Submit observations through the web interface
2. Add to the knowledge base by creating new markdown files in the `kb_docs/` directory
3. Report issues and suggest improvements

## Acknowledgments

- Built for the BioScout Islamabad hackathon
- Uses open-source AI models and libraries
- Inspired by citizen science platforms like iNaturalist 