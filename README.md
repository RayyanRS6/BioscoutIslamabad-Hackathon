# BioScout Islamabad

BioScout Islamabad is an AI-powered web platform for community-driven biodiversity monitoring in Islamabad and surrounding areas. The platform enables users to submit biodiversity observations, view species data, and get AI-assisted information about local biodiversity.

## Features

- **Community Biodiversity Observation Hub**
  - Submit observations with species details, location, coordinates, and a photo
  - AI-assisted species identification: upload a photo and get the closest matches among species recorded around Islamabad, then fill the form with one click
  - View observations on an interactive map
  - Filter and search through submitted observations

- **Biodiversity Q&A over a local knowledge base**
  - Ask questions about local biodiversity, or try one of the example questions
  - Answers show the most relevant knowledge base sections and community observations, with their sources
  - New observations can be searched as soon as they are submitted

## Setup Instructions

Requires Python 3.10 or newer.

1. **Clone the repository**
   ```bash
   git clone https://github.com/RayyanRS6/BioscoutIslamabad-Hackathon.git
   cd BioscoutIslamabad-Hackathon
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

### AI models

The app uses two open-source models from Hugging Face, downloaded automatically the first time they are needed (an internet connection is required for that first download):

- [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (~90 MB) for semantic search in the Q&A tab
- [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) (~600 MB) for species identification from photos

If a model can't be downloaded, the app keeps working: the Q&A falls back to keyword search, and observations can still be submitted without AI suggestions.

## Project Structure

- `app.py`: Main Streamlit application
- `observations.csv`: Database of biodiversity observations
- `kb_docs/`: Knowledge base documents about local biodiversity
- `uploaded_images/`: Directory for storing uploaded observation images
- `.streamlit/config.toml`: Streamlit settings (maximum photo upload size)
- `requirements.txt`: Python dependencies

## Knowledge Base

The knowledge base includes information about:
- Margalla Hills National Park biodiversity
- Rawal Lake ecosystem
- Urban biodiversity in Islamabad

Each markdown section (`##` / `###` heading) is searched separately, so answers point to the specific part of a document. Species listed as `- Common Name (Scientific name)` are also used as candidate labels for AI species identification.

## Contributing

1. Submit observations through the web interface
2. Add to the knowledge base by creating new markdown files in the `kb_docs/` directory
3. Report issues and suggest improvements

## Acknowledgments

- Built for the BioScout Islamabad hackathon
- Uses open-source AI models and libraries
- Inspired by citizen science platforms like iNaturalist
