__Text Summarizer__
**Overview**

Text Summarizer is a Python web app built with Flask that creates short, clear summaries of long text.
It uses TF-IDF and TextRank algorithms to pick out the most important sentences. You can adjust a compression ratio slider to decide how detailed or concise the summary should be (e.g., 0.1 for longer summaries, 1.0 for very short ones).

**Libraries Used**

    nltk → text preprocessing (tokenization, stopwords)
    scikit-learn → TF-IDF vectorization and cosine similarity
    numpy → numerical operations
    networkx → TextRank graph-based scoring
    flask → web framework for the UI

**Installation**

1. Clone the repository:
    git clone https://github.com/yourusername/text-summarizer.git
    cd text-summarizer
2. Create and activate a virtual environment:
    python -m venv venv
        Windows: venv\Scripts\activate
        macOS/Linux: source venv/bin/activate
3. Install dependencies:
    pip install -r requirements.txt
4. Download NLTK resources:
    python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger

**Usage**

1. Start the Flask app:
    python app.py
2. Open your browser at http://127.0.0.1:5000/
3. Paste text into the input box.
4. Adjust the compression ratio (from detailed to concise).
5. Click “Generate Summary” to see the result.

