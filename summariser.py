import nltk
import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class Summariser:
    def __init__(self):
        # Make sure all the NLTK resources needed are available
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('averaged_perceptron_tagger')

    def clean_text(self, text):
        """Prepare raw text by removing noise like refs, links, and symbols."""
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        
        text = re.sub(r'\[[0-9]*\]', ' ', text)  # remove [1], [23] style refs
        text = re.sub(r'http\S+|www\S+', ' ', text)  # strip URLs
        text = text.lower()  # normalize case
        text = re.sub(r'[^a-z0-9\s]', ' ', text)  # keep only words/numbers
        text = re.sub(r'\s+', ' ', text).strip()  # collapse spaces
        return text

    def create_frequency_table(self, sentences):
        """Turn list of sentences into TF-IDF word weights."""
        if not sentences:
            raise ValueError("No sentences provided for frequency table")
        
        vectorizer = TfidfVectorizer(
            stop_words=stopwords.words('english'),
            tokenizer=word_tokenize,
            lowercase=True
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            # total TF-IDF value of each word across all sentences
            freq_table = dict(zip(feature_names, np.sum(tfidf_matrix.toarray(), axis=0)))
            return freq_table, tfidf_matrix
        except ValueError as e:
            raise ValueError("Error in TF-IDF computation: check input text") from e

    def sentence_tokenize(self, text):
        """Split text into sentences, and clean each for later use."""
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        
        original_sentences = sent_tokenize(text)
        cleaned_sentences = [self.clean_text(sent) for sent in original_sentences]
        return original_sentences, cleaned_sentences

    def score_sentences(self, sentences, freq_table, tfidf_matrix):
        """Assign a score to each sentence using TF-IDF and TextRank."""
        if not sentences or not freq_table:
            raise ValueError("Empty sentences or frequency table")
        
        # First: score sentences by how many strong words they contain (TF-IDF sum)
        sentence_scores = {}
        for idx, sentence in enumerate(sentences):
            sent_tokens = [w for w in word_tokenize(sentence.lower()) if w not in string.punctuation]
            word_count = len(sent_tokens) or 1
            score = sum(freq_table.get(w, 0) for w in sent_tokens) / word_count
            sentence_scores[idx] = score

        # Second: create similarity graph and apply TextRank
        similarity_matrix = cosine_similarity(tfidf_matrix)
        nx_graph = nx.from_numpy_array(similarity_matrix)
        try:
            textrank_scores = nx.pagerank(nx_graph, max_iter=100, tol=1e-6)
        except nx.PowerIterationFailedConvergence:
            # if TextRank fails, fall back on TF-IDF only
            textrank_scores = {i: score for i, score in sentence_scores.items()}
        
        # Mix both methods (weighted average) for final scores
        final_scores = {i: 0.7 * sentence_scores[i] + 0.3 * textrank_scores.get(i, 0)
                       for i in range(len(sentences))}
        return final_scores

    def find_average_score(self, sentence_scores):
        """Get average score of all sentences (used for thresholding)."""
        if not sentence_scores:
            return 0
        return sum(sentence_scores.values()) / len(sentence_scores)

    def generate_summary(self, original_sentences, sentence_scores, threshold, compression_ratio=0.3):
        """Pick top sentences based on score and build the summary."""
        if not original_sentences or not sentence_scores:
            return ""
        
        # Decide number of sentences to keep
        effective_ratio = 1.0 - compression_ratio
        num_sentences = max(1, int(len(original_sentences) * effective_ratio))
        sorted_scores = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, score in sorted_scores[:num_sentences] if score > threshold]
        
        # if nothing passes the threshold, at least keep top few
        if not selected_indices:
            selected_indices = [idx for idx, _ in sorted_scores[:num_sentences]]

        # preserve original sentence order
        selected_indices.sort()
        summary = ' '.join(original_sentences[idx] for idx in selected_indices).strip()
        return summary

    def summarize(self, text, compression_ratio=0.3):
        """Run full pipeline: tokenize → score → select → summarize."""
        try:
            original_sents, cleaned_sents = self.sentence_tokenize(text)
            if not original_sents:
                return ""
            
            freq_table, tfidf_matrix = self.create_frequency_table(cleaned_sents)
            sentence_scores = self.score_sentences(cleaned_sents, freq_table, tfidf_matrix)
            
            # Lower threshold → more sentences are eligible
            threshold = self.find_average_score(sentence_scores) * 0.8
            
            # If text is short, avoid over-compressing
            if len(original_sents) < 10:
                compression_ratio = min(compression_ratio, 0.8)
            
            return self.generate_summary(original_sents, sentence_scores, threshold, compression_ratio)
        except Exception as e:
            raise RuntimeError(f"Error summarizing text: {str(e)}") from e

# Example run
if __name__ == "__main__":
    sample_text = """Artificial Intelligence (AI) is rapidly transforming the way we live and work. From healthcare to finance, AI technologies are enabling faster decision-making, reducing human errors, and automating repetitive tasks. In healthcare, AI helps doctors diagnose diseases with higher accuracy and suggests personalized treatments. In the finance industry, AI-powered tools are used to detect fraudulent transactions and provide better customer services. Despite these benefits, AI also brings challenges. Concerns about job displacement, data privacy, and ethical decision-making are growing worldwide. To ensure AI benefits everyone, governments, companies, and researchers must collaborate to build transparent, fair, and responsible AI systems."""
    summarizer = Summariser()
    summary = summarizer.summarize(sample_text, compression_ratio=0.5)
    print("Summary:", summary)
