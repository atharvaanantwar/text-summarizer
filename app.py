from flask import Flask, render_template, request
from summariser import Summariser
import re

app = Flask(__name__)
summarizer = Summariser()

# Custom Jinja2 filter for word count
def wordcount(text):
    if not text or not isinstance(text, str):
        return 0
    words = len(re.split(r'\s+', text.strip()))
    return words

app.jinja_env.filters['wordcount'] = wordcount

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form.get('text')
        try:
            compression_ratio = float(request.form.get('compression_ratio', 0.3))
            if not 0.1 <= compression_ratio <= 1.0:
                raise ValueError("Compression ratio must be between 0.1 and 1.0")
            if not text or len(text.strip()) == 0:
                return render_template('index.html', error="Please enter some text.")
            summary = summarizer.summarize(text, compression_ratio)
            return render_template(
                'index.html',
                original_text=text,
                summary=summary,
                compression_ratio=compression_ratio
            )
        except ValueError as e:
            return render_template('index.html', error=str(e))
        except Exception as e:
            return render_template('index.html', error=f"Error processing text: {str(e)}")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)