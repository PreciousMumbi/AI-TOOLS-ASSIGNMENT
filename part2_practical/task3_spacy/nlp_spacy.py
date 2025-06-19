import spacy

# Load model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "I love my new Apple iPhone 14. It's better than my old Samsung phone."

# Process text
doc = nlp(text)

# Named Entity Recognition (NER)
for ent in doc.ents:
    print(ent.text, ent.label_)

# Rule-based sentiment
positive_words = ['love', 'excellent', 'good', 'happy', 'better']
negative_words = ['bad', 'worse', 'hate', 'poor']

sentiment_score = sum(1 for token in doc if token.text.lower() in positive_words) - \
                  sum(1 for token in doc if token.text.lower() in negative_words)

print("Sentiment Score:", sentiment_score)
print("Sentiment:", "Positive" if sentiment_score > 0 else "Negative")
