import spacy
from spacy.matcher import DependencyMatcher

# Load SpaCy model
nlp = spacy.load("en_core_web_trf")

def extract_entities(text):
    """
    Extract named entities from the text using SpaCy.
    """
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_relationships(text, entities):
    """
    Extract relationships between entities from the text.
    """
    doc = nlp(text)
    relationships = []
    
    for token in doc:
        # Simple example: extract subject-verb-object relationships
        if token.dep_ == "ROOT":  # Main verb in a sentence
            subject = [w.text for w in token.lefts if w.dep_ == "nsubj"]  # Subject
            object_ = [w.text for w in token.rights if w.dep_ == "dobj"]  # Object
            
            if subject and object_:
                relationships.append((subject[0], token.text, object_[0]))

    return relationships

def extract_contextual_numerical_data(text):
    """
    Extract numerical data with contextual information.
    """
    doc = nlp(text)
    numerical_data = []

    for token in doc:
        if token.like_num:  # Identify numbers
            context = " ".join([child.text for child in token.head.children])
            numerical_data.append({
                "Value": token.text,
                "Context": context or "Unknown"
            })

    return numerical_data
