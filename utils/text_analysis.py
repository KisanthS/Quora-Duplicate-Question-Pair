def explain_difference(q1, q2, nlp):
    """
    Analyze and return shared, unique keywords from each question using spaCy NLP.
    """
    doc1 = nlp(q1)
    doc2 = nlp(q2)

    tokens1 = set([t.lemma_.lower() for t in doc1 if t.is_alpha and not t.is_stop])
    tokens2 = set([t.lemma_.lower() for t in doc2 if t.is_alpha and not t.is_stop])

    shared = tokens1 & tokens2
    only_q1 = tokens1 - tokens2
    only_q2 = tokens2 - tokens1

    return shared, only_q1, only_q2
