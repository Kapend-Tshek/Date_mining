from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def verifier_documents(documents):
    """Vérifie si les documents sont valides."""
    if not isinstance(documents, list) or not all(isinstance(doc, str) for doc in documents):
        raise ValueError("Documents doivent être une liste de chaînes de caractères")

def decouper_binaire(documents):
    """Découpe les documents avec une pondération binaire."""
    verifier_documents(documents)
    vectoriseur = CountVectorizer(binary=True)
    return vectoriseur.fit_transform(documents).toarray()


def decouper_frequence(documents):
    """Découpe les documents avec une pondération fréquentielle."""
    verifier_documents(documents)
    vectoriseur = CountVectorizer(binary=False)
    return vectoriseur.fit_transform(documents).toarray()


def decouper_tfidf(documents):
    """Découpe les documents avec une pondération TF*IDF."""
    verifier_documents(documents)
    vectoriseur = TfidfVectorizer()
    return vectoriseur.fit_transform(documents).toarray()
