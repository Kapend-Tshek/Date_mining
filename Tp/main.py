from Conversion import decouper_binaire, decouper_frequence, decouper_tfidf

# Exemple de documents
document1 = [
    "bayerne est un club de football allemand qui gagne cinq fois la champions league et a perdu huit champions league",
    "Dans le foot moderne gagner la champions league est une priorité",
    "Le mariage est une ceremonie qui a beaucoup de place dans la vie d'un jeune adulte"
    

]

# Utilisation des fonctions
vecteurs_binaires = decouper_binaire(document1)
vecteurs_frequences = decouper_frequence(document1)
vecteurs_tfidf = decouper_tfidf(document1)

# Affichage des résultats
print("Vecteurs binaires:")
for i, doc in enumerate(document1):
    print(f"Document {i + 1}: {doc}")
    print(vecteurs_binaires[i])
    print("=" * 40)

print("\n\nVecteurs fréquentiels:")
for i, doc in enumerate(document1):
    print(f"Document {i + 1}: {doc}")
    print(vecteurs_frequences[i])
    print("=" * 40)

print("\n\nVecteurs TF*IDF:")
for i, doc in enumerate(document1):
    print(f"Document {i + 1}: {doc}")
    print(vecteurs_tfidf[i])
    print("=" * 40)
