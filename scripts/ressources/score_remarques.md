Accuracy — "Combien de fois j'ai eu raison en tout"
Sur 150 articles, si le modèle en classe correctement 86 → Accuracy = 86/150 = 57.3%.
Problème : si 80% de tes articles sont NEUTRAL, un modèle qui répond toujours NEUTRAL aura 80% d'accuracy sans rien avoir appris. C'est pourquoi l'accuracy seule est trompeuse sur des datasets déséquilibrés comme le tien (NEGATIVE 44.7%, NEUTRAL 43.8%, POSITIVE 11.6%).

Precision — "Quand je dis POSITIVE, j'ai raison à quelle fréquence ?"
Sur tous les articles que le modèle a classés POSITIVE, combien étaient vraiment POSITIVE. Precision haute = peu de fausses alertes. Precision basse = le modèle crie POSITIVE trop souvent pour rien.

Recall — "Parmi tous les vrais POSITIVE, combien j'en ai trouvé ?"
Sur tous les articles qui sont vraiment POSITIVE, combien le modèle en a détecté. Recall bas = le modèle rate beaucoup de cas réels.

F1 Macro — "La moyenne équilibrée entre precision et recall, sur toutes les classes"
C'est la moyenne du F1 de chaque classe (POSITIVE, NEUTRAL, NEGATIVE) traités avec le même poids, même si une classe est rare. C'est la métrique la plus fiable dans ton cas.

Ce que ça explique dans tes résultats
distilcamembert-base-nli en FR :
Accuracy 46.7% | Precision 15.6% | Recall 33.3% | F1 21.2%
Le modèle prédit presque toujours NEUTRAL. Il tombe juste par hasard sur les articles vraiment neutres (d'où l'accuracy acceptable), mais échoue complètement sur POSITIVE et NEGATIVE (d'où precision/F1 effondrés).
nli-deberta-v3-large vs bart-large-mnli en Topic :
nli-deberta : Accuracy 52.7% | F1 34.4%
bart-large  : Accuracy 48.7% | F1 45.2%
nli-deberta est meilleur sur les thèmes dominants (Conflict, Politics) mais rate les thèmes rares. bart-large est plus équilibré sur toutes les 18 classes — donc F1 macro plus élevé même avec une accuracy inférieure. Pour le fine-tuning, bart-large est le meilleur point de départ malgré son accuracy inférieure.