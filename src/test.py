from src.sentiment_analysis import LLMSentiment

tests = [
    # =========================
    # Arabic - harder edge cases
    # =========================
    ("ar", "هل تحسن الاقتصاد هذا العام؟"),
    ("ar", "هل ارتفعت البطالة هذا العام؟"),
    ("ar", "متى يبدأ تنفيذ الاتفاق؟"),
    ("ar", "لماذا ارتفعت الأسعار؟"),
    ("ar", "أعلنت الحكومة خفض أسعار الوقود."),
    ("ar", "أعلنت الحكومة زيادة الضرائب."),
    ("ar", "تراجع عدد الإصابات هذا الشهر."),
    ("ar", "ارتفع عد�� الضحايا بعد القصف."),
    ("ar", "تم التوصل إلى وقف إطلاق النار."),
    ("ar", "استمرار التوتر رغم الاتفاق."),
    ("ar", "ناقش الوزراء خطة الإصلاح الاقتصادي."),
    ("ar", "انخفض التضخم خلال الربع الأخير."),
    ("ar", "فرضت الدولة عقوبات جديدة."),
    ("ar", "حذرت السلطات من عاصفة قوية."),
    ("ar", "افتتاح مستشفى جديد في المدينة."),

    # =========================
    # English - harder edge cases
    # =========================
    ("en", "Has the economy improved this year?"),
    ("en", "Did unemployment rise this year?"),
    ("en", "When will the agreement be implemented?"),
    ("en", "Why did prices increase?"),
    ("en", "The government announced fuel price cuts."),
    ("en", "The government announced tax increases."),
    ("en", "The number of infections declined this month."),
    ("en", "The death toll rose after the attack."),
    ("en", "A ceasefire agreement was reached."),
    ("en", "Tensions continue despite the agreement."),
    ("en", "Ministers discussed the economic reform plan."),
    ("en", "Inflation declined in the last quarter."),
    ("en", "The state imposed new sanctions."),
    ("en", "Authorities warned of a strong storm."),
    ("en", "A new hospital was inaugurated in the city."),

    # =========================
    # French - harder edge cases
    # =========================
    ("fr", "L'économie s'est-elle améliorée cette année ?"),
    ("fr", "Le chômage a-t-il augmenté cette année ?"),
    ("fr", "Quand l'accord sera-t-il appliqué ?"),
    ("fr", "Pourquoi les prix ont-ils augmenté ?"),
    ("fr", "Le gouvernement a annoncé une baisse des prix du carburant."),
    ("fr", "Le gouvernement a annoncé une hausse des impôts."),
    ("fr", "Le nombre d'infections a diminué ce mois-ci."),
    ("fr", "Le bilan des victimes a augmenté après l'attaque."),
    ("fr", "Un accord de cessez-le-feu a été conclu."),
    ("fr", "Les tensions persistent malgré l'accord."),
    ("fr", "Les ministres ont discuté du plan de réforme économique."),
    ("fr", "L'inflation a diminué au dernier trimestre."),
    ("fr", "L'État a imposé de nouvelles sanctions."),
    ("fr", "Les autorités ont averti d'une forte tempête."),
    ("fr", "Un nouvel hôpital a été inauguré dans la ville."),
]

analyzer = LLMSentiment()

for i, (lang, text) in enumerate(tests, 1):
    result = analyzer.predict(text, lang=lang)
    print(f"\n[{i}] LANG={lang}")
    print("TEXT :", text)
    print("LABEL:", result.label)
    print("SCORE:", result.score)
    print("PROBS:", result.probs)