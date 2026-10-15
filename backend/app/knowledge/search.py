"""Местный поиск похожих описаний через TF-IDF."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.storage.models import KnowledgeArticle
from app.knowledge.seed import LEGACY_DEMO_IDS



def article_payload(art):
    return {
        "id": art.id, "crop": art.crop, "category": art.category,
        "title": art.title, "growth_stages": art.growth_stages,
        "symptoms_text": art.symptoms_text, "recommendations": art.recommendations,
        "source": {"title": art.source_title, "section": art.source_section,
                   "url_or_code": art.source_code},
    }


class KnowledgeSearch:
    def retrieve(self, db, crop, growth_stage, query_symptoms, top_k=3):
        """Ранжируем карточки по сходству текста, а не по вероятности болезни."""
        if crop != "Картофель" or not (query_symptoms or "").strip() or top_k <= 0:
            return []
        articles = active_articles(db).all()
        if not articles:
            return []
        texts = [f"{a.title}. {a.symptoms_text}".lower().replace("ё", "е") for a in articles]
        # Фрагменты слов помогают сопоставлять разные окончания.
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        vectors = vectorizer.fit_transform(texts)
        query = vectorizer.transform([query_symptoms.lower().replace("ё", "е")])
        # Чем выше сходство, тем раньше карточка попадёт в выдачу.
        scores = cosine_similarity(query, vectors)[0]
        ranked = sorted(zip(articles, scores), key=lambda pair: pair[1], reverse=True)
        return [dict(article_payload(a), similarity_score=float(score))
                for a, score in ranked[:top_k] if score > 0]


def active_articles(db):
    return db.query(KnowledgeArticle).filter(
        KnowledgeArticle.crop == "Картофель", KnowledgeArticle.id.notin_(LEGACY_DEMO_IDS))
