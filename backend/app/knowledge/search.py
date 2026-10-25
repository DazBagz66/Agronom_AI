"""Местный поиск похожих описаний через TF-IDF."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.storage.models import KnowledgeArticle
from app.knowledge.seed import LEGACY_DEMO_IDS
from app.knowledge.corn import corn_scope, corn_article, corn_catalog



def article_payload(art):
    payload = {
        "id": art.id, "crop": art.crop, "category": art.category,
        "title": art.title, "growth_stages": art.growth_stages,
        "symptoms_text": art.symptoms_text, "recommendations": art.recommendations,
        "source": {"title": art.source_title, "section": art.source_section,
                   "url_or_code": art.source_code},
    }
    current = corn_article(art.id) if art.crop == 'Кукуруза' else None
    if current and art.source_code in corn_catalog().get('legacy_source_articles', {}):
        payload.update(current)
    elif current:
        payload['additional_sources'] = current['additional_sources']
    return payload


class KnowledgeSearch:
    def retrieve(self, db, crop, growth_stage, query_symptoms, top_k=3, plant_part='leaf', model_label=None):
        """Ранжируем карточки по сходству текста, а не по вероятности болезни."""
        if crop not in {'Картофель', 'Кукуруза'} or top_k <= 0:
            return []
        articles = active_articles(db, crop).all()
        if crop == 'Кукуруза':
            scope = corn_scope()
            articles = [a for a in articles if plant_part in scope[a.id]['organs']]
            if model_label:
                # An accepted image class scopes retrieval; TF-IDF never creates a diagnosis.
                return [dict(article_payload(a), similarity_score=0.) for a in articles
                        if model_label in scope[a.id]['labels']][:top_k]
            articles = [a for a in articles if not scope[a.id].get('general')]
        if not (query_symptoms or '').strip():
            return []
        if not articles:
            return []
        payloads = [article_payload(a) for a in articles]
        texts = [f"{a['title']}. {a['symptoms_text']}".lower().replace("ё", "е") for a in payloads]
        # Фрагменты слов помогают сопоставлять разные окончания.
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        vectors = vectorizer.fit_transform(texts)
        query = vectorizer.transform([query_symptoms.lower().replace("ё", "е")])
        # Чем выше сходство, тем раньше карточка попадёт в выдачу.
        scores = cosine_similarity(query, vectors)[0]
        ranked = sorted(zip(articles, scores), key=lambda pair: pair[1], reverse=True)
        return [dict(article_payload(a), similarity_score=float(score))
                for a, score in ranked[:top_k] if score > (0.12 if crop == 'Кукуруза' else 0)]


def active_articles(db, crop='Картофель'):
    query = db.query(KnowledgeArticle).filter(KnowledgeArticle.crop == crop,
                                             KnowledgeArticle.id.notin_(LEGACY_DEMO_IDS))
    if crop == 'Кукуруза':
        query = query.filter(KnowledgeArticle.id.in_(corn_scope()))
    return query
