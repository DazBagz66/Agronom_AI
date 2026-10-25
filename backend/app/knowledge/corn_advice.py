"""Ground corn advice in retrieved DB cards without changing visual evidence."""
from app.schemas import KnowledgeMatch, SourceReference
from app.knowledge.search import article_payload, active_articles
from app.knowledge.corn import corn_scope, article_sources
from app.diagnostics.corn_care import reported_damage, positive_corn_symptoms


def enrich_corn_advice(db, request, result, search):
    if request.crop != 'Кукуруза':
        return result
    # Unresolved automatic routing must not silently retrieve leaf diagnoses.
    part = request.plant_part
    accepted = (result.status == 'success' and result.assessment_basis == 'model'
                and result.overall_confidence >= 65 and bool(result.model_label)
                and len(result.probable_causes) == 1 and result.probable_causes[0].probability >= 65)
    articles = []
    basis = 'model' if accepted else 'symptoms'
    if accepted or reported_damage(request.symptoms):
        articles = search.retrieve(db, request.crop, request.growth_stage,
            positive_corn_symptoms(request.symptoms or ''), top_k=1 if accepted else 2,
            plant_part=part, model_label=result.model_label if accepted else None)
    if not articles and part == 'ear':
        articles = [article_payload(a) for a in active_articles(db, request.crop).all()
                    if corn_scope()[a.id].get('general')]
        basis = 'organ'
    result = result.model_copy(deep=True)
    result.knowledge_matches = [KnowledgeMatch(article_id=a['id'], title=a['title'],
        symptoms_text=a['symptoms_text'], recommendations=a['recommendations'],
        source=SourceReference(**a['source']), additional_sources=a.get('additional_sources', []), basis=basis) for a in articles]
    if accepted and articles:
        result.recommendations = list(dict.fromkeys(rec for a in articles for rec in a['recommendations']))
        result.sources = [SourceReference(**source) for a in articles for source in article_sources(a)]
    # For uncertain images, reference cards are shown separately from "what to do".
    # Similarity, symptom descriptions and missing models do not become image confidence.
    return result
