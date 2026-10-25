"""Refresh reviewed advice at display time without rewriting saved observations."""
from app.knowledge.corn import corn_article, corn_catalog, article_sources
from app.schemas import KnowledgeMatch, SourceReference


def refresh_corn_references(result):
    result = result.model_copy(deep=True)
    legacy = corn_catalog().get('legacy_source_articles', {})
    cards = []
    for card in result.knowledge_matches:
        current = corn_article(card.article_id)
        if current and card.source.url_or_code in legacy:
            card = KnowledgeMatch(article_id=current['id'], title=current['title'],
                symptoms_text=current['symptoms_text'], recommendations=current['recommendations'],
                source=current['source'], additional_sources=current['additional_sources'], basis=card.basis)
            if card.basis == 'model':
                result.recommendations = list(card.recommendations)
        cards.append(card)
    result.knowledge_matches = cards
    sources = []
    for source in result.sources:
        current = corn_article(legacy.get(source.url_or_code))
        sources.extend([SourceReference(**s) for s in article_sources(current)] if current else [source])
    if result.experimental_observation == 'smut_like':
        current = corn_article('corn_smut_v1')
        result.recommendations = list(current['recommendations'])
        sources = [SourceReference(**s) for s in article_sources(current)]
    result.sources = list({s.url_or_code: s for s in sources}.values())
    return result
