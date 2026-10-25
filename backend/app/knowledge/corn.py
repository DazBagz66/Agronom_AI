"""Versioned retrieval scope for the reviewed corn knowledge catalogue."""
import json
from functools import lru_cache
from pathlib import Path

CORN_PATH = Path(__file__).resolve().parents[2] / 'data/knowledge/corn.json'


@lru_cache(maxsize=1)
def corn_catalog():
    return json.loads(CORN_PATH.read_text(encoding='utf-8'))


def corn_scope():
    return corn_catalog()['scope']


def corn_article(article_id):
    """Current reviewed content, also used when displaying older DB records."""
    article = next((a for a in corn_catalog()['articles'] if a['id'] == article_id), None)
    if article is None:
        return None
    return {key: article[key] for key in ('id', 'crop', 'category', 'title',
            'growth_stages', 'symptoms_text', 'recommendations')} | {
        'source': {'title': article['source_title'], 'section': article['source_section'],
                   'url_or_code': article['source_code']},
        'additional_sources': article.get('additional_sources', []),
    }


def corn_article_for_label(label):
    return next((corn_article(key) for key, scope in corn_scope().items()
                 if label in scope['labels']), None)


def article_sources(article):
    return [article['source'], *article.get('additional_sources', [])]
