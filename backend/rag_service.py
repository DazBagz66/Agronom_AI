import chromadb
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from models import KnowledgeArticle


class AgroRAGService:
    def __init__(self, persist_dir: str = "./chroma_storage"):
        # Инициализируем персистентную векторную БД на диске
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)

        # Создаем или загружаем коллекцию с косинусным расстоянием (HNSW индекс)
        self.collection = self.chroma_client.get_or_create_collection(
            name="agro_knowledge_vectors",
            metadata={"hnsw:space": "cosine"}
        )

    def sync_sql_to_vector_db(self, db: Session):
        """
        Индексирует статьи из SQLite в векторную БД ChromaDB (только те, которых еще нет в индексе)
        """
        articles = db.query(KnowledgeArticle).all()
        if not articles:
            return

        existing_ids = set(self.collection.get()["ids"])

        new_ids = []
        new_docs = []
        new_metadatas = []

        for art in articles:
            if art.id not in existing_ids:
                # Формируем богатый текст для векторного эмбеддинга
                stages_str = ", ".join(art.growth_stages)
                doc_text = f"Культура: {art.crop}. Проблема: {art.title} ({art.category}). Фазы: {stages_str}. Симптомы: {art.symptoms_text}"

                new_ids.append(art.id)
                new_docs.append(doc_text)
                new_metadatas.append({
                    "crop": art.crop,
                    "category": art.category
                })

        if new_ids:
            self.collection.add(
                ids=new_ids,
                documents=new_docs,
                metadatas=new_metadatas
            )

    def retrieve(self, db: Session, crop: str, growth_stage: str, query_symptoms: str, top_k: int = 3) -> List[
        Dict[str, Any]]:
        """
        1. Делает векторный поиск в ChromaDB с предфильтрацией по культуре
        2. Забирает полные записи и официальные источники из SQLite по найденным ID
        """
        query_text = f"Культура: {crop}. Фаза: {growth_stage}. Симптомы: {query_symptoms}"

        # Сначала пробуем векторный поиск со строгим фильтром по культуре
        vector_results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where={"crop": crop}
        )

        matched_ids = vector_results["ids"][0] if vector_results["ids"] else []
        distances = vector_results["distances"][0] if vector_results["distances"] else []

        # Если по конкретной культуре в БД еще мало записей — ищем по всей векторной базе без фильтра
        if not matched_ids:
            vector_results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k
            )
            matched_ids = vector_results["ids"][0] if vector_results["ids"] else []
            distances = vector_results["distances"][0] if vector_results["distances"] else []

        if not matched_ids:
            return []

        # Достаем полные структурированные карточки из SQLite по первичному ключу (O(1) по индексу)
        articles = db.query(KnowledgeArticle).filter(KnowledgeArticle.id.in_(matched_ids)).all()
        article_map = {art.id: art for art in articles}

        results = []
        for doc_id, dist in zip(matched_ids, distances):
            art = article_map.get(doc_id)
            if art:
                # Переводим косинусное расстояние в коэффициент близости (similarity)
                similarity = round(max(0.0, 1.0 - float(dist)), 3)
                results.append({
                    "id": art.id,
                    "similarity_score": similarity,
                    "crop": art.crop,
                    "category": art.category,
                    "title": art.title,
                    "growth_stages": art.growth_stages,
                    "symptoms_text": art.symptoms_text,
                    "recommendations": art.recommendations,
                    "source": {
                        "title": art.source_title,
                        "section": art.source_section,
                        "url_or_code": art.source_code
                    }
                })

        return results