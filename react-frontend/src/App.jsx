import React, { useState, useEffect, useRef } from "react";

const API_URL = "http://127.0.0.1:8000/api";

const CROP_STAGES = {
  Картофель: ["Всходы", "Смыкание ботвы", "Бутонизация", "Цветение"],
  Пшеница: ["Всходы", "Кущение", "Выход в трубку", "Флаговый лист", "Колошение"],
  Кукуруза: ["Всходы", "3-5 листьев", "6-8 листьев", "Выметывание"],
};

export default function App() {
  const [crop, setCrop] = useState("Картофель");
  const [growthStage, setGrowthStage] = useState("Бутонизация");
  const [region, setRegion] = useState("Кемеровская область");

  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      type: "welcome",
      text: "Привет! Я ИИ-Советник агронома. Прикрепите фото ростка или листа (кнопка 📷 внизу) или опишите симптомы на поле. Можно отправлять запросы сколько угодно раз подряд и отвечать на уточняющие вопросы.",
    },
  ]);

  const [inputText, setInputText] = useState("");
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreviewName, setImagePreviewName] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const [isOffline, setIsOffline] = useState(false);
  const [offlineBundle, setOfflineBundle] = useState(null);
  const [historyList, setHistoryList] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Безопасное получение списка фаз (даже если из БД пришла нестандартная культура)
  const availableStages = CROP_STAGES[crop] || ["Всходы", "Вегетация", "Цветение"];

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  useEffect(() => {
    fetchHistory();
    fetchOfflineBundle();
  }, []);

  const handleCropChange = (newCrop) => {
    setCrop(newCrop);
    const stages = CROP_STAGES[newCrop] || ["Вегетация"];
    setGrowthStage(stages[0]);
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_URL}/history`);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data)) {
          setHistoryList(data);
        }
      }
    } catch (err) {
      console.warn("Сервер истории пока недоступен:", err);
    }
  };

  const fetchOfflineBundle = async () => {
    try {
      const res = await fetch(`${API_URL}/offline-bundle`);
      if (res.ok) {
        const data = await res.json();
        setOfflineBundle(data);
        localStorage.setItem("agro_offline_bundle", JSON.stringify(data));
      }
    } catch {
      try {
        const saved = localStorage.getItem("agro_offline_bundle");
        if (saved) setOfflineBundle(JSON.parse(saved));
      } catch (e) {
        console.warn("Не удалось прочитать localStorage", e);
      }
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImagePreviewName(file.name);

    const reader = new FileReader();
    reader.onload = () => {
      setSelectedImage(reader.result);
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  };

  const startNewChat = () => {
    setMessages([
      {
        id: Date.now(),
        role: "assistant",
        type: "welcome",
        text: `Начата новая проверка (${crop}, фаза: ${growthStage}). Загрузите фото листа или опишите проблему на поле.`,
      },
    ]);
    setInputText("");
    setSelectedImage(null);
    setImagePreviewName("");
  };

  const loadFromHistory = (item) => {
    if (!item) return;
    if (CROP_STAGES[item.crop]) {
      setCrop(item.crop);
    }
    setGrowthStage(item.growth_stage || "Вегетация");
    setRegion(item.region || "Кемеровская область");
    setMessages([
      {
        id: `user-${item.id}`,
        role: "user",
        text: item.symptoms || "Запрос из истории",
        image: null,
        meta: `${item.crop} • ${item.growth_stage} • ${item.region}`,
      },
      {
        id: `ai-${item.id}`,
        role: "assistant",
        type: "diagnosis",
        data: { ...(item.result || {}), check_id: item.id },
        isOffline: false,
      },
    ]);
  };

  const handleSendMessage = async (e, customText = null) => {
    if (e) e.preventDefault();
    const textToSend = customText !== null ? customText : inputText.trim();

    if (!textToSend && !selectedImage) return;
    if (isLoading) return;

    const currentImage = selectedImage;
    const userMessage = {
      id: Date.now(),
      role: "user",
      text: textToSend || "📷 [Отправлено только фото растения без текста]",
      image: currentImage,
      meta: `${crop} • ${growthStage} • ${region}`,
    };

    const previousUserTexts = messages
      .filter((m) => m.role === "user")
      .map((m) => m.text)
      .join(" | Дополнение: ");

    const combinedSymptoms = previousUserTexts
      ? `${previousUserTexts} | Уточнение: ${textToSend}`
      : textToSend || "Определи проблему по фотографии листа или ростка.";

    setMessages((prev) => [...prev, userMessage]);
    setInputText("");
    setSelectedImage(null);
    setImagePreviewName("");
    setIsLoading(true);

    if (isOffline) {
      setTimeout(() => {
        const offlineResp = buildOfflineResponse(crop, growthStage, combinedSymptoms);
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now() + 1,
            role: "assistant",
            type: "diagnosis",
            data: offlineResp,
            isOffline: true,
          },
        ]);
        setIsLoading(false);
      }, 400);
      return;
    }

    try {
      const res = await fetch(`${API_URL}/diagnose`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          crop,
          growth_stage: growthStage,
          region,
          symptoms: combinedSymptoms,
          image_base64: currentImage,
        }),
      });

      if (!res.ok) throw new Error(`Код ответа сервера: ${res.status}`);
      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: "assistant",
          type: "diagnosis",
          data: data,
          isOffline: false,
        },
      ]);
      fetchHistory();
    } catch (err) {
      const offlineResp = buildOfflineResponse(crop, growthStage, combinedSymptoms);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: "assistant",
          type: "diagnosis",
          data: offlineResp,
          isOffline: true,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const buildOfflineResponse = (cropName, stageName, queryText) => {
    const articles = offlineBundle?.articles || [];
    const byCrop = articles.filter(
      (a) => a.crop && a.crop.toLowerCase() === cropName.toLowerCase()
    );
    const pool = byCrop.length > 0 ? byCrop : articles;

    const words = (queryText || "")
      .toLowerCase()
      .split(/\s+/)
      .filter((w) => w.length > 3);
    let bestArt = pool[0];
    let maxHits = -1;

    pool.forEach((art) => {
      const hay = `${art.title || ""} ${art.symptoms_text || ""}`.toLowerCase();
      let hits = 0;
      words.forEach((w) => {
        if (hay.includes(w)) hits++;
      });
      if (hits > maxHits) {
        maxHits = hits;
        bestArt = art;
      }
    });

    if (!bestArt) {
      return {
        status: "insufficient_data",
        overall_confidence: 25,
        human_explanation:
          "Офлайн-справочник еще не загружен или сервер FastAPI (порт 8000) выключен.",
        probable_causes: [],
        recommendations: ["Проверьте, запущен ли uvicorn на порту 8000."],
        clarifying_questions: [],
        sources: [],
      };
    }

    return {
      status: "success",
      overall_confidence: 65,
      human_explanation: `[Автономный полевой режим]: Диагноз подобран по внутренней памяти устройства для культуры «${cropName}». Наиболее близкое совпадение — ${bestArt.title}.`,
      probable_causes: [
        {
          name: bestArt.title,
          category: bestArt.category,
          probability: 65,
          reasoning: `Локальный поиск в кеше телефона по культуре (${cropName}) и симптомам.`,
        },
      ],
      recommendations: bestArt.recommendations || [],
      clarifying_questions: [
        "При появлении интернета отправьте фото повторно для проверки через Vision-агента.",
      ],
      sources: bestArt.source ? [bestArt.source] : [],
    };
  };

  const handleFeedback = async (checkId, isHelpful) => {
    if (!checkId) return;
    const comment = isHelpful
      ? "Диагноз подтвержден агрономом"
      : prompt("Что оказалось на самом деле? (Комментарий сохранится в БД):");

    try {
      await fetch(`${API_URL}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          check_id: checkId,
          is_helpful: isHelpful,
          user_comment: comment || "",
        }),
      });
      alert("Обратная связь сохранена в базе данных!");
      fetchHistory();
    } catch {
      alert("Не удалось сохранить отзыв в офлайн-режиме.");
    }
  };

  return (
    <div className="flex h-screen w-screen bg-zinc-900 text-zinc-100 font-sans overflow-hidden">
      {/* ЛЕВАЯ ПАНЕЛЬ (САЙДБАР) */}
      <aside
        className={`${
          sidebarOpen ? "w-72" : "w-0"
        } bg-zinc-950 border-r border-zinc-800 flex flex-col transition-all duration-200 overflow-hidden shrink-0`}
      >
        <div className="p-4 border-b border-zinc-800 flex items-center space-x-2">
          <span className="text-xl">🌱</span>
          <div>
            <h1 className="font-bold text-sm text-white">Советник агронома</h1>
            <p className="text-[11px] text-emerald-400">Атвинта · AI + RAG</p>
          </div>
        </div>

        <div className="p-3">
          <button
            onClick={startNewChat}
            className="w-full bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium py-2.5 px-4 rounded-xl transition"
          >
            ＋ Новая диагностика
          </button>
        </div>

        <div className="px-3 mb-2">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-3 flex items-center justify-between">
            <div>
              <div className="text-xs font-medium text-zinc-200">
                {isOffline ? "🟠 Полевой офлайн" : "🟢 Онлайн (AI + RAG)"}
              </div>
              <div className="text-[10px] text-zinc-400">
                {isOffline ? "Локальный кеш" : "Gemini + ChromaDB"}
              </div>
            </div>
            <button
              onClick={() => setIsOffline(!isOffline)}
              className="px-2.5 py-1 rounded-lg text-xs font-semibold bg-zinc-800 text-zinc-200 hover:bg-zinc-700"
            >
              {isOffline ? "Вкл сеть" : "Офлайн"}
            </button>
          </div>
        </div>

        <div className="px-3 py-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
          История проверок ({historyList.length})
        </div>
        <div className="flex-1 overflow-y-auto px-2 space-y-1">
          {historyList.map((item) => (
            <button
              key={item.id}
              onClick={() => loadFromHistory(item)}
              className="w-full text-left px-3 py-2 rounded-xl hover:bg-zinc-900 transition flex flex-col"
            >
              <div className="flex items-center justify-between w-full">
                <span className="text-xs font-medium text-zinc-200 truncate">
                  {item.crop} · {item.growth_stage}
                </span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400">
                  {item.overall_confidence}%
                </span>
              </div>
              <span className="text-[11px] text-zinc-500 truncate mt-0.5">
                {item.symptoms}
              </span>
            </button>
          ))}
        </div>

        <div className="p-3 border-t border-zinc-800 space-y-1 text-xs text-zinc-400">
          <a
            href="http://127.0.0.1:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="block hover:text-emerald-400"
          >
            ⚙️ API & Swagger ↗
          </a>
        </div>
      </aside>

      {/* ОСНОВНАЯ ЧАСТЬ С ЧАТОМ */}
      <div className="flex-1 flex flex-col h-full bg-zinc-900">
        <header className="border-b border-zinc-800 px-4 py-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-xs text-zinc-300"
            >
              ☰
            </button>

            <select
              value={crop}
              onChange={(e) => handleCropChange(e.target.value)}
              className="bg-zinc-800 border border-zinc-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
            >
              {Object.keys(CROP_STAGES).map((c) => (
                <option key={c} value={c}>
                  🌾 {c}
                </option>
              ))}
            </select>

            <select
              value={growthStage}
              onChange={(e) => setGrowthStage(e.target.value)}
              className="bg-zinc-800 border border-zinc-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
            >
              {availableStages.map((st) => (
                <option key={st} value={st}>
                  Фаза: {st}
                </option>
              ))}
            </select>

            <input
              type="text"
              value={region}
              onChange={(e) => setRegion(e.target.value)}
              className="bg-zinc-800 border border-zinc-700 rounded-lg px-2.5 py-1.5 text-xs text-white w-40"
            />
          </div>

          <div className="flex items-center space-x-1.5">
            <button
              type="button"
              onClick={() => {
                setCrop("Картофель");
                setGrowthStage("Бутонизация");
                handleSendMessage(
                  null,
                  "На листьях картофеля темно-бурые мокнущие пятна, с нижней стороны белый паутинистый налет."
                );
              }}
              className="text-[11px] bg-emerald-950 hover:bg-emerald-900 text-emerald-300 border border-emerald-800 px-2.5 py-1 rounded-lg"
            >
              ⚡ Демо: Картофель
            </button>
            <button
              type="button"
              onClick={() =>
                handleSendMessage(null, "Лист немного пожелтел, что делать?")
              }
              className="text-[11px] bg-amber-950 hover:bg-amber-900 text-amber-300 border border-amber-800 px-2.5 py-1 rounded-lg"
            >
              ⚡ Демо: Мало данных
            </button>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${
                  msg.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                {msg.role === "user" && (
                  <div className="max-w-xl bg-emerald-700 text-white rounded-2xl px-4 py-3 shadow space-y-2">
                    <div className="text-[10px] text-emerald-200">{msg.meta}</div>
                    {msg.image && (
                      <img
                        src={msg.image}
                        alt="Лист"
                        className="max-h-60 rounded-xl object-cover"
                      />
                    )}
                    <p className="text-sm">{msg.text}</p>
                  </div>
                )}

                {msg.role === "assistant" && msg.type === "welcome" && (
                  <div className="max-w-2xl bg-zinc-800 border border-zinc-700 rounded-2xl px-5 py-4 text-sm text-zinc-200">
                    {msg.text}
                  </div>
                )}

                {msg.role === "assistant" && msg.type === "diagnosis" && msg.data && (
                  <div className="w-full max-w-2xl bg-zinc-800 border border-zinc-700 rounded-2xl p-5 shadow-lg space-y-4">
                    <div className="flex items-center justify-between border-b border-zinc-700 pb-3">
                      <span
                        className={`text-xs font-bold px-2.5 py-1 rounded-full ${
                          msg.data.status === "success"
                            ? "bg-emerald-500/20 text-emerald-300"
                            : "bg-amber-500/20 text-amber-300"
                        }`}
                      >
                        {msg.data.status === "success"
                          ? "✓ Диагноз готов"
                          : "⚠️ Нужны уточнения"}
                      </span>
                      <span className="text-sm font-bold text-white">
                        Уверенность: {msg.data.overall_confidence || 0}%
                      </span>
                    </div>

                    <div className="text-sm text-zinc-100 bg-zinc-900/60 p-3.5 rounded-xl">
                      {msg.data.human_explanation}
                    </div>

                    {Array.isArray(msg.data.probable_causes) &&
                      msg.data.probable_causes.length > 0 && (
                        <div className="space-y-2">
                          <h4 className="text-xs font-semibold uppercase text-zinc-400">
                            Вероятные причины:
                          </h4>
                          {msg.data.probable_causes.map((cause, idx) => (
                            <div
                              key={idx}
                              className="bg-zinc-900 border border-zinc-700 rounded-xl p-3"
                            >
                              <div className="flex justify-between mb-1">
                                <span className="font-semibold text-sm text-emerald-300">
                                  {cause.name} ({cause.category})
                                </span>
                                <span className="text-xs text-emerald-400">
                                  {cause.probability}%
                                </span>
                              </div>
                              <p className="text-xs text-zinc-300">{cause.reasoning}</p>
                            </div>
                          ))}
                        </div>
                      )}

                    {Array.isArray(msg.data.recommendations) &&
                      msg.data.recommendations.length > 0 && (
                        <div>
                          <h4 className="text-xs font-semibold uppercase text-zinc-400 mb-2">
                            Рекомендации:
                          </h4>
                          <ul className="list-decimal list-inside text-xs text-zinc-200 space-y-1 bg-zinc-900/60 p-3.5 rounded-xl">
                            {msg.data.recommendations.map((rec, i) => (
                              <li key={i}>{rec}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                    {Array.isArray(msg.data.clarifying_questions) &&
                      msg.data.clarifying_questions.length > 0 && (
                        <div className="bg-amber-950/40 border border-amber-700/50 rounded-xl p-3.5">
                          <div className="text-xs font-semibold text-amber-300 mb-1">
                            Ответьте в чате ниже для уточнения:
                          </div>
                          <ul className="list-disc list-inside text-xs text-amber-100 space-y-1">
                            {msg.data.clarifying_questions.map((q, i) => (
                              <li key={i}>{q}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                    {Array.isArray(msg.data.sources) &&
                      msg.data.sources.length > 0 && (
                        <div className="pt-2 border-t border-zinc-700">
                          <h4 className="text-[11px] font-semibold uppercase text-zinc-400 mb-1">
                            Источники (RAG):
                          </h4>
                          {msg.data.sources.map((src, i) => (
                            <div key={i} className="text-xs text-zinc-400">
                              📚 <span className="text-zinc-200">{src.title}</span> —{" "}
                              {src.section} ({src.url_or_code})
                            </div>
                          ))}
                        </div>
                      )}

                    {msg.data.check_id && (
                      <div className="pt-2 border-t border-zinc-700 flex justify-between items-center">
                        <span className="text-[11px] text-zinc-500">
                          ID проверки: #{msg.data.check_id}
                        </span>
                        <div className="space-x-2">
                          <button
                            onClick={() => handleFeedback(msg.data.check_id, true)}
                            className="text-xs bg-zinc-700 hover:bg-emerald-700 px-3 py-1 rounded-lg"
                          >
                            👍 Точно
                          </button>
                          <button
                            onClick={() => handleFeedback(msg.data.check_id, false)}
                            className="text-xs bg-zinc-700 hover:bg-rose-700 px-3 py-1 rounded-lg"
                          >
                            👎 Ошибка
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}

            {isLoading && (
              <div className="text-xs text-zinc-300 bg-zinc-800 p-3 rounded-xl w-fit">
                ⏳ ИИ-агенты анализируют фото и сверяются с базой...
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
        </main>

        {/* ПАНЕЛЬ ВВОДА ВНИЗУ */}
        <footer className="p-4 bg-zinc-900 border-t border-zinc-800">
          <form
            onSubmit={(e) => handleSendMessage(e)}
            className="max-w-3xl mx-auto space-y-2"
          >
            {selectedImage && (
              <div className="flex items-center space-x-3 bg-zinc-800 p-2 rounded-xl w-fit">
                <img
                  src={selectedImage}
                  alt="Превью"
                  className="h-12 w-12 object-cover rounded-lg"
                />
                <span className="text-xs text-emerald-400">
                  {imagePreviewName || "Фото прикреплено"}
                </span>
                <button
                  type="button"
                  onClick={() => setSelectedImage(null)}
                  className="text-zinc-400 hover:text-rose-400 px-2"
                >
                  ✕
                </button>
              </div>
            )}

            <div className="flex items-center space-x-2 bg-zinc-800 border border-zinc-700 rounded-2xl px-3 py-2">
              <input
                type="file"
                accept="image/*"
                ref={fileInputRef}
                onChange={handleFileChange}
                className="hidden"
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="p-2 rounded-xl bg-zinc-700 hover:bg-zinc-600 text-xs text-white shrink-0"
              >
                📷 Фото
              </button>

              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Опишите симптомы или отправьте только фото..."
                className="w-full bg-transparent text-sm text-white focus:outline-none px-2"
              />

              <button
                type="submit"
                disabled={isLoading || (!inputText.trim() && !selectedImage)}
                className="px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:bg-zinc-700 text-white text-sm font-medium shrink-0"
              >
                Отправить ➤
              </button>
            </div>
          </form>
        </footer>
      </div>
    </div>
  );
}