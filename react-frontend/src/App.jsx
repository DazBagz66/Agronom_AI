import React, { useState, useEffect, useRef } from "react";

import { visibleDiagnosis } from "./diagnosis.js";
import { assessmentContext } from "./assessment.js";
import { CROPS, cropContextKey } from "./crops.js";

// По умолчанию страница обращается к API на том же локальном сервере.
const API_URL = import.meta.env.VITE_API_URL || "/api";

export default function App() {
  const [crop, setCrop] = useState("Кукуруза");
  const [growthStage, setGrowthStage] = useState("Вегетация");

  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      type: "welcome",
      text: "Выберите кукурузу или подсолнечник и прикрепите чёткое фото листа. Для подсолнечника также доступна проверка корзинки. Описание симптомов поможет заметить противоречия, но для классификации нужна фотография. Результат предварительный: модель различает только классы обучающего набора.",
    },
  ]);

  const [inputText, setInputText] = useState("");
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreviewName, setImagePreviewName] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const [historyList, setHistoryList] = useState([]);
  const [plantPart, setPlantPart] = useState("leaf");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const availableStages = CROPS[crop].stages;
  const visibleHistory = historyList.filter((item) => item.crop === crop);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  useEffect(() => {
    fetchHistory();
  }, []);

  async function fetchHistory() {
    try {
      const res = await fetch(`${API_URL}/history?limit=100`);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data)) {
          setHistoryList(data);
        }
      }
    } catch (err) {
      console.warn("Сервер истории пока недоступен:", err);
    }
  }

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

  const startNewChat = (nextCrop = crop) => {
    if (isLoading) return;
    setMessages([
      {
        id: Date.now(),
        role: "assistant",
        type: "welcome",
        text: `Начата новая проверка: ${nextCrop}. ${CROPS[nextCrop].hint} Загрузите фото и при необходимости опишите изменения.`,
      },
    ]);
    setInputText("");
    setSelectedImage(null);
    setImagePreviewName("");
  };

  // Открываем сохранённый результат; повторного анализа здесь нет.
  const loadFromHistory = (item) => {
    if (!item || isLoading) return;

    setCrop(item.crop);
    setGrowthStage(item.growth_stage || "Вегетация");
    setPlantPart("leaf");
    setSelectedImage(null);
    setImagePreviewName("");
    setInputText("");
    setMessages([
      {
        id: `user-${item.id}`,
        role: "user",
        text: item.symptoms || "Запрос из истории",
        image: null,
        meta: `${item.crop} • ${item.growth_stage}`,
      },
      {
        id: `ai-${item.id}`,
        role: "assistant",
        type: "diagnosis",
        data: visibleDiagnosis({ ...(item.result || {}), check_id: item.id }),

      },
    ]);
  };

  const handleSendMessage = async (e, customText = null) => {
    if (e) e.preventDefault();
    const textToSend = customText !== null ? customText : inputText.trim();

    if (!textToSend && !selectedImage) return;
    if (isLoading) return;

    // Новое фото начинает проверку образца, текст может уточнять прошлое фото.
    const currentImage = selectedImage;
    const contextKey = cropContextKey(crop, plantPart, growthStage);
    const { imageForAnalysis, combinedSymptoms } = assessmentContext(messages, currentImage, textToSend, contextKey);
    const userMessage = {
      id: Date.now(),
      role: "user",
      text: textToSend || "📷 [Отправлено только фото растения без текста]",
      image: currentImage,
      imageContext: imageForAnalysis,
      symptomsContext: combinedSymptoms,
      contextKey,
      meta: `${crop} • ${growthStage}`,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputText("");
    setSelectedImage(null);
    setImagePreviewName("");
    setIsLoading(true);

    try {
      const res = await fetch(`${API_URL}/diagnose`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          crop,
          growth_stage: growthStage,
          symptoms: combinedSymptoms,
          image_base64: imageForAnalysis,
          plant_part: plantPart,
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
          data: visibleDiagnosis(data),

        },
      ]);
      fetchHistory();
    } catch {
      // Возвращаем ввод, чтобы после ошибки можно было повторить отправку.
      setInputText(textToSend);
      setSelectedImage(currentImage);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: "assistant",
          type: "error",
          text: "Не удалось получить результат. Проверьте соединение с сервером и повторите запрос.",

        },
      ]);
    } finally {
      setIsLoading(false);
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
          <span className="text-xl">{CROPS[crop].icon}</span>
          <div>
            <h1 className="font-bold text-sm text-white">Советник агронома</h1>
            <p className="text-[11px] text-emerald-400">Кукуруза и подсолнечник</p>
          </div>
        </div>

        <div className="p-3">
          <button
            onClick={() => startNewChat()}
            disabled={isLoading}
            className="w-full bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium py-2.5 px-4 rounded-xl transition"
          >
            ＋ Новая диагностика
          </button>
        </div>

        <div className="px-3 py-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
          История проверок ({visibleHistory.length})
        </div>
        <div className="flex-1 overflow-y-auto px-2 space-y-1">
          {visibleHistory.map((item) => (
            <button
              key={item.id}
              onClick={() => loadFromHistory(item)}
              disabled={isLoading}
              className="w-full text-left px-3 py-2 rounded-xl hover:bg-zinc-900 transition flex flex-col"
            >
              <div className="flex items-center justify-between w-full">
                <span className="text-xs font-medium text-zinc-200 truncate">
                  {item.crop} · {item.growth_stage}
                </span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400">
                  {item.status === "healthy" ? "Без явных повреждений" : `${item.overall_confidence}%`}
                </span>
              </div>
              <span className="text-[11px] text-zinc-500 truncate mt-0.5">
                {item.symptoms}
              </span>
            </button>
          ))}
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

            <select aria-label="Культура" value={crop} disabled={isLoading}
              onChange={(e) => {
                const nextCrop = e.target.value;
                setCrop(nextCrop);
                setGrowthStage("Вегетация");
                setPlantPart("leaf");
                startNewChat(nextCrop);
              }}
              className="bg-zinc-800 border border-zinc-700 rounded-lg px-2.5 py-1.5 text-xs text-white">
              {Object.entries(CROPS).map(([name, config]) => <option key={name} value={name}>{config.icon} {name}</option>)}
            </select>

            <select
              value={growthStage}
              disabled={isLoading}
              onChange={(e) => setGrowthStage(e.target.value)}
              className="bg-zinc-800 border border-zinc-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
            >
              {availableStages.map((st) => (
                <option key={st} value={st}>
                  Фаза: {st}
                </option>
              ))}
            </select>

            <select aria-label="Часть растения" value={plantPart} disabled={isLoading} onChange={(e) => setPlantPart(e.target.value)}
              className="bg-zinc-800 border border-zinc-700 rounded-lg px-2.5 py-1.5 text-xs text-white">
              {CROPS[crop].parts.map(([value, name]) => <option key={value} value={value}>{name}</option>)}
            </select>

          </div>

        </header>

        <main className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
          <div className="max-w-3xl mx-auto space-y-6">
            <div className="text-xs text-amber-100 bg-amber-950/40 border border-amber-700/50 rounded-xl p-3" role="note">
              Предварительная проверка с ограниченным набором классов. {CROPS[crop].hint}
              {" "}Отсутствие болезни по фото не подтверждается. Снимайте выбранный орган крупным планом;
              общий вид растения может привести к ошибке.
            </div>
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
                        alt="Фото растения"
                        className="max-h-60 rounded-xl object-cover"
                      />
                    )}
                    <p className="text-sm">{msg.text}</p>
                  </div>
                )}

                {msg.role === "assistant" && ["welcome", "error"].includes(msg.type) && (
                  <div className="max-w-2xl bg-zinc-800 border border-zinc-700 rounded-2xl px-5 py-4 text-sm text-zinc-200">
                    {msg.text}
                  </div>
                )}

                {msg.role === "assistant" && msg.type === "diagnosis" && msg.data && (
                  <div className="w-full max-w-2xl bg-zinc-800 border border-zinc-700 rounded-2xl p-5 shadow-lg space-y-4">
                    <div className="flex items-center justify-between border-b border-zinc-700 pb-3">
                      <span
                        className={`text-xs font-bold px-2.5 py-1 rounded-full ${
                          ["success", "healthy"].includes(msg.data.status)
                            ? "bg-emerald-500/20 text-emerald-300"
                            : "bg-amber-500/20 text-amber-300"
                        }`}
                      >
                        {["success", "healthy"].includes(msg.data.status)
                          ? (msg.data.status === "healthy" ? "✓ Видимых признаков болезни нет" : "✓ Оценка готова")
                          : "Нет уверенного вывода"}
                      </span>
                      {msg.data.assessment_basis !== "visual_observation" && <span className="text-sm font-bold text-white">
                        Оценка модели: {msg.data.overall_confidence || 0}%
                      </span>}
                    </div>

                    <div className="text-sm text-zinc-100 bg-zinc-900/60 p-3.5 rounded-xl">
                      {msg.data.human_explanation}
                    </div>

                    {Array.isArray(msg.data.probable_causes) &&
                      msg.data.probable_causes.length > 0 && (
                        <div className="space-y-2">
                          <h4 className="text-xs font-semibold uppercase text-zinc-400">
                            Предварительная гипотеза:
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

                    {msg.data.possible_factors?.length > 0 && (
                      <div className="bg-zinc-900/60 p-3.5 rounded-xl space-y-2">
                        <h4 className="text-xs font-semibold uppercase text-zinc-400">Что могло способствовать заболеванию:</h4>
                        <p className="text-xs text-zinc-400">Это возможные условия, а не установленные по фото факты.</p>
                        <ul className="list-disc list-inside text-xs text-zinc-200 space-y-1">
                          {msg.data.possible_factors.map((factor, i) => <li key={i}>{factor}</li>)}
                        </ul>
                      </div>
                    )}

                    {Array.isArray(msg.data.recommendations) &&
                      msg.data.recommendations.length > 0 && (
                        <div>
                          <h4 className="text-xs font-semibold uppercase text-zinc-400 mb-2">
                            Что делать:
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


                  </div>
                )}
              </div>
            ))}

            {isLoading && (
              <div className="text-xs text-zinc-300 bg-zinc-800 p-3 rounded-xl w-fit">
                ⏳ Рассматриваем фото и оцениваем состояние растения...
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
