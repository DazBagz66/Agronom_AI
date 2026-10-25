import { useEffect, useState } from 'react';
import { AlertTriangle } from 'lucide-react';

import { visibleDiagnosis } from './diagnosis.js';
import { fetchCapabilities, fetchSessions, fetchSession, fetchHistory, diagnose, imageSource } from './api.js';
import MobileFrame from './components/MobileFrame.jsx';
import Header from './components/Header.jsx';
import BottomNav from './components/BottomNav.jsx';
import Scanner from './components/Scanner.jsx';
import ScanningOverlay from './components/ScanningOverlay.jsx';
import SessionView from './components/SessionView.jsx';
import CropCatalog from './components/CropCatalog.jsx';
import FieldJournal from './components/FieldJournal.jsx';

const PART_NAMES = { leaf: 'Лист', ear: 'Початок', flower: 'Корзинка' };

function observationMeta(crop, stage, part) {
  return `${crop} • ${stage}${part && part !== 'auto' ? ` • ${PART_NAMES[part] || part}` : ''}`;
}

// Адрес вида #catalog, #journal или #session=<id> открывает нужный экран сразу.
function initialLocation() {
  const hash = window.location.hash.slice(1);
  if (hash === 'catalog' || hash === 'journal') return { tab: hash, sessionId: null };
  if (hash.startsWith('session=')) return { tab: 'diagnostics', sessionId: hash.slice('session='.length) };
  return { tab: 'diagnostics', sessionId: null };
}

export default function App() {
  const [activeTab, setActiveTab] = useState(() => initialLocation().tab);
  const [crop, setCrop] = useState('Кукуруза');
  const [growthStage, setGrowthStage] = useState('Вегетация');
  const [plantPart, setPlantPart] = useState('auto');
  const [capabilities, setCapabilities] = useState({});
  const [online, setOnline] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [session, setSession] = useState(null); // { id, title, observations: [...] }
  const [busy, setBusy] = useState(false);
  const [pendingImage, setPendingImage] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    refreshSessions();
    fetchCapabilities()
      .then((data) => { setCapabilities(data); setOnline(true); })
      .catch(() => setOnline(false));
    const { sessionId } = initialLocation();
    if (sessionId) openSession({ id: sessionId });
    // Только при первом показе: адрес читается один раз.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function refreshSessions() {
    try {
      const data = await fetchSessions();
      if (Array.isArray(data)) setSessions(data);
    } catch (err) {
      console.warn('Список сессий пока недоступен:', err);
    }
  }

  const chooseCrop = (nextCrop) => {
    setCrop(nextCrop);
    setGrowthStage('Вегетация');
    setPlantPart(nextCrop === 'Кукуруза' ? 'auto' : 'leaf');
    setSession(null);
    setError('');
  };

  const startNewCheck = () => {
    if (busy) return;
    setSession(null);
    setError('');
  };

  // Отправляем одно наблюдение; контекст сессии хранит сервер.
  async function submitObservation({ text, image }) {
    if (busy) return false;
    setBusy(true);
    setError('');
    setPendingImage(image);
    try {
      const data = await diagnose({
        crop,
        session_id: session?.id ?? null,
        growth_stage: growthStage,
        symptoms: text,
        image_base64: image,
        plant_part: plantPart,
      });
      const observation = {
        id: data.check_id ?? Date.now(),
        text: text || 'Фото растения без описания',
        image,
        hasImage: Boolean(image),
        meta: observationMeta(crop, growthStage, plantPart),
        data: visibleDiagnosis(data),
      };
      setSession((current) => ({
        id: data.session_id,
        title: current?.title || '',
        observations: [...(current?.observations || []), observation],
      }));
      setOnline(true);
      refreshSessions();
      return true;
    } catch {
      setError('Не удалось получить результат. Проверьте соединение с сервером и повторите запрос.');
      return false;
    } finally {
      setBusy(false);
      setPendingImage(null);
    }
  }

  // Открываем сохранённую сессию; повторного анализа здесь нет.
  async function openSession(item) {
    if (busy) return;
    setBusy(true);
    setError('');
    try {
      const loaded = await fetchSession(item.id);
      const last = loaded.observations.at(-1);
      setCrop(loaded.crop);
      setGrowthStage(item.growth_stage || last?.growth_stage || 'Вегетация');
      setPlantPart(loaded.crop === 'Кукуруза' ? 'auto' : item.plant_part || last?.plant_part || 'leaf');
      setSession({
        id: loaded.id,
        title: loaded.title,
        observations: loaded.observations.map((observation) => ({
          id: observation.id,
          text: observation.symptoms || 'Фото растения без описания',
          image: imageSource(observation.image_base64),
          hasImage: observation.has_image,
          meta: `${loaded.crop} • ${observation.growth_stage} • ${new Date(observation.created_at).toLocaleString('ru-RU')}`,
          data: visibleDiagnosis({ ...observation.result, check_id: observation.id }),
        })),
      });
      setActiveTab('diagnostics');
    } catch {
      setError('Не удалось открыть сессию. Повторите попытку.');
    } finally {
      setBusy(false);
    }
  }

  // После уточнения пользователя сервер меняет отображение записи — перечитываем её.
  async function refreshObservation(observation) {
    const items = await fetchHistory();
    refreshSessions();
    const item = items.find((entry) => entry.id === observation.data.check_id);
    if (!item) return;
    setSession((current) => current && {
      ...current,
      observations: current.observations.map((entry) => entry.id === observation.id
        ? { ...entry, data: visibleDiagnosis({ ...item.result, check_id: item.id }) }
        : entry),
    });
  }

  const subtitle = activeTab === 'catalog'
    ? 'Справочник культур и рекомендаций'
    : activeTab === 'journal'
      ? 'История обследований'
      : session
        ? `Сессия: ${crop}`
        : 'Экспресс-диагностика кукурузы и подсолнечника';

  return (
    <MobileFrame>
      <Header subtitle={subtitle} online={online} />

      <main className="flex-1 overflow-y-auto px-4 py-4 pb-safe-nav bg-slate-50/60">
        {error && (
          <div className="mb-4 bg-rose-50 border border-rose-200 rounded-xl p-3 text-xs text-rose-800 flex items-start gap-2" role="alert">
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-rose-600" />
            <span>{error}</span>
          </div>
        )}

        {activeTab === 'diagnostics' && (session ? (
          <SessionView
            session={session}
            crop={crop}
            busy={busy}
            pendingImage={pendingImage}
            onBack={startNewCheck}
            onSubmit={submitObservation}
            onFeedbackSaved={refreshObservation}
          />
        ) : busy ? (
          <ScanningOverlay image={pendingImage} />
        ) : (
          <Scanner
            crop={crop}
            onCropChange={chooseCrop}
            growthStage={growthStage}
            onStageChange={setGrowthStage}
            plantPart={plantPart}
            onPartChange={setPlantPart}
            capabilities={capabilities}
            busy={busy}
            onSubmit={submitObservation}
          />
        ))}

        {activeTab === 'catalog' && <CropCatalog key={crop} initialCrop={crop} capabilities={capabilities} />}

        {activeTab === 'journal' && (
          <FieldJournal
            sessions={sessions}
            busy={busy}
            onOpenSession={openSession}
            onNavigateToScanner={() => { startNewCheck(); setActiveTab('diagnostics'); }}
          />
        )}
      </main>

      <BottomNav activeTab={activeTab} onTabChange={setActiveTab} journalCount={sessions.length} />
    </MobileFrame>
  );
}
