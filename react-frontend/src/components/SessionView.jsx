import { useEffect, useRef } from 'react';
import { ArrowLeft, RotateCcw, Layers } from 'lucide-react';
import DiagnosisResult from './DiagnosisResult.jsx';
import Composer from './Composer.jsx';
import ScanningOverlay from './ScanningOverlay.jsx';

// Открытая сессия: наблюдения с заключениями и поле для дополнения.
export default function SessionView({ session, crop, busy, pendingImage, onBack, onSubmit, onFeedbackSaved }) {
  const endRef = useRef(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [session.observations.length, busy]);

  return (
    <div className="space-y-4 pb-2">
      <div className="flex items-center justify-between gap-2">
        <button type="button" onClick={onBack} className="inline-flex items-center gap-1.5 py-1.5 px-3 rounded-xl bg-slate-100 text-slate-700 hover:bg-slate-200 text-xs font-bold transition min-h-[40px]">
          <ArrowLeft className="w-4 h-4" />
          <span>Назад</span>
        </button>
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-800 text-[11px] font-semibold">
          <Layers className="w-3.5 h-3.5" />
          <span>Наблюдений: {session.observations.length}</span>
        </span>
        <button type="button" onClick={onBack} title="Новая проверка" className="inline-flex items-center gap-1 py-1.5 px-2.5 rounded-xl border border-slate-200 text-slate-600 hover:bg-slate-50 text-xs font-semibold transition min-h-[40px]">
          <RotateCcw className="w-4 h-4" />
          <span>Заново</span>
        </button>
      </div>

      {session.title && <p className="text-xs text-slate-500 m-0 truncate">{session.title}</p>}

      {session.observations.map((observation) => (
        <div key={observation.id} className="space-y-2">
          <div className="flex justify-end">
            <div className="max-w-[85%] bg-emerald-700 text-white rounded-2xl rounded-br-md px-3.5 py-2.5 shadow-sm space-y-1.5">
              <div className="text-[10px] text-emerald-200">{observation.meta}</div>
              {observation.image && !observation.data && <img src={observation.image} alt="Фото растения" className="max-h-48 rounded-xl object-cover" />}
              <p className="text-sm m-0">{observation.text}</p>
            </div>
          </div>
          {observation.data && (
            <DiagnosisResult
              data={observation.data}
              crop={crop}
              image={observation.image}
              hasImage={observation.hasImage}
              onFeedbackSaved={() => onFeedbackSaved(observation)}
            />
          )}
        </div>
      ))}

      {busy && <ScanningOverlay image={pendingImage} />}

      <div className="pt-1">
        <p className="text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-1.5 m-0">Дополнить сессию</p>
        <Composer variant="compact" busy={busy} placeholder="Ответьте на вопросы или добавьте фото…" onSubmit={onSubmit} />
      </div>
      <div ref={endRef} />
    </div>
  );
}
