import { useEffect, useState } from 'react';
import { Scan } from 'lucide-react';

const STEPS = [
  'Отправка наблюдения на сервер…',
  'Определение органа растения…',
  'Оценка признаков моделью…',
  'Подбор карточек рекомендаций…',
];

// Экран ожидания: настоящий запрос ещё идёт, поэтому проценты не показываем.
export default function ScanningOverlay({ image, onCancel }) {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setStep((current) => (current + 1) % STEPS.length), 900);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="bg-slate-900 rounded-2xl p-4 text-white shadow-2xl relative overflow-hidden animate-pulse-glow">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Scan className="w-4 h-4 text-emerald-400 animate-spin" />
          <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider">Анализ наблюдения</span>
        </div>
        {onCancel && (
          <button type="button" onClick={onCancel} className="text-[11px] text-slate-400 hover:text-white px-2 py-1 rounded-lg hover:bg-slate-800">
            Скрыть
          </button>
        )}
      </div>

      <div className="relative w-full aspect-square max-h-[320px] rounded-xl overflow-hidden mt-3 bg-black">
        {image && <img src={image} alt="Проверяемое растение" className="absolute inset-0 w-full h-full object-cover opacity-85" />}
        <div className="absolute inset-0 radar-grid-bg pointer-events-none opacity-40" />
        <div className="absolute left-0 right-0 h-16 pointer-events-none animate-radar-sweep border-b-2 border-emerald-400 bg-gradient-to-b from-transparent via-emerald-400/20 to-emerald-400/40 shadow-[0_0_15px_rgba(52,211,153,0.8)]" />
        <div className="absolute top-3 left-3 w-6 h-6 border-t-2 border-l-2 border-emerald-400 rounded-tl-sm pointer-events-none" />
        <div className="absolute top-3 right-3 w-6 h-6 border-t-2 border-r-2 border-emerald-400 rounded-tr-sm pointer-events-none" />
        <div className="absolute bottom-3 left-3 w-6 h-6 border-b-2 border-l-2 border-emerald-400 rounded-bl-sm pointer-events-none" />
        <div className="absolute bottom-3 right-3 w-6 h-6 border-b-2 border-r-2 border-emerald-400 rounded-br-sm pointer-events-none" />
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="w-20 h-20 border border-emerald-400/50 rounded-full flex items-center justify-center">
            <div className="w-2 h-2 bg-emerald-400 rounded-full animate-ping" />
          </div>
        </div>
        <div className="absolute bottom-2 left-2 right-2 bg-slate-950/80 backdrop-blur-md rounded-lg p-2 border border-slate-700/60">
          <div className="text-[11px] font-semibold text-slate-200 mb-1">{STEPS[step]}</div>
          <div className="w-full bg-slate-700 h-1.5 rounded-full overflow-hidden">
            <div className="bg-gradient-to-r from-emerald-500 to-emerald-300 h-full rounded-full w-1/3 animate-pulse" />
          </div>
        </div>
      </div>
      <p className="mt-3 text-center text-[11px] text-slate-400 m-0">Время ответа зависит от размера фото и загрузки сервера.</p>
    </div>
  );
}
