import { Sprout } from 'lucide-react';

// Шапка: название, подпись текущего экрана и честный индикатор связи с сервером.
export default function Header({ subtitle, online }) {
  const status = online === null
    ? { dot: 'bg-slate-400', text: 'Проверка связи', box: 'bg-slate-50 border-slate-200 text-slate-600' }
    : online
      ? { dot: 'bg-emerald-500 animate-pulse', text: 'Сервер на связи', box: 'bg-emerald-50 border-emerald-200/80 text-emerald-800' }
      : { dot: 'bg-rose-500', text: 'Нет связи', box: 'bg-rose-50 border-rose-200 text-rose-800' };

  return (
    <header className="sticky top-0 z-30 w-full bg-white/95 backdrop-blur-md border-b border-slate-200/80 px-4 pt-safe pb-3 shadow-sm">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="w-10 h-10 shrink-0 rounded-xl bg-gradient-to-br from-emerald-500 to-emerald-700 text-white flex items-center justify-center shadow-md shadow-emerald-600/20 ring-1 ring-emerald-600/30">
            <Sprout className="w-6 h-6 stroke-[2.2]" aria-hidden="true" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-1.5">
              <h1 className="font-extrabold text-base tracking-tight text-slate-900 leading-none m-0">Помощник агронома</h1>
              <span className="px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-emerald-100 text-emerald-800 rounded">Агро ИИ</span>
            </div>
            <p className="text-xs text-slate-500 font-medium mt-0.5 leading-none truncate m-0">{subtitle}</p>
          </div>
        </div>
        <div className={`flex items-center gap-1 px-2.5 py-1 rounded-full border text-[11px] font-semibold shrink-0 ${status.box}`}>
          <span className={`w-2 h-2 rounded-full ${status.dot}`} aria-hidden="true" />
          <span>{status.text}</span>
        </div>
      </div>
    </header>
  );
}
