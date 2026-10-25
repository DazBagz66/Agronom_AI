import { useState } from 'react';
import { ClipboardList, Calendar, ScanLine, ExternalLink, Layers } from 'lucide-react';
import { CROPS } from '../crops.js';

// Полевой журнал: сохранённые на сервере сессии с фильтром по культуре.
export default function FieldJournal({ sessions, busy, onOpenSession, onNavigateToScanner }) {
  const [filter, setFilter] = useState('all');
  const visible = sessions.filter((item) => filter === 'all' || item.crop === filter);
  const countFor = (crop) => sessions.filter((item) => item.crop === crop).length;

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-black text-slate-900 tracking-tight m-0">Полевой журнал</h2>
        <p className="text-xs text-slate-500 mt-0.5 m-0">Сессии обследований, сохранённые на сервере</p>
      </div>

      <div className="grid grid-cols-3 gap-2">
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-2 text-center">
          <span className="text-[9px] font-bold text-slate-500 uppercase block">Всего</span>
          <span className="text-base font-black text-slate-900">{sessions.length}</span>
        </div>
        {Object.entries(CROPS).map(([name, config]) => (
          <div key={name} className="bg-emerald-50 border border-emerald-200 rounded-xl p-2 text-center">
            <span className="text-[9px] font-bold text-emerald-700 uppercase block">{config.icon} {name}</span>
            <span className="text-base font-black text-emerald-800">{countFor(name)}</span>
          </div>
        ))}
      </div>

      <div className="flex gap-2 overflow-x-auto scrollbar-none pb-1">
        {[['all', 'Все'], ...Object.keys(CROPS).map((name) => [name, `${CROPS[name].icon} ${name}`])].map(([value, label]) => (
          <button key={value} type="button" onClick={() => setFilter(value)} className={`px-3 py-1.5 rounded-full text-xs font-bold whitespace-nowrap transition min-h-[36px] ${filter === value ? 'bg-emerald-700 text-white shadow-xs' : 'bg-white border border-slate-200 text-slate-700 hover:border-slate-300'}`}>
            {label}
          </button>
        ))}
      </div>

      {visible.length === 0 ? (
        <div className="bg-white border border-dashed border-slate-300 rounded-2xl p-6 text-center space-y-3">
          <div className="w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center mx-auto"><ClipboardList className="w-6 h-6 text-slate-500" /></div>
          <div>
            <h4 className="font-bold text-sm text-slate-800 m-0">Журнал пока пуст</h4>
            <p className="text-xs text-slate-500 max-w-xs mx-auto mt-1 leading-relaxed m-0">Выполните проверку на вкладке «Диагностика» — сессия сохранится здесь автоматически.</p>
          </div>
          <button type="button" onClick={onNavigateToScanner} className="inline-flex items-center gap-1.5 px-4 py-2 bg-emerald-700 text-white font-bold text-xs rounded-xl hover:bg-emerald-800 transition min-h-[40px]">
            <ScanLine className="w-4 h-4" />
            <span>Начать проверку</span>
          </button>
        </div>
      ) : (
        <div className="space-y-2.5">
          {visible.map((item) => (
            <button key={item.id} type="button" disabled={busy} onClick={() => onOpenSession(item)} className="w-full text-left bg-white border border-slate-200 rounded-xl p-3 shadow-xs flex items-center gap-3 hover:border-emerald-300 hover:shadow-md transition group disabled:opacity-60">
              <div className="w-14 h-14 bg-emerald-50 rounded-lg shrink-0 border border-emerald-100 flex items-center justify-center text-2xl" aria-hidden="true">{CROPS[item.crop]?.icon || '🌱'}</div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] font-bold text-slate-500 uppercase">{item.crop}</span>
                  <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-100 text-emerald-800"><Layers className="w-2.5 h-2.5" />{item.observation_count} набл.</span>
                </div>
                <h4 className="font-bold text-xs text-slate-900 truncate mt-0.5 group-hover:text-emerald-700 transition m-0">{item.title}</h4>
                <div className="flex items-center gap-2 text-[10px] text-slate-500 mt-1 min-w-0">
                  <span className="flex items-center gap-1 shrink-0"><Calendar className="w-3 h-3 text-slate-400" />{new Date(item.updated_at).toLocaleDateString('ru-RU')}</span>
                  <span>•</span>
                  <span className="truncate">{item.symptoms || 'Фотография'}</span>
                </div>
              </div>
              <ExternalLink className="w-4 h-4 text-slate-400 group-hover:text-emerald-600 shrink-0" />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
