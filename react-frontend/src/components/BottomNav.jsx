import { ScanLine, BookOpen, ClipboardList } from 'lucide-react';

const TABS = [
  { id: 'diagnostics', label: 'Диагностика', icon: ScanLine },
  { id: 'catalog', label: 'Справочник', icon: BookOpen },
  { id: 'journal', label: 'Полевой журнал', icon: ClipboardList },
];

// Нижняя навигация с тремя вкладками; у журнала показано число сессий.
export default function BottomNav({ activeTab, onTabChange, journalCount = 0 }) {
  return (
    <nav
      role="tablist"
      aria-label="Основная навигация приложения"
      className="fixed bottom-0 left-0 right-0 sm:left-auto sm:right-auto sm:w-full sm:max-w-md z-40 bg-white/95 backdrop-blur-md border-t border-slate-200/90 shadow-lg px-2 pt-1 pb-safe"
    >
      <div className="flex items-center justify-around h-14">
        {TABS.map(({ id, label, icon: Icon }) => {
          const active = activeTab === id;
          const badge = id === 'journal' && journalCount > 0 ? journalCount : null;
          return (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => onTabChange(id)}
              className={`relative flex flex-col items-center justify-center flex-1 min-h-[48px] py-1 px-2 rounded-xl transition-all duration-200 select-none outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 ${
                active ? 'text-emerald-700 bg-emerald-50/70' : 'text-slate-500 hover:text-slate-800 hover:bg-slate-50'
              }`}
            >
              {active && <span className="absolute -top-1 w-8 h-1 bg-emerald-600 rounded-full" aria-hidden="true" />}
              <div className="relative">
                <Icon className={`w-5 h-5 transition-transform duration-150 ${active ? 'scale-110 stroke-[2.4]' : 'stroke-[1.8]'}`} />
                {badge && (
                  <span
                    className="absolute -top-1.5 -right-2.5 min-w-[18px] h-[18px] px-1 bg-emerald-600 text-white text-[10px] font-bold rounded-full flex items-center justify-center ring-2 ring-white"
                    aria-label={`Сессий в журнале: ${badge}`}
                  >
                    {badge}
                  </span>
                )}
              </div>
              <span className={`text-[11px] mt-1 leading-tight tracking-tight ${active ? 'text-emerald-800 font-bold' : 'text-slate-600 font-medium'}`}>
                {label}
              </span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}
