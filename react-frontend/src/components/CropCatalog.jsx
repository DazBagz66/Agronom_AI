import { useEffect, useMemo, useState } from 'react';
import { Search, X, ArrowLeft, ArrowRight, Info, ShieldCheck, BookOpen, CheckCircle2, ExternalLink, Tag } from 'lucide-react';
import { CROPS } from '../crops.js';
import { fetchKnowledge } from '../api.js';

function ArticleView({ article, crop, onBack }) {
  const [tab, setTab] = useState('symptoms');
  const sources = [article.source, ...(article.additional_sources || [])].filter((source) => source?.url_or_code);
  const tabs = [
    { id: 'symptoms', label: 'Признаки', icon: Info },
    { id: 'actions', label: 'Рекомендации', icon: ShieldCheck },
    { id: 'sources', label: 'Источники', icon: BookOpen },
  ];
  return (
    <div className="space-y-3">
      <button type="button" onClick={onBack} className="inline-flex items-center gap-1.5 py-1.5 px-3 rounded-xl bg-slate-100 text-slate-700 hover:bg-slate-200 text-xs font-bold transition min-h-[40px]">
        <ArrowLeft className="w-4 h-4" />
        <span>К списку</span>
      </button>
      <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm">
        <div className="p-4 border-b border-slate-100">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-[10px] font-bold text-slate-500 uppercase">{CROPS[crop]?.icon} {crop}</span>
            {article.category && <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-100 text-emerald-800">{article.category}</span>}
          </div>
          <h2 className="text-lg font-black text-slate-900 leading-snug mt-1 m-0">{article.title}</h2>
          {article.growth_stages?.length > 0 && <p className="text-xs text-slate-500 mt-1 m-0">Фазы: {article.growth_stages.join(', ')}</p>}
        </div>
        <div className="grid grid-cols-3 border-b border-slate-200 bg-slate-50/80 p-1 gap-1">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button key={id} type="button" onClick={() => setTab(id)} className={`py-2 px-1 rounded-lg text-xs font-bold transition flex flex-col items-center justify-center gap-1 min-h-[44px] ${tab === id ? 'bg-white text-emerald-800 shadow-xs border border-slate-200/80' : 'text-slate-500 hover:text-slate-800'}`}>
              <Icon className="w-3.5 h-3.5" />
              <span className="text-[10px] leading-none">{label}</span>
            </button>
          ))}
        </div>
        <div className="p-4 space-y-2 text-xs text-slate-700 leading-relaxed">
          {tab === 'symptoms' && <p className="m-0">{article.symptoms_text || 'Описание признаков для этой категории ещё не подготовлено.'}</p>}
          {tab === 'actions' && (article.recommendations?.length
            ? article.recommendations.map((text, i) => (
              <div key={i} className="flex items-start gap-2.5 bg-emerald-50/60 p-2.5 rounded-xl border border-emerald-100">
                <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                <span>{text}</span>
              </div>
            ))
            : <p className="m-0 text-slate-500">Рекомендации ещё не подготовлены.</p>)}
          {tab === 'sources' && (sources.length
            ? sources.map((source) => (
              <a key={source.url_or_code} href={source.url_or_code.startsWith('http') ? source.url_or_code : undefined} target="_blank" rel="noopener noreferrer" className="flex items-start gap-1.5 text-emerald-800 underline underline-offset-2">
                <ExternalLink className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                <span>{source.title}{source.section ? ` — ${source.section}` : ''}</span>
              </a>
            ))
            : <p className="m-0 text-slate-500">Источники не указаны.</p>)}
        </div>
      </div>
    </div>
  );
}

// Справочник: карточки рекомендаций с сервера и классы установленных моделей.
export default function CropCatalog({ initialCrop, capabilities }) {
  const [crop, setCrop] = useState(initialCrop);
  const [query, setQuery] = useState('');
  // Ответ сервера хранится вместе с культурой, для которой он загружен.
  const [loaded, setLoaded] = useState({ crop: null, data: null, error: '' });
  const [selected, setSelected] = useState(null);
  const knowledge = loaded.crop === crop ? loaded.data : null;
  const error = loaded.crop === crop ? loaded.error : '';

  useEffect(() => {
    let cancelled = false;
    fetchKnowledge(crop)
      .then((data) => { if (!cancelled) setLoaded({ crop, data, error: '' }); })
      .catch(() => { if (!cancelled) setLoaded({ crop, data: null, error: 'Не удалось загрузить справочник. Проверьте связь с сервером.' }); });
    return () => { cancelled = true; };
  }, [crop]);

  const articles = useMemo(() => {
    const needle = query.toLowerCase().trim();
    return (knowledge?.articles || []).filter((article) =>
      !needle || article.title.toLowerCase().includes(needle) || (article.symptoms_text || '').toLowerCase().includes(needle));
  }, [knowledge, query]);

  if (selected) return <ArticleView article={selected} crop={crop} onBack={() => setSelected(null)} />;

  const labels = capabilities[crop]?.labels || {};

  return (
    <div className="space-y-3.5">
      <div>
        <h2 className="text-lg font-black text-slate-900 tracking-tight m-0">Справочник культур</h2>
        <p className="text-xs text-slate-500 mt-0.5 m-0">{knowledge ? `${knowledge.count} карточек` : 'Загрузка…'} · {knowledge?.scope || 'карточки рекомендаций и классы моделей'}</p>
      </div>

      <div className="relative">
        <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
        <input
          type="text"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Поиск по названию или признакам…"
          className="w-full pl-9 pr-8 py-2.5 bg-white border border-slate-200 rounded-xl text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition"
        />
        {query && (
          <button type="button" onClick={() => setQuery('')} className="absolute right-2.5 top-1/2 -translate-y-1/2 p-0.5 text-slate-400 hover:text-slate-600" aria-label="Очистить поиск">
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      <div className="flex gap-2 overflow-x-auto scrollbar-none pb-1">
        {Object.entries(CROPS).map(([name, config]) => (
          <button key={name} type="button" onClick={() => setCrop(name)} className={`px-3 py-1.5 rounded-full text-xs font-bold whitespace-nowrap transition min-h-[36px] flex items-center gap-1.5 ${crop === name ? 'bg-emerald-700 text-white shadow-xs' : 'bg-white border border-slate-200 text-slate-700 hover:border-slate-300'}`}>
            <span aria-hidden="true">{config.icon}</span>
            <span>{name}</span>
          </button>
        ))}
      </div>

      <div className="bg-emerald-50/70 border border-emerald-200 rounded-xl p-3 text-xs space-y-1.5">
        <span className="font-extrabold text-emerald-900 flex items-center gap-1.5"><Tag className="w-4 h-4 text-emerald-700" /> Классы установленной модели</span>
        {Object.keys(labels).length
          ? <div className="flex flex-wrap gap-1.5">{Object.values(labels).map((label) => <span key={label} className="px-2 py-0.5 rounded-full bg-white border border-emerald-200 text-emerald-900 text-[11px] font-semibold">{label}</span>)}</div>
          : <p className="text-[11px] text-slate-600 m-0">Модель для этой культуры сейчас недоступна.</p>}
      </div>

      {error && <div className="bg-rose-50 border border-rose-200 rounded-xl p-3 text-xs text-rose-800">{error}</div>}

      <div className="space-y-2.5">
        {knowledge && articles.length === 0 && (
          <div className="bg-white border border-dashed border-slate-300 rounded-xl p-6 text-center text-slate-500 text-xs">Ничего не найдено{query ? ` по запросу «${query}»` : ''}.</div>
        )}
        {articles.map((article) => (
          <button key={article.id} type="button" onClick={() => setSelected(article)} className="w-full text-left bg-white border border-slate-200 rounded-xl p-3 shadow-xs flex items-center gap-3 hover:border-emerald-500 hover:shadow-md transition group active:scale-[0.99]">
            <div className="w-12 h-12 bg-emerald-50 rounded-lg shrink-0 border border-emerald-100 flex items-center justify-center text-2xl" aria-hidden="true">{CROPS[crop]?.icon}</div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5">
                <span className="text-[10px] font-bold text-slate-500 uppercase">{crop}</span>
                {article.category && <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-100 text-emerald-800">{article.category}</span>}
              </div>
              <h4 className="font-bold text-xs text-slate-900 truncate mt-0.5 group-hover:text-emerald-700 transition-colors m-0">{article.title}</h4>
              <p className="text-[11px] text-slate-500 truncate mt-0.5 m-0">{article.symptoms_text || article.source?.title || ''}</p>
            </div>
            <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-emerald-600 group-hover:translate-x-0.5 transition shrink-0" />
          </button>
        ))}
      </div>
    </div>
  );
}
