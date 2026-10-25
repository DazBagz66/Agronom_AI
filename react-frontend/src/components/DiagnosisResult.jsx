import { useState } from 'react';
import { Info, Bug, ShieldCheck, BookOpen, CheckCircle2, AlertTriangle, ExternalLink } from 'lucide-react';
import { diagnosisTitle } from '../diagnosis.js';
import StatusBadge from './StatusBadge.jsx';
import DiagnosisFeedback from '../DiagnosisFeedback.jsx';
import { API_URL } from '../api.js';

const PART_NAMES = { leaf: 'лист', ear: 'початок', whole: 'общий вид растения или посадки', unknown: 'не определён уверенно' };

const TABS = [
  { id: 'summary', label: 'Вывод', icon: Info },
  { id: 'causes', label: 'Гипотезы', icon: Bug },
  { id: 'actions', label: 'Что делать', icon: ShieldCheck },
  { id: 'knowledge', label: 'Справочник', icon: BookOpen },
];

function SectionTitle({ icon: Icon, children, color = 'text-emerald-600' }) {
  return (
    <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5 m-0">
      <Icon className={`w-4 h-4 ${color}`} />
      <span>{children}</span>
    </h4>
  );
}

function Empty({ children }) {
  return <p className="text-xs text-slate-500 m-0 py-2">{children}</p>;
}

function SourceLink({ source }) {
  return (
    <a
      href={source.url_or_code}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-start gap-1.5 text-xs text-emerald-800 underline underline-offset-2 hover:text-emerald-900"
    >
      <ExternalLink className="w-3.5 h-3.5 shrink-0 mt-0.5" />
      <span>{source.title}{source.section ? ` — ${source.section}` : ''}</span>
    </a>
  );
}

// Карточка заключения по одному наблюдению: фото, статус, четыре раздела ответа сервера.
export default function DiagnosisResult({ data, crop, image, hasImage, onFeedbackSaved }) {
  const causes = data.probable_causes || [];
  const factors = data.possible_factors || [];
  const recommendations = data.recommendations || [];
  const questions = data.clarifying_questions || [];
  const matches = data.knowledge_matches || [];
  const sources = data.sources || [];
  // Сначала открываем самый содержательный раздел: гипотезы, иначе действия.
  const [tab, setTab] = useState(causes.length ? 'causes' : recommendations.length + questions.length ? 'actions' : 'summary');
  const counts = { causes: causes.length, actions: recommendations.length + questions.length, knowledge: matches.length + sources.length };

  return (
    <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm">
      {image ? (
        <div className="relative w-full aspect-[16/10] bg-slate-900">
          <img src={image} alt="Фото растения" className="absolute inset-0 w-full h-full object-cover" />
          <div className="absolute top-3 left-3 flex items-center gap-1.5 flex-wrap">
            <span className="px-2.5 py-1 rounded-full text-xs font-bold bg-black/70 backdrop-blur-md text-white border border-white/20">{crop}</span>
            <StatusBadge status={data.status} size="md" />
          </div>
          {data.assessment_basis === 'model' && (
            <div className="absolute bottom-3 right-3 px-3 py-1.5 rounded-xl bg-black/75 backdrop-blur-md border border-white/20 text-white text-right">
              <span className="text-[10px] uppercase font-bold text-slate-300 block leading-none">Оценка модели</span>
              <span className="text-sm font-black text-emerald-400 font-mono">{data.overall_confidence || 0}%</span>
            </div>
          )}
        </div>
      ) : (
        <div className="px-4 pt-4 flex items-center gap-1.5 flex-wrap">
          <span className="px-2.5 py-1 rounded-full text-xs font-bold bg-slate-100 text-slate-700">{crop}</span>
          <StatusBadge status={data.status} size="md" />
          {data.assessment_basis === 'model' && <span className="text-xs font-bold text-slate-700 ml-auto">Оценка модели: {data.overall_confidence || 0}%</span>}
        </div>
      )}

      <div className="p-4 border-b border-slate-100">
        <h2 className="text-lg font-black text-slate-900 leading-snug m-0">{diagnosisTitle(data)}</h2>
        {data.organ_detection_method === 'automatic' && (
          <p className="text-xs font-semibold text-emerald-800 mt-0.5 m-0">Объект на фото: {PART_NAMES[data.detected_plant_part] || 'не определён'}</p>
        )}
        <p className="text-xs text-slate-700 mt-2 leading-relaxed m-0">{data.human_explanation}</p>
      </div>

      <div className="grid grid-cols-4 border-b border-slate-200 bg-slate-50/80 p-1 gap-1">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            onClick={() => setTab(id)}
            className={`relative py-2 px-1 rounded-lg text-xs font-bold transition flex flex-col items-center justify-center gap-1 min-h-[44px] ${
              tab === id ? 'bg-white text-emerald-800 shadow-xs border border-slate-200/80' : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            <Icon className="w-3.5 h-3.5" />
            <span className="text-[10px] leading-none">{label}</span>
            {counts[id] > 0 && <span className="absolute top-1 right-1 min-w-[14px] h-[14px] px-1 rounded-full bg-emerald-600 text-white text-[9px] font-bold flex items-center justify-center">{counts[id]}</span>}
          </button>
        ))}
      </div>

      <div className="p-4 space-y-3">
        {tab === 'summary' && (
          <>
            {data.symptom_assessment && data.assessment_basis === 'model' && <p className="text-xs text-slate-700 leading-relaxed m-0">{data.symptom_assessment}</p>}
            {data.previous_photo_assessment && (
              <details className="text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded-xl p-2.5">
                <summary className="font-semibold">Предыдущая оценка фотографии</summary>
                <p className="mt-2 m-0 leading-relaxed">{data.previous_photo_assessment}</p>
              </details>
            )}
            {data.context_observations?.length > 0 && (
              <details className="text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded-xl p-2.5">
                <summary className="font-semibold">Предыдущие наблюдения, учтённые в ответе</summary>
                <ul className="list-disc list-inside mt-2 space-y-1 m-0">{data.context_observations.map((text, i) => <li key={i}>{text}</li>)}</ul>
              </details>
            )}
            {!data.symptom_assessment && !data.previous_photo_assessment && !data.context_observations?.length && (
              <Empty>Подробности в разделах «Гипотезы», «Что делать» и «Справочник».</Empty>
            )}
          </>
        )}

        {tab === 'causes' && (
          <>
            <SectionTitle icon={Bug}>Предварительные гипотезы</SectionTitle>
            {causes.length === 0 && <Empty>Ни одна гипотеза не достигла порога 65 %.</Empty>}
            {causes.map((cause, i) => (
              <div key={i} className="bg-slate-50 border border-slate-200 rounded-xl p-2.5">
                <div className="flex justify-between gap-2 mb-1">
                  <span className="font-bold text-xs text-slate-900">{cause.name} <span className="font-medium text-slate-500">({cause.category})</span></span>
                  <span className="text-xs font-mono font-bold text-emerald-700 shrink-0">{cause.probability}%</span>
                </div>
                <p className="text-xs text-slate-700 leading-relaxed m-0">{cause.reasoning}</p>
              </div>
            ))}
            {factors.length > 0 && (
              <div className="pt-1 space-y-1.5">
                <SectionTitle icon={Info} color="text-amber-600">Что могло способствовать</SectionTitle>
                <p className="text-[11px] text-slate-500 m-0">Это возможные условия, а не установленные по фото факты.</p>
                <ul className="list-disc list-inside text-xs text-slate-700 space-y-1 m-0">{factors.map((factor, i) => <li key={i}>{factor}</li>)}</ul>
              </div>
            )}
          </>
        )}

        {tab === 'actions' && (
          <>
            <SectionTitle icon={ShieldCheck}>Что делать</SectionTitle>
            {recommendations.length === 0 && questions.length === 0 && <Empty>Рекомендаций для этого наблюдения нет.</Empty>}
            {recommendations.map((text, i) => (
              <div key={i} className="flex items-start gap-2.5 text-xs text-slate-700 leading-relaxed bg-slate-50 p-2.5 rounded-xl border border-slate-200">
                <span className="w-5 h-5 rounded-full bg-slate-200 text-slate-700 text-[10px] font-black flex items-center justify-center shrink-0 mt-0.5">{i + 1}</span>
                <span>{text}</span>
              </div>
            ))}
            {questions.length > 0 && (
              <div className="bg-amber-50 border border-amber-200 rounded-xl p-2.5 text-[11px] text-amber-900 leading-relaxed">
                <div className="flex items-center gap-1.5 font-bold mb-1"><AlertTriangle className="w-4 h-4 text-amber-700" /> Ответьте в поле ниже для уточнения:</div>
                <ul className="list-disc list-inside space-y-1 m-0">{questions.map((question, i) => <li key={i}>{question}</li>)}</ul>
              </div>
            )}
          </>
        )}

        {tab === 'knowledge' && (
          <>
            <SectionTitle icon={BookOpen}>Инструкции и меры борьбы</SectionTitle>
            {matches.length === 0 && sources.length === 0 && <Empty>Подходящих карточек справочника нет.</Empty>}
            {matches.map((card) => (
              <details key={card.article_id} className="bg-emerald-50/60 border border-emerald-100 rounded-xl p-2.5 text-xs">
                <summary className="font-bold text-emerald-900">{card.title}</summary>
                <p className="text-amber-800 mt-2 m-0">
                  {card.basis === 'model'
                    ? 'Рекомендации к предварительной гипотезе модели.'
                    : card.basis === 'symptoms'
                      ? 'Справка по вашему описанию. Совпадение текста не подтверждает болезнь на фотографии.'
                      : 'Общие рекомендации по осмотру выбранного органа.'}
                </p>
                <p className="text-slate-700 mt-2 m-0"><span className="font-semibold">Признаки из справочника:</span> {card.symptoms_text}</p>
                <ul className="mt-2 space-y-1.5 m-0 list-none p-0">
                  {card.recommendations.map((text, i) => (
                    <li key={i} className="flex items-start gap-2 text-slate-700"><CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" /><span>{text}</span></li>
                  ))}
                </ul>
                <div className="mt-2 space-y-1">{[card.source, ...(card.additional_sources || [])].map((source) => <SourceLink key={source.url_or_code} source={source} />)}</div>
              </details>
            ))}
            {sources.length > 0 && (
              <div className="space-y-1 pt-1">
                <p className="text-[11px] font-bold text-slate-500 m-0">Источники рекомендаций:</p>
                {sources.map((source) => <SourceLink key={source.url_or_code} source={source} />)}
              </div>
            )}
          </>
        )}

        {crop === 'Кукуруза' && hasImage && data.check_id && (
          <DiagnosisFeedback key={data.check_id} checkId={data.check_id} apiUrl={API_URL} onSaved={onFeedbackSaved} />
        )}
      </div>
    </div>
  );
}
