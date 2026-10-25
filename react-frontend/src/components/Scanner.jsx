import { Settings2, Info } from 'lucide-react';
import { CROPS, photoParts } from '../crops.js';
import Composer from './Composer.jsx';

// Экран новой проверки: ввод наблюдения, параметры культуры и подсказки по съёмке.
export default function Scanner({ crop, onCropChange, growthStage, onStageChange, plantPart, onPartChange, capabilities, busy, onSubmit }) {
  const capability = capabilities[crop];
  const parts = photoParts(crop, capability);
  const hint = capability
    ? `Доступные категории: ${Object.values(capability.labels).join(', ') || 'модель недоступна'}.`
    : CROPS[crop].hint;

  return (
    <div className="space-y-4">
      <Composer variant="hero" busy={busy} placeholder="Например: на листьях ржаво-бурые пустулы, нижний ярус желтеет…" onSubmit={onSubmit} />

      <div className="bg-white border border-slate-200 rounded-2xl p-3 shadow-xs space-y-3">
        <div className="flex items-center gap-1.5 text-xs font-bold text-slate-800 uppercase tracking-wider">
          <Settings2 className="w-4 h-4 text-emerald-600" />
          <span>Параметры проверки</span>
        </div>
        <div className="flex gap-2 overflow-x-auto scrollbar-none pb-0.5">
          {Object.entries(CROPS).map(([name, config]) => (
            <button
              key={name}
              type="button"
              disabled={busy}
              onClick={() => onCropChange(name)}
              className={`px-3 py-1.5 rounded-full text-xs font-bold whitespace-nowrap transition min-h-[36px] flex items-center gap-1.5 ${
                crop === name ? 'bg-emerald-700 text-white shadow-xs' : 'bg-white border border-slate-200 text-slate-700 hover:border-slate-300'
              }`}
            >
              <span aria-hidden="true">{config.icon}</span>
              <span>{name}</span>
            </button>
          ))}
        </div>
        <div className={`grid gap-2 ${crop === 'Кукуруза' ? 'grid-cols-1' : 'grid-cols-2'}`}>
          <label className="text-[11px] font-bold text-slate-500 uppercase">
            Фаза развития
            <select
              value={growthStage}
              disabled={busy}
              onChange={(event) => onStageChange(event.target.value)}
              className="mt-1 block w-full rounded-xl border border-slate-200 bg-slate-50 px-2.5 py-2 text-xs font-semibold normal-case text-slate-900 focus:outline-none focus:ring-2 focus:ring-emerald-500"
            >
              {CROPS[crop].stages.map((stage) => <option key={stage} value={stage}>{stage}</option>)}
            </select>
          </label>
          {crop !== 'Кукуруза' && (
            <label className="text-[11px] font-bold text-slate-500 uppercase">
              Часть растения
              <select
                value={plantPart}
                disabled={busy}
                onChange={(event) => onPartChange(event.target.value)}
                className="mt-1 block w-full rounded-xl border border-slate-200 bg-slate-50 px-2.5 py-2 text-xs font-semibold normal-case text-slate-900 focus:outline-none focus:ring-2 focus:ring-emerald-500"
              >
                {parts.map(([value, name]) => <option key={value} value={value}>{name}</option>)}
              </select>
            </label>
          )}
        </div>
        {crop === 'Кукуруза' && (
          <p className="text-[11px] text-slate-500 m-0">Орган кукурузы (лист или початок) определяется по фото автоматически.</p>
        )}
      </div>

      <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 text-[11px] text-amber-900 leading-relaxed flex items-start gap-2" role="note">
        <Info className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
        <span>
          Предварительная проверка с ограниченным набором классов. {hint} Отсутствие болезни по фото не подтверждается.
          Снимайте один орган крупным планом; общий вид растения может привести к ошибке.
        </span>
      </div>

      {crop === 'Кукуруза' && (
        <details className="bg-white border border-slate-200 rounded-xl p-3 text-xs text-slate-700 shadow-xs">
          <summary className="font-bold text-slate-800">Как сфотографировать кукурузу</summary>
          <ul className="list-disc list-inside mt-3 space-y-2 leading-relaxed">
            <li>Лист или початок должен занимать большую часть кадра. Для длинного листа сделайте несколько перекрывающихся снимков. Початок снимите целиком и отдельно крупно изменённые зёрна, не разрушая наросты.</li>
            <li>Рассеянный дневной свет, резкие жилки или зёрна, без вспышки, бликов, фильтров и сильных теней.</li>
            <li>Лист снимите с обеих сторон; для початка добавьте крупный план изменённых зёрен и снимок соседнего початка для сравнения. Не обрезайте повреждения.</li>
            <li>Отправляйте исходное фото с телефона, желательно от 800 пикселей по короткой стороне. Скриншот страницы и общий вид поля хуже подходят для этой модели.</li>
            <li>Удачное фото уменьшает ошибки съёмки, но не гарантирует правильный ответ модели.</li>
          </ul>
        </details>
      )}
    </div>
  );
}
