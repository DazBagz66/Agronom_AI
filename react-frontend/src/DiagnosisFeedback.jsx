import { useEffect, useState } from 'react';
import { MessageSquareWarning } from 'lucide-react';

const FIELD = 'mt-1 block w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-normal text-slate-900 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent';

// Уточнение диагноза пользователем: сохраняется для проверки, модель не меняет.
export default function DiagnosisFeedback({ checkId, apiUrl, onSaved }) {
  const [labels, setLabels] = useState({});
  const [label, setLabel] = useState('unknown_disease');
  const [part, setPart] = useState('unknown');
  const [notes, setNotes] = useState('');
  const [group, setGroup] = useState('');
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);
  const [previous, setPrevious] = useState(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${apiUrl}/feedback/labels`).then((r) => {
      if (!r.ok) throw new Error('Не удалось загрузить категории');
      return r.json();
    }).then((d) => { if (!cancelled) setLabels(d.labels); })
      .catch((e) => { if (!cancelled) setMessage(e.message); });
    return () => { cancelled = true; };
  }, [apiUrl]);

  useEffect(() => {
    let cancelled = false;
    fetch(`${apiUrl}/history/${checkId}/feedback`).then((r) => {
      if (!r.ok) throw new Error('Не удалось проверить сохранённые уточнения');
      return r.json();
    }).then((data) => { if (!cancelled) setPrevious(data.feedback); })
      .catch((error) => { if (!cancelled) setMessage(error.message); });
    return () => { cancelled = true; };
  }, [apiUrl, checkId]);

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    try {
      const response = await fetch(`${apiUrl}/history/${checkId}/feedback`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ proposed_labels: [label], plant_part: part, notes, specimen_group: group }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Не удалось сохранить уточнение');
      setMessage(data.message); setSaved(true);
      setPrevious({ proposed_labels: [label], status: data.status });
      if (onSaved) onSaved().catch(() => setMessage(`${data.message} Обновите историю, чтобы увидеть изменение.`));
    } catch (e) { setMessage(e.message); }
    finally { setBusy(false); }
  }

  return (
    <details className="bg-slate-50 border border-slate-200 rounded-xl p-3 text-xs text-slate-700">
      <summary className="font-bold text-slate-800 flex items-center gap-1.5">
        <MessageSquareWarning className="w-4 h-4 text-amber-600" />
        {previous ? `Ваше уточнение: ${previous.proposed_labels.map((key) => labels[key] || key).join(', ')}` : 'Результат неверный? Уточнить диагноз'}
      </summary>
      {previous && <p className="mt-2 m-0">{previous.status === 'pending' ? 'Ожидает проверки.' : 'Уточнение сохранено.'} Это ваша отметка, а не новый вывод модели.</p>}
      <p className="mt-2 m-0 leading-relaxed">Укажите предполагаемую причину или заключение специалиста. Отметка сохраняется для проверки и будущего обучения; текущая модель от неё автоматически не меняется.</p>
      {!saved && (
        <form onSubmit={submit} className="mt-3 grid gap-3">
          <label className="font-semibold">Предполагаемая причина
            <select className={FIELD} value={label} onChange={(e) => setLabel(e.target.value)}>
              {Object.entries(labels).map(([key, name]) => <option key={key} value={key}>{name}</option>)}
            </select>
          </label>
          <label className="font-semibold">Что на фото
            <select className={FIELD} value={part} onChange={(e) => setPart(e.target.value)}>
              {[['unknown', 'Не уверен'], ['leaf', 'Лист'], ['ear', 'Початок'], ['stem', 'Стебель'], ['whole', 'Растение целиком']].map(([key, name]) => <option key={key} value={key}>{name}</option>)}
            </select>
          </label>
          <label className="font-semibold">Название образца или серии фото
            <input className={FIELD} value={group} maxLength={100} onChange={(e) => setGroup(e.target.value)} placeholder="Например: поле 2, растение 15" />
          </label>
          <label className="font-semibold">Как установлена причина
            <textarea className={FIELD} rows={3} value={notes} maxLength={2000} onChange={(e) => setNotes(e.target.value)} placeholder="Наблюдения, заключение агронома или лаборатории" />
          </label>
          <button type="submit" disabled={busy || !Object.keys(labels).length} className="rounded-xl bg-emerald-700 hover:bg-emerald-800 py-2.5 text-white font-bold disabled:bg-slate-200 disabled:text-slate-400 transition">
            {busy ? 'Сохраняем…' : 'Сохранить уточнение'}
          </button>
        </form>
      )}
      {message && <p role="status" className="mt-2 m-0 text-emerald-900">{message}</p>}
    </details>
  );
}
