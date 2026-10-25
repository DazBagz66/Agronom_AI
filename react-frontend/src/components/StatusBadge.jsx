import { CheckCircle2, AlertTriangle, AlertOctagon } from 'lucide-react';

// Цвет и подпись по статусу ответа сервера: healthy / success / insufficient_data.
function statusTone(status) {
  if (status === 'healthy') {
    return { label: 'Без признаков', badge: 'bg-emerald-100 text-emerald-800 border-emerald-300', icon: CheckCircle2, iconColor: 'text-emerald-700' };
  }
  if (status === 'success') {
    return { label: 'Вероятна болезнь', badge: 'bg-rose-100 text-rose-800 border-rose-300', icon: AlertOctagon, iconColor: 'text-rose-700' };
  }
  return { label: 'Нужно уточнение', badge: 'bg-amber-100 text-amber-800 border-amber-300', icon: AlertTriangle, iconColor: 'text-amber-700' };
}

export default function StatusBadge({ status, label, size = 'sm' }) {
  const tone = statusTone(status);
  const Icon = tone.icon;
  const sizing = size === 'sm'
    ? 'px-1.5 py-0.5 rounded text-[10px] gap-1'
    : 'px-2.5 py-1 rounded-full text-xs gap-1.5 border';
  return (
    <span className={`inline-flex items-center font-bold ${sizing} ${tone.badge}`}>
      <Icon className={`${size === 'sm' ? 'w-2.5 h-2.5' : 'w-3.5 h-3.5'} ${tone.iconColor}`} />
      <span>{label || tone.label}</span>
    </span>
  );
}
