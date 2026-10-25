import { useRef, useState } from 'react';
import { Camera, Upload, Send, Sparkles, X, ImagePlus } from 'lucide-react';

// Предпросмотр выбранного фото с подсказкой по размеру кадра.
function PhotoPreview({ image, name, onRemove }) {
  const [size, setSize] = useState(null);
  const small = size && Math.min(size.width, size.height) < 600;
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-3 space-y-2 text-xs shadow-xs">
      <div className="flex items-start gap-3">
        <img
          src={image}
          alt="Прикреплённое фото целиком"
          className="max-h-40 max-w-[45%] object-contain rounded-lg bg-slate-100"
          onLoad={(event) => setSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })}
        />
        <div className="min-w-0 flex-1 space-y-1.5">
          <p className="font-bold text-emerald-800 break-all m-0">{name || 'Фото прикреплено'}</p>
          {size && (
            <p className={`m-0 ${small ? 'text-amber-800' : 'text-slate-600'}`}>
              {size.width} × {size.height} пикселей.
              {small
                ? ' Желательно фото крупнее: от 600 пикселей по короткой стороне. Отправить это фото тоже можно.'
                : ' Разрешения достаточно; проверьте резкость и размер объекта.'}
            </p>
          )}
        </div>
        <button type="button" aria-label="Убрать фото" onClick={onRemove} className="p-1 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-rose-50">
          <X className="w-4 h-4" />
        </button>
      </div>
      <p className="text-slate-700 m-0">Ориентир: один лист или початок занимает примерно ⅔–⅘ длины кадра. Если он мелкий — поднесите камеру ближе.</p>
    </div>
  );
}

// Ввод наблюдения: фото с камеры или из файла и текстовое описание.
// variant "hero" — крупная карточка для новой проверки, "compact" — строка для дополнения сессии.
export default function Composer({ variant = 'hero', busy, placeholder, onSubmit }) {
  const [text, setText] = useState('');
  const [image, setImage] = useState(null);
  const [imageName, setImageName] = useState('');
  const cameraRef = useRef(null);
  const fileRef = useRef(null);

  const readFile = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => { setImage(reader.result); setImageName(file.name); };
    reader.readAsDataURL(file);
    event.target.value = '';
  };

  const submit = async (event) => {
    event.preventDefault();
    if (busy || (!text.trim() && !image)) return;
    const ok = await onSubmit({ text: text.trim(), image });
    if (ok) { setText(''); setImage(null); setImageName(''); }
  };

  const inputs = (
    <>
      <input type="file" accept="image/*" capture="environment" ref={cameraRef} onChange={readFile} className="hidden" aria-label="Снять фото растения" />
      <input type="file" accept="image/*" ref={fileRef} onChange={readFile} className="hidden" aria-label="Загрузить фото растения" />
    </>
  );

  if (variant === 'compact') {
    return (
      <form onSubmit={submit} className="space-y-2">
        {inputs}
        {image && <PhotoPreview key={image} image={image} name={imageName} onRemove={() => setImage(null)} />}
        <div className="flex items-center gap-2 bg-white border border-slate-200 rounded-2xl px-2 py-1.5 shadow-xs focus-within:ring-2 focus-within:ring-emerald-500">
          <button type="button" onClick={() => fileRef.current?.click()} title="Добавить фото" className="p-2 rounded-xl text-emerald-700 hover:bg-emerald-50 shrink-0">
            <ImagePlus className="w-5 h-5" />
          </button>
          <input
            type="text"
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder={placeholder}
            className="w-full min-w-0 bg-transparent text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none"
          />
          <button
            type="submit"
            aria-label="Отправить"
            disabled={busy || (!text.trim() && !image)}
            className="p-2 rounded-xl bg-emerald-700 text-white hover:bg-emerald-800 disabled:bg-slate-200 disabled:text-slate-400 shrink-0 transition"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </form>
    );
  }

  return (
    <form onSubmit={submit} className="space-y-3">
      {inputs}
      <div className="bg-gradient-to-br from-emerald-600 to-emerald-800 text-white rounded-2xl p-5 shadow-lg shadow-emerald-900/10 relative overflow-hidden">
        <div className="absolute -right-4 -bottom-4 w-32 h-32 bg-emerald-500/20 rounded-full blur-xl pointer-events-none" />
        <div className="relative z-10">
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/30 text-emerald-100 text-xs font-semibold mb-2">
            <Sparkles className="w-3.5 h-3.5" />
            <span>ИИ-анализ растений</span>
          </span>
          <h2 className="text-xl font-extrabold leading-tight tracking-tight text-white m-0">Проверьте растение по фотографии</h2>
          <p className="text-xs text-emerald-100 mt-1 leading-relaxed m-0">
            Снимите лист, початок или корзинку крупным планом или опишите симптомы словами. Результат предварительный.
          </p>
          <div className="grid grid-cols-2 gap-2.5 mt-4">
            <button
              type="button"
              onClick={() => cameraRef.current?.click()}
              className="flex items-center justify-center gap-2 py-3 px-3 bg-white text-emerald-800 rounded-xl font-bold text-xs shadow-sm hover:bg-emerald-50 active:scale-95 transition min-h-[48px]"
            >
              <Camera className="w-4 h-4 text-emerald-700" />
              <span>Камера</span>
            </button>
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              className="flex items-center justify-center gap-2 py-3 px-3 bg-emerald-700/80 border border-emerald-400/40 text-white rounded-xl font-bold text-xs hover:bg-emerald-700 active:scale-95 transition min-h-[48px]"
            >
              <Upload className="w-4 h-4 text-emerald-200" />
              <span>Загрузить</span>
            </button>
          </div>
        </div>
      </div>

      {image && <PhotoPreview key={image} image={image} name={imageName} onRemove={() => setImage(null)} />}

      <div className="bg-white border border-slate-200 rounded-2xl p-3 shadow-xs space-y-2">
        <label className="block text-xs font-bold text-slate-800">
          Описание симптомов
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            rows={3}
            placeholder={placeholder}
            className="mt-1.5 block w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-normal text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent resize-y"
          />
        </label>
        <button
          type="submit"
          disabled={busy || (!text.trim() && !image)}
          className="w-full flex items-center justify-center gap-2 py-3 bg-emerald-700 text-white rounded-xl font-bold text-sm hover:bg-emerald-800 disabled:bg-slate-200 disabled:text-slate-400 transition min-h-[48px]"
        >
          <Send className="w-4 h-4" />
          <span>Отправить на анализ</span>
        </button>
      </div>
    </form>
  );
}
