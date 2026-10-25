// На широком экране приложение показывается как телефон по центру,
// на мобильном занимает весь экран.
export default function MobileFrame({ children }) {
  return (
    <div className="flex justify-center min-h-dvh bg-slate-100 text-slate-900 selection:bg-emerald-100 selection:text-emerald-900">
      <div className="w-full sm:max-w-md min-h-dvh bg-white sm:shadow-2xl sm:border-x sm:border-slate-200/90 flex flex-col relative">
        <div className="hidden sm:flex justify-center items-center pt-2 pb-1 bg-white border-b border-slate-100 select-none">
          <div className="w-16 h-1 bg-slate-300 rounded-full" aria-hidden="true" />
        </div>
        <div className="flex-1 flex flex-col w-full relative">{children}</div>
      </div>
    </div>
  );
}
