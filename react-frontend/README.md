# Интерфейс «Помощник агронома»

React 19 + Vite + Tailwind CSS 4, иконки lucide-react. Оформление повторяет
макет agro-helper-test: экран-«телефон» по центру на широком мониторе, шапка,
нижняя навигация с тремя вкладками, светлая палитра emerald/slate.

```powershell
npm.cmd ci            # зависимости
npm.cmd run dev       # разработка, запросы /api идут на http://127.0.0.1:8000
npm.cmd run build     # сборка в dist/, её отдаёт FastAPI
npm.cmd run lint      # oxlint
node --test           # проверки diagnosis.js, crops.js, assessment.js
```

## Экраны

- **Диагностика** (`components/Scanner.jsx`, `Composer.jsx`): фото с камеры или
  из файла, описание симптомов, культура, фаза и часть растения. Пока сервер
  отвечает, показывается `ScanningOverlay.jsx`.
- **Сессия** (`components/SessionView.jsx`): наблюдения и заключения
  `DiagnosisResult.jsx` с разделами «Вывод», «Гипотезы», «Что делать»,
  «Справочник»; внизу поле для дополнения той же сессии. Для кукурузы с фото
  доступно уточнение диагноза (`DiagnosisFeedback.jsx`).
- **Справочник** (`components/CropCatalog.jsx`): карточки `/api/knowledge` и
  классы установленных моделей из `/api/capabilities`.
- **Полевой журнал** (`components/FieldJournal.jsx`): сессии `/api/sessions`
  с фильтром по культуре; открытие сессии переводит на вкладку диагностики.

Адрес `#catalog`, `#journal` или `#session=<id>` открывает нужный экран сразу.
Все обращения к серверу собраны в `src/api.js`; порог отображения гипотез и
заголовки заключений — в `src/diagnosis.js`.
