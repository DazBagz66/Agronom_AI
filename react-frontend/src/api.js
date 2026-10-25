// Все обращения к серверу собраны здесь, чтобы экраны не знали адресов.
export const API_URL = import.meta.env.VITE_API_URL || '/api';

async function getJson(path) {
  const response = await fetch(`${API_URL}${path}`);
  if (!response.ok) throw new Error(`Код ответа сервера: ${response.status}`);
  return response.json();
}

export const fetchCapabilities = () => getJson('/capabilities');
export const fetchSessions = () => getJson('/sessions?limit=100');
export const fetchSession = (id) => getJson(`/sessions/${id}`);
export const fetchHistory = () => getJson('/history?limit=100');
export const fetchKnowledge = (crop) => getJson(`/knowledge?crop=${encodeURIComponent(crop)}`);

export async function diagnose(payload) {
  const response = await fetch(`${API_URL}/diagnose`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(`Код ответа сервера: ${response.status}`);
  return response.json();
}

// Фото из истории приходит без префикса data:, браузеру он нужен.
export function imageSource(base64) {
  if (!base64) return null;
  return base64.startsWith('data:') ? base64 : `data:image/jpeg;base64,${base64}`;
}
