"""Create an offline, editable review sheet; approving requires reviewer + evidence."""
import argparse
import csv
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from app.corn_taxonomy import CORN_LABELS


def render(source=None,offset=0,limit=100,organ=None):
    queue=ROOT/'backend/data/datasets/corn_field/review_queue.csv'
    with queue.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    unique={}
    for row in rows:
        if (source is None or row['source']==source) and (organ is None or row['organ']==organ):
            unique.setdefault(row['sha256'],row)
    rows=list(unique.values())[offset:offset+limit]
    cards=[]
    for i,row in enumerate(rows):
        uri=(ROOT/row['path']).resolve().as_uri()
        organs='<option value="">Выберите орган</option>'+''.join(
            f'<option value="{key}"'+(' selected' if row.get('organ')==key else '')+f'>{label}</option>'
            for key,label in [('leaf','Лист'),('ear','Початок'),('stem','Стебель'),('whole','Всё растение')])
        cards.append(f'<article><img loading="lazy" src="{html.escape(uri)}" alt="Образец {i+1}">'
            f'<p>{i+1}. {html.escape(row["source"])} — {html.escape(row["raw_label"])}</p>'
            f'<p>Сейчас: {html.escape(row["labels"])}</p>'
            f'<p>Причина проверки: {html.escape(row["reasons"])}</p>'
            f'<label>Метки (через ;)<input data-labels value="{html.escape(row["labels"])}"></label>'
            f'<label>Орган<select data-organ>{organs}</select></label>'
            '<label>Основание решения<input data-evidence placeholder="Заключение специалиста, источник разметки"></label>'
            '<label>Решение<select data-status><option value="pending">Отложить</option><option value="accepted">Принять разметку</option>'
            '<option value="rejected">Исключить фото</option></select></label></article>')
    data=json.dumps(rows,ensure_ascii=False).replace('<','\\u003c')
    vocabulary=html.escape(json.dumps(CORN_LABELS,ensure_ascii=False,indent=2))
    document='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверка фотографий кукурузы</title>
<style>body{font:16px system-ui;margin:24px;background:#f4f5f1;color:#17231c}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}article{background:white;border:1px solid #ccc;padding:14px;border-radius:12px}img{width:100%;height:230px;object-fit:contain}label{display:block;margin:8px 0}input,select{box-sizing:border-box;display:block;width:100%;padding:8px}header{position:sticky;top:0;background:#f4f5f1;padding:12px 0;z-index:1}button{padding:12px;background:#246345;color:white;border:0;border-radius:8px}pre{white-space:pre-wrap}</style>
<header><h1>Проверка фотографий кукурузы</h1><label>Кто проверяет<input id="reviewer" placeholder="Имя и квалификация"></label>
<button id="save">Скачать решения CSV</button><p id="message" role="status"></p></header>
<p>Спорные диагнозы оставляйте отложенными. Внешний вид сам по себе не подтверждает возбудителя. Неизвестная болезнь — отдельный результат, а не здоровый образец.</p>
<details><summary>Допустимые обозначения</summary><pre>VOCABULARY</pre></details><main>CARDS</main>
<script>const rows=ROWS;const vocabulary=LABEL_KEYS;
const quote=value=>'"'+String(value??'').replaceAll('"','""')+'"';
document.querySelector('#save').onclick=()=>{
 const reviewer=document.querySelector('#reviewer').value.trim();const output=[];
 for(const [i,card] of [...document.querySelectorAll('article')].entries()){
  const status=card.querySelector('[data-status]').value;if(status==='pending')continue;
  const evidence=card.querySelector('[data-evidence]').value.trim();
  if(!reviewer||!evidence){document.querySelector('#message').textContent='Укажите проверяющего и основание для каждого принятого решения.';return;}
  const labels=card.querySelector('[data-labels]').value.trim().split(';').map(x=>x.trim());
  if(status==='accepted'&&(!card.querySelector('[data-organ]').value||labels.some(x=>!vocabulary.includes(x)))){
   document.querySelector('#message').textContent='Для принятой записи выберите орган и метки из справочника.';return;
  }
  output.push([rows[i].sha256,labels.join(';'),card.querySelector('[data-organ]').value,reviewer,evidence,status].map(quote).join(','));
 }
 const content='sha256,labels,organ,reviewer,evidence,status\\n'+output.join('\\n')+'\\n';
 const url=URL.createObjectURL(new Blob(['\\ufeff'+content],{type:'text/csv;charset=utf-8'}));
 const link=document.createElement('a');link.href=url;link.download='corn-review-decisions.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 document.querySelector('#message').textContent='Сохранено решений: '+output.length;
};</script></html>'''
    document=document.replace('VOCABULARY',vocabulary).replace('CARDS',''.join(cards)).replace('ROWS',data).replace('LABEL_KEYS',json.dumps(list(CORN_LABELS)))
    output=ROOT/'.cache/datasets/corn_field/review.html'
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(document,encoding='utf-8')
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source');parser.add_argument('--offset',type=int,default=0);parser.add_argument('--limit',type=int,default=100)
    parser.add_argument('--organ', choices=['leaf','ear','stem','whole'])
    args=parser.parse_args()
    if args.offset<0 or not 1<=args.limit<=500:parser.error('Invalid offset or limit')
    render(args.source,args.offset,args.limit,args.organ)
