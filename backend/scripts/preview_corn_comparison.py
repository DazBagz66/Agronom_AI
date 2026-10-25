"""Render the frozen crop annotations without model predictions."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]


def main():
    rows = json.loads((ROOT/'backend/data/datasets/corn_comparison_v1/crop_annotations.json').read_text(encoding='utf-8'))['records']
    destination = ROOT/'.cache/datasets/corn_comparison_v1/crop_review'
    destination.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(rows),12):
        subset = rows[start:start+12]
        canvas = Image.new('RGB',(1200,((len(subset)+3)//4)*324),'white')
        draw = ImageDraw.Draw(canvas)
        for offset,row in enumerate(subset):
            with Image.open(ROOT/row['path']) as raw:
                image = ImageOps.contain(ImageOps.exif_transpose(raw).convert('RGB'),(296,296))
            x,y = (offset%4)*300,(offset//4)*324
            canvas.paste(image,(x,y))
            a,b,c,d = row['bbox_xyxy']
            draw.rectangle((x+a*image.width,y+b*image.height,x+c*image.width-1,y+d*image.height-1),outline='red',width=3)
            draw.text((x+2,y+300),f'{start+offset}: {row["source"]}',fill='black')
        canvas.save(destination/f'boxes_{start//12}.jpg')
    print(destination)


if __name__=='__main__':
    main()
