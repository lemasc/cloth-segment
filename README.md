# cloth-segment (โค้ดสำหรับส่ง)

โค้ดหลักสำหรับเทรนและประเมินโมเดลแยกส่วนของเสื้อผ้าในภาพคน (human parsing) ประกอบด้วย
U-Net (ResNet-34), DeepLabV3+ (ResNet-50) และ SegFormer-B2 รวมถึง SegFormer-B2 แบบ
multi-task ที่ทำนาย attribute ของเสื้อผ้าจากชุดข้อมูล DeepFashion-MultiModal ไปพร้อมกัน

ทุกโมเดลใช้ label set 12 คลาสชุดเดียวกัน, data pipeline เดียวกัน, ลูปเทรนเดียวกัน
และวิธีประเมินเดียวกัน แพ็กเกจนี้ **ไม่มี** weights, ชุดข้อมูล หรือผลการทดลอง

## โครงสร้างโค้ด

จุดเริ่มต้นสำหรับสั่งรัน (entry point) อยู่ใน [`src/cloth_segment/`](src/cloth_segment/)
ส่วนโค้ดที่ใช้ร่วมกันอยู่ใน [`src/cloth_segment/lib/`](src/cloth_segment/lib/)

| ไฟล์ / โฟลเดอร์ | หน้าที่ |
|---|---|
| [`train.py`](src/cloth_segment/train.py) | entry point และลูปเทรน รันด้วย `python -m cloth_segment.train <config>` |
| [`evaluate.py`](src/cloth_segment/evaluate.py) | entry point การประเมิน: confusion matrix, mIoU, Dice |
| [`attr_eval.py`](src/cloth_segment/attr_eval.py) | entry point ประเมิน attribute (เฉพาะโมเดล multi-task) |
| [`lib/data/labels.py`](src/cloth_segment/lib/data/labels.py) | label set ร่วมและการแปลง label ดิบของแต่ละชุดข้อมูล |
| [`lib/data/sources.py`](src/cloth_segment/lib/data/sources.py) | ที่อยู่ของภาพและ mask ของแต่ละชุดข้อมูล รวมถึงความหมายของ label ดิบ |
| [`lib/data/splits.py`](src/cloth_segment/lib/data/splits.py) | กติกาแบ่ง train / val / test และการโหลด split manifest |
| [`lib/data/dataset.py`](src/cloth_segment/lib/data/dataset.py) | `Dataset` และ sampler |
| [`lib/data/transforms.py`](src/cloth_segment/lib/data/transforms.py) | augmentation สำหรับเทรน และ transform สำหรับประเมิน |
| [`lib/models/`](src/cloth_segment/lib/models/) | ตัวลงทะเบียนโมเดล และโมเดลทั้งสี่: [`unet.py`](src/cloth_segment/lib/models/unet.py), [`deeplab.py`](src/cloth_segment/lib/models/deeplab.py), [`segformer.py`](src/cloth_segment/lib/models/segformer.py), [`multitask.py`](src/cloth_segment/lib/models/multitask.py) |
| [`lib/attributes.py`](src/cloth_segment/lib/attributes.py) | attribute head: การกำหนดค่า, loss และ metric |
| [`lib/runs.py`](src/cloth_segment/lib/runs.py) | รวม config, คำนวณ hash, สร้างโฟลเดอร์ run, เช็คว่าเคยรันแล้วหรือยัง, GPU lock |
| [`lib/hardware.py`](src/cloth_segment/lib/hardware.py) | บันทึกข้อมูล GPU ที่ใช้ใน `meta.json` |
| [`lib/paths.py`](src/cloth_segment/lib/paths.py) | ตำแหน่งของ `datasets/`, `data/`, `runs/` และ `configs/` |
| [`configs/`](configs/) | [`base.yaml`](configs/base.yaml) (ค่าที่ใช้ร่วมกัน) และไฟล์เล็ก ๆ ของแต่ละโมเดล |
| [`tests/`](tests/) | unit test ทั้งหมด ไม่ต้องใช้ชุดข้อมูลจริงและไม่ต้องมี GPU |

## การติดตั้ง

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest
```

## โครงสร้างข้อมูล

โค้ดอ่านและเขียนสามโฟลเดอร์ที่ root ของโปรเจกต์ (สร้างเอง หรือทำ symlink ไปยังที่เก็บอื่นก็ได้)

- `datasets/` เก็บชุดข้อมูลดิบที่ดาวน์โหลดมา: LIP, CIHP และ DeepFashion-MultiModal
- `data/` เก็บ mask ที่แปลงเป็น label set ร่วมแล้ว (`data/masks/...`) และ split manifest
  (`data/splits/<version>/`) ค่า `data.version` ใน [`configs/base.yaml`](configs/base.yaml)
  ต้องตรงกับโฟลเดอร์ที่มีอยู่จริง
- `runs/` เก็บผลลัพธ์ หนึ่งโฟลเดอร์ต่อหนึ่ง run

สคริปต์ที่สร้าง `data/` (แปลง mask, หาภาพซ้ำ, แบ่ง split) **ไม่ได้อยู่ในแพ็กเกจนี้**
แต่อยู่ใน repository เต็มของโปรเจกต์

## เทรนและประเมิน

```bash
uv run python -m cloth_segment.train configs/segformer_b2.yaml
uv run python -m cloth_segment.evaluate <โฟลเดอร์ run>              # test split
uv run python -m cloth_segment.evaluate <โฟลเดอร์ run> --split val
uv run python -m cloth_segment.attr_eval <โฟลเดอร์ run>             # เฉพาะ run แบบ multi-task
```

config ของแต่ละโมเดล
([`unet_r34`](configs/unet_r34.yaml), [`deeplabv3p_r50`](configs/deeplabv3p_r50.yaml),
[`segformer_b2`](configs/segformer_b2.yaml), [`segformer_b2_mt`](configs/segformer_b2_mt.yaml))
แก้เฉพาะส่วนของโมเดล ส่วนข้อมูล ขนาดภาพ augmentation และงบการเทรนมาจาก
[`base.yaml`](configs/base.yaml) และใช้ร่วมกันทุกโมเดล

ถ้า run ที่มี config hash เดียวกันเทรนเสร็จแล้ว จะไม่เทรนซ้ำ และถ้ามี GPU โค้ดจะใช้ GPU เอง
ถ้าไม่มีจะใช้ CPU (ช้ามาก ใช้ได้แค่ทดสอบว่าโค้ดรันผ่าน)
