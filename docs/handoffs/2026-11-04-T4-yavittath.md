# Handoff — T4 Data loader and augmentation (yavittath)

## In plain words
มี data loader สำหรับ 3 dataset แล้ว คืนภาพ 512×512, mask และ attribute ของ DFMM
ในทุก sample ค่า input/augmentation ที่ทุกโมเดลต้องใช้ร่วมกัน (D5) อยู่ใน
`configs/base.yaml` แล้ว ขั้นต่อไปคือ T5 (training loop) ซึ่งใช้ `load_split()` ได้ทันที

## Summary
- `src/cloth_segment/data/transforms.py`: transform ของ train และ eval
  - **Train:** ปรับด้านยาวเป็น `512 × s` (s สุ่มจาก [0.5, 2.0]) → flip 50% →
    สุ่มตัดหน้าต่าง 512×512 → color jitter (brightness/contrast/saturation 0.2)
    → ImageNet normalize ถ้าด้านไหนสั้นกว่า 512 จะวางภาพที่ตำแหน่งสุ่มบนพื้น pad
  - **Eval:** ปรับด้านยาวเป็น 512 แล้ว pad ด้านล่าง/ขวา คืน `valid_hw` (ส่วนที่ไม่ใช่
    pad) และ `orig_hw` เพื่อให้ T5/T9 ตัด logits แล้วขยายกลับไปเท่า mask จริง
  - ส่วนที่ pad: ภาพ = 0 (หลัง normalize คือค่าเฉลี่ย), mask = 255 (ignore)
  - mask ใช้ nearest เท่านั้น (`NEAREST_EXACT`)
- `src/cloth_segment/data/dataset.py`
  - `load_split(cfg, split)`: สร้าง dataset จาก config และ **error ถ้า
    `data.version` ไม่ตรงกับ manifest ที่ build ไว้**
  - `SegDataset`: อ่านภาพต้นฉบับจาก `datasets/` และ mask จาก `data/masks/labels-v1/`
    แปลงภาพ grayscale เป็น RGB error ถ้าค่า mask แปลก หรือขนาดภาพ/mask ไม่ตรงกัน
  - `attribute_targets()`: attribute ของ DFMM ครบทั้ง 18 ตัว ค่า NA และ -1 → 255
    ภาพจาก LIP/CIHP ได้ 255 ทั้งหมด
  - `collate`, `make_sampler` (`natural` | `balanced`), `class_pixel_counts`
- `configs/base.yaml`: `data.version`, `data.sampling`, `input.size`, `augment.*`
- `tests/test_dataset.py`: 22 test ใช้ไฟล์สังเคราะห์เล็ก ๆ ไม่ต้องใช้ข้อมูลจริงหรือ GPU

## Decisions and why
- **512×512:** ค่ามาตรฐานของงาน segmentation หาร 32 ลงตัว (SegFormer ต้องการ)
  ยังไม่ได้วัดว่า GPU 12 GB รับ batch ได้เท่าไร ทำใน T5
- **ใช้ resize ด้านยาว + crop/pad แทนการบีบเป็นสี่เหลี่ยม:** ภาพแต่ละ dataset
  มีสัดส่วนต่างกันมาก (LIP แนวตั้งเล็ก, CIHP แนวนอน, DFMM แนวตั้งใหญ่)
  การบีบจะทำให้รูปทรงเสื้อผ้าผิดไป
- **Scale [0.5, 2.0]:** ช่วงมาตรฐาน (เช่นใน mmsegmentation)
- **Sampling `natural`:** ใช้สัดส่วนตามธรรมชาติ (LIP 41%, CIHP 43%, DFMM 16%)
  เป็นค่าเริ่มต้นของการเปรียบเทียบหลัก เพราะง่ายที่สุด `balanced` มีให้ใช้ใน E2
- **ใช้ `torchvision.transforms.v2`** ไม่เพิ่ม dependency ใหม่
- **ไม่ทำ hue jitter, rotation:** ยังไม่มีเหตุผลที่ต้องใช้ ปรับใน config ได้
- **Attribute:** loader คืนรหัสดิบทุกตัว แต่ยังไม่เลือก head (Q1 ไว้ทำ E1)
  T1 แนะนำให้ sleeve code 4 ("not_long") เป็น ignore ด้วย ซึ่งเป็นเรื่องของ head เลยไว้ทำใน E1

## Results / findings
ทดสอบกับข้อมูลจริงบน CPU (ไม่ได้เทรน ไม่มี run folder):

| Split | LIP | CIHP | DFMM | รวม |
|---|---|---|---|---|
| train | 25,653 | 26,667 | 10,180 | 62,500 |
| val | 1,358 | 1,420 | 1,228 | 4,006 |
| test | 9,632 | 5,000 | 1,293 | 15,925 |

ตรงกับ `splits-v2-d023252b81d9` (handoff ของ M2)

- ความเร็ว loader ตอน train: ประมาณ 256 ภาพ/วินาที ที่ 8 workers, batch 16
  (รวมเวลาเริ่มต้น วัดครั้งเดียวจาก 41 batch) น่าจะเร็วพอโดยไม่ต้องทำ cache
- สัดส่วน pixel ต่อ class ใน train (%): background 68.5, upper 7.2, skin 5.5,
  outer 4.3, pants 4.0, hair 3.2, face 2.7, dress 2.6, footwear 1.05, skirt 0.41,
  headwear 0.40, accessory 0.16 (จาก `class_pixel_counts`)
- ใน batch แรก ประมาณ 25% ของ pixel เป็น ignore เพราะ scale < 1 ทำให้มีพื้นที่ pad

## Gotchas
- `df.mask` คือ method ของ pandas ต้องใช้ `df["mask"]`
- Augmentation ใช้ global torch RNG ใน worker ถ้าอยากให้ผลซ้ำได้ T5 ต้องส่ง
  `generator` ที่ seed จาก config ให้ `DataLoader`
- `natural` ทำให้ `make_sampler` คืน `None` ต้องใช้ `shuffle=True` แทน

## Open items
- **Freeze contract:** เมื่อ T4 merge แล้ว label set และ splits จะ freeze
  (CONTRIBUTING ข้อ 4) ค่า threshold ของ duplicate ยืนยันแล้วสำหรับ baseline แรก
  (`CORR_MIN` = 0.70, `CORR_MIN_DFMM_PAIR` = 0.98, `splits-v2`) ส่วนสถานะ `frozen`
  ในตาราง CONTRIBUTING ข้อ 4 maintainer เป็นคนแก้เอง
- **Shared comparison fields:** PR นี้ใส่ค่าลงใน `configs/base.yaml` (contract ยังเปิดอยู่)
  ค่าเหล่านี้ยังแก้ได้จนกว่า T5 จะมี baseline run ที่ตกลงกัน
- batch size และจำนวน worker เป็นเรื่องของ T5

## Next
ปลดล็อก T5 (training/eval pipeline), แล้วต่อด้วย T6–T8 และ E1 (attribute heads)
