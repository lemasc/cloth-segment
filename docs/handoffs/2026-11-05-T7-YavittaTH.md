# Handoff — T7 Model B: DeepLabV3+ (ResNet-50) (YavittaTH)

## In plain words
เพิ่มโมเดล DeepLabV3+ ที่ใช้ encoder ResNet-50 พร้อม config และ test ยังไม่ได้เทรน
ใช้ `train.py` เรียกได้แล้วด้วย `configs/deeplabv3p_r50.yaml`
ขั้นต่อไปคือ X3: เทรน 40k (ต้องมีคำสั่งจากผู้ใช้ก่อนใช้ GPU)

## Summary
- `src/cloth_segment/models/deeplab.py`: `@register("deeplabv3p_r50")` ใช้
  `smp.DeepLabV3Plus("resnet50", encoder_output_stride=16, ...)` ห่อด้วย
  `PadToMultiple(16)` คืน logits `[B, 12, H, W]` ไม่มี softmax
- `models/__init__.py`: import `deeplab` และลบ `deeplabv3p_r50` ออกจาก `PLANNED`
- `configs/deeplabv3p_r50.yaml`: แก้แค่ `model.*` (`name`, `pretrained: true`)
- `tests/test_models.py`: test ใหม่ 7 ตัว (ขนาด output, เป็น logits, ResNet-50 +
  output stride 16, backward 1 step, config) รันบน CPU ไม่โหลด weight
- `tests/test_train.py`: `test_build_model_errors` ใช้ชื่อโมเดลปลอมใน `PLANNED`
  แทน `deeplabv3p_r50` T8 จึงไม่ต้องแก้ test นี้อีก

## Decisions and why
- **lr = 1e-4 (ค่าจาก base):** เท่ากับ U-Net (X2) ยังไม่ได้ปรับจูน ไม่มีเวลา GPU พอ
  ถ้าจูนแค่บางโมเดลจะเทียบไม่ยุติธรรม ต้องเขียนเป็นข้อจำกัดใน T10
- **output stride 16 (ค่า default ของ smp):** output stride คือภาพเล็กลงกี่เท่าก่อนเข้า
  ASPP GPU probe ของ T5 วัดหน่วยความจำ batch 8 ด้วยค่านี้ ค่า 8 อาจได้ขอบละเอียดกว่า
  แต่กินหน่วยความจำมากกว่า และยังไม่มีใครวัด
- **output stride เป็นค่าคงที่ในโค้ด ไม่อยู่ใน config:** ทำแบบเดียวกับ U-Net ชื่อ
  `deeplabv3p_r50` หมายถึงสถาปัตยกรรมเดียว ถ้าจะลอง stride 8 ควรเป็นชื่อโมเดลใหม่
- **ค่าอื่นของ decoder ใช้ default ของ smp:** 256 channels, atrous rates 12/24/36

## Results / findings
ยังไม่มีผลการเทรน จำนวนพารามิเตอร์ (`count_parameters`, 12 คลาส):

| โมเดล | encoder | decoder | head | รวม |
|---|---|---|---|---|
| `unet_r34` | 21,284,672 | 3,151,552 | 1,740 | 24,437,964 |
| `deeplabv3p_r50` | 23,508,032 | 3,169,296 | 3,084 | 26,680,412 |

ต่างกันแค่ประมาณ 2.2 ล้าน เกือบทั้งหมดมาจาก encoder (ResNet-50 ใช้ bottleneck block
จึงมีพารามิเตอร์มากกว่า ResNet-34 ไม่มาก) decoder ขนาดใกล้กัน

## Gotchas
- `PadToMultiple` จำเป็นจริงสำหรับโมเดลนี้: smp 0.6.0 ไม่รับ H หรือ W ที่หาร 16
  ไม่ลงตัว (U-Net รับได้ จึงเป็นแค่การป้องกันเพิ่ม) pipeline ส่ง 512×512 อยู่แล้ว
- ASPP มี batch norm ใน pooling branch เทรนด้วย batch 1 ไม่ได้
  (`drop_last=True` ของ T5 กันไว้แล้ว)
- รันจริงครั้งแรกจะดาวน์โหลด weight ResNet-50 ImageNet ของ smp

## Open items
- ตามเงื่อนไขของ M7: ถ้า X3 `best.pt` อยู่ที่ step 40000 อีก ให้แจ้ง lemasc

## Next
ปลดล็อก X3 (DeepLabV3+ R50, 40k) ใช้คำสั่ง
`uv run python -m cloth_segment.train configs/deeplabv3p_r50.yaml`
ตาม CONTRIBUTING section 7 (GPU lock) ใช้ `scripts/accessory_by_raw_class.py`
กับ run นี้ได้ (M6)
