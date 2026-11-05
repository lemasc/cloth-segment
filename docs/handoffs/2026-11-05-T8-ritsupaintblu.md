# Handoff — T8 Model C: SegFormer-B2 (ritsupaintblu)

## In plain words
เพิ่มโมเดล SegFormer-B2 พร้อม config และ test ยังไม่ได้เทรน
ใช้ smp (ไลบรารีเดียวกับ U-Net และ DeepLabV3+) และตั้งค่าให้ตรงกับ SegFormer ต้นฉบับใน 3 จุดที่ค่า default ของ smp ไม่ตรง
ขั้นต่อไปคือ X4: รัน GPU probe ซ้ำด้วย builder ตัวจริง แล้วเทรน 40k (ต้องมีคำสั่งจากผู้ใช้ก่อนใช้ GPU)

## Summary
- `src/cloth_segment/models/segformer.py`: `@register("segformer_b2")` ใช้
  `smp.Segformer("mit_b2", decoder_segmentation_channels=768, upsampling=1)`
  แทรก `Dropout2d` หน้า classifier แล้วห่อด้วย `PadToMultiple(32)`
  คืน logits `[B, 12, H, W]` ไม่มี softmax
- `models/__init__.py`: import `segformer` และทำให้ `PLANNED` ว่าง (ไม่มีโมเดลที่รอทำแล้ว)
- `configs/segformer_b2.yaml`: แก้แค่ `model.*` (`name`, `pretrained: true`, `classifier_dropout: 0.1`)
- `tests/test_models.py`: test ใหม่ 8 ฟังก์ชัน 11 กรณี (ขนาด output, เป็น logits, สถาปัตยกรรม B2,
  การขยายแบบ `align_corners=False`, dropout ไม่เปลี่ยน weight และ key, ค่า dropout ผิด,
  backward 1 step, config) รันบน CPU ไม่โหลด weight

## Decisions and why
- **smp ไม่ใช่ Hugging Face (HF):** encoder MiT-B2 เหมือนกันทุกอย่าง (24.2M พารามิเตอร์)
  แต่ smp ทำให้ทั้ง 3 โมเดลมาจากไลบรารีเดียวกัน ใช้คลาสฐาน วิธี init และ `PadToMultiple` เดียวกัน
  ความต่างที่วัดได้จึงมาจากสถาปัตยกรรม ไม่ใช่จากไลบรารี (D5) อีกทั้ง smp pin revision
  ของ weight ไว้แล้ว และคืน tensor ตรงๆ ไม่ต้องมี wrapper
- **ห้ามใช้ checkpoint ที่ fine-tune มาแล้ว** เช่น `nvidia/segformer-b2-finetuned-ade-512-512`
  หรือ `mattmdjaga/segformer_b2_clothes` เพราะผิด D5 (ใช้แค่ encoder ImageNet) และตัว clothes เทรนบน ATR (D2)
- **กฎที่ใช้: ถ้าค่า default ของ smp ไม่ตรงกับต้นฉบับ ให้ตั้งให้ตรง** ต้นฉบับคือ
  `NVlabs/SegFormer` (`local_configs/segformer/B2/segformer.b2.512x512.ade.160k.py`
  และ `mmseg/models/decode_heads/decode_head.py`) อ่านจากซอร์สจริง:

  | จุด | smp default | ต้นฉบับ | T8 ใช้ |
  |---|---|---|---|
  | ความกว้าง decoder | 256 | 768 | 768 |
  | dropout หน้า classifier | ไม่มี | `nn.Dropout2d(0.1)` | `nn.Dropout2d(0.1)` |
  | `align_corners` ตอนขยาย ×4 | `True` | `False` | `False` |

- **`Dropout2d` ไม่ใช่ `Dropout`:** `Dropout2d` ปิดทั้ง channel ทุก pixel ของภาพนั้น
  ส่วน `Dropout` ปิดทีละค่า HF ใช้ `Dropout` จึงไม่ตรงกับต้นฉบับในข้อนี้ ผลต่อ mIoU ยังไม่ได้วัด
  dropout ทำงานแค่ตอนเทรน ตอน eval ไม่มีผล
- **`align_corners=False` ผ่าน `upsampling=1`:** smp ไม่มี parameter ให้ตั้ง `align_corners`
  จึงให้ smp คืน logits ขนาด 1/4 แล้วให้ `PadToMultiple` ขยาย ซึ่งใช้ `align_corners=False` อยู่แล้ว
  test เช็กว่าได้ค่าเท่ากับ `F.interpolate(..., align_corners=False)` ทุกค่า
- **ใส่ `nn.Identity()` แทนเมื่อ `classifier_dropout: 0`:** ถ้าไม่ใส่อะไรเลย key ของ classifier
  ใน state_dict จะเลื่อนจาก `segmentation_head.1` เป็น `.0` และ checkpoint จะโหลดข้ามกันไม่ได้
- **lr = 1e-4 (ค่าจาก base):** เท่ากับ U-Net (X2) และ DeepLabV3+ (T7) ต้นฉบับใช้ lr 6e-5,
  head lr ×10 และไม่ใช้ weight decay กับ norm และ `pos_block` สองอย่างหลังต้องแบ่ง param group
  ใน `train.py` ซึ่งเป็นโค้ดร่วมของทุกโมเดล จึงไม่ทำ ต้องเขียนเป็นข้อจำกัดใน T10
  ค่าอื่นของ optimizer ตรงกับต้นฉบับอยู่แล้ว (AdamW betas, weight decay 0.01, warmup 1500, poly 1.0)

## Results / findings
ยังไม่มีผลการเทรน จำนวนพารามิเตอร์ (`count_parameters`, 12 คลาส):

| โมเดล | encoder | decoder | head | รวม |
|---|---|---|---|---|
| `unet_r34` | 21,284,672 | 3,151,552 | 1,740 | 24,437,964 |
| `deeplabv3p_r50` | 23,508,032 | 3,169,296 | 3,084 | 26,680,412 |
| `segformer_b2` | 24,196,288 | 3,150,336 | 9,228 | 27,355,852 |

(สองแถวแรกมาจาก handoff ของ T7) ยอดรวมของ `segformer_b2` เท่ากับ 27.4M ที่ GPU probe ของ T5 วัดจากเวอร์ชัน HF

เช็กบน CPU ด้วย config จริง (`load_config` + `seeded_model`) แล้ว:
- weight ImageNet โหลดเข้า encoder ครบ (332/332 tensor ไม่เท่ากับค่าสุ่ม)
- seed เดียวกันได้ weight เริ่มต้นเท่ากันทุกตัว รวม decoder ที่สุ่ม
- input 512×512 ได้ output `(1, 12, 512, 512)`

**`align_corners` มีผลแค่ไหน:** วัดด้วย proxy ไม่ได้เทรน ใช้ mask จาก val ชุดละ 300 ภาพ
สร้าง "prediction ที่สมบูรณ์ขนาด 1/4" (เฉลี่ย one-hot ของ mask ทีละ block 4×4)
แล้วขยายกลับด้วย `True` และ `False` ผลคือ argmax เปลี่ยนประมาณ 0.8–1.1% ของ pixel ส่วนใหญ่อยู่ที่ขอบ class
แต่ mIoU ขยับได้ 0.5–4 จุด แล้วแต่ dataset และขึ้นกับว่าสมมติให้ feature อยู่ตรงไหน
(class เล็กและบางเสีย IoU ได้มาก) ตัวเลขนี้จึงบอกไม่ได้ว่าแบบไหนดีกว่า
บอกได้แค่ว่าเรื่องนี้ใหญ่พอๆ กับความต่างระหว่างโมเดลที่เราจะเทียบ จึงตั้งให้ตรงกับต้นฉบับ
สคริปต์นี้ไม่ได้ commit เพราะไม่ใช่ส่วนของ pipeline

## Gotchas
- **HF token ที่หมดอายุทำให้ดาวน์โหลด weight ไม่ได้:** ถ้ามี `~/.cache/huggingface/token`
  ที่หมดอายุ `huggingface_hub` จะส่ง token ไปด้วยแล้วได้ 401 ("OAuth token has expired")
  ทั้งที่ repo `smp-hub/mit_b2.imagenet` เป็น public แก้ได้ 2 แบบ: login ใหม่ หรือรันด้วย
  `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` (ไม่ส่ง token) เกิดกับ Linux user `whats` บนเครื่องทีม
- weight อยู่ใน `~/.cache/huggingface/hub/models--smp-hub--mit_b2.imagenet` (95 MB)
  แยกตาม Linux user คนที่รัน X4 ด้วย user อื่นจะต้องดาวน์โหลดเอง
- `PadToMultiple` จำเป็นจริง: smp 0.6.0 ไม่รับ H หรือ W ที่หาร 32 ไม่ลงตัว
  ต่างจาก HF ที่รับได้ทุกขนาด pipeline ส่ง 512×512 อยู่แล้ว
- smp คืน logits 1/4 (128×128 ที่ input 512) ตัว wrapper ขยายเป็น 512 ก่อนส่งให้ `train.py`
  ไม่ต้องพึ่ง `logits_at`

## Open items
- **GPU probe ของ T5 วัดจากเวอร์ชัน HF ไม่ใช่ builder นี้** จำนวนพารามิเตอร์เท่ากัน
  และ loop เทรนขยาย logits เป็น 512 เหมือนกัน หน่วยความจำจึงน่าจะใกล้กัน แต่ยังไม่ได้วัด
  batch 8 เหลือที่ว่างแค่ประมาณ 2.4 GB ควรวัดซ้ำก่อนเริ่ม X4 (ใช้ GPU)
- ผลของการใช้ lr ร่วม 1e-4 แทนสูตรต้นฉบับยังไม่รู้ ต้องเขียนเป็นข้อจำกัดใน T10

## Next
ปลดล็อก X4 (SegFormer-B2, 40k) ใช้คำสั่ง
`uv run python -m cloth_segment.train configs/segformer_b2.yaml`
ตาม CONTRIBUTING section 7 (GPU lock)
