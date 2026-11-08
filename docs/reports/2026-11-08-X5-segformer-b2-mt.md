# รายงานการเทรน: SegFormer-B2 multi-task + attribute heads (X5)

ผู้เขียน: lemasc

## In plain words
เทรน `segformer_b2_mt` (SegFormer-B2 บวก head ทำนาย attribute รูปทรง 12 ตัวของ DFMM บน encoder ตัวเดียวกัน) ครบ 40,000 iteration บน RTX 4070 ด้วย seed 1
test mIoU (segmentation): LIP **0.6154**, CIHP **0.6305**, DFMM **0.8394** (eval-v2, `best.pt` step 40000)
attribute บน DFMM test: macro-F1 เฉลี่ย **0.8036** เทียบกับการทายคลาสที่พบมากสุด 0.3310
**ยังสรุปไม่ได้ว่า multi-task ช่วยหรือเสีย segmentation** เพราะ run อ้างอิงที่ seed และ GPU เดียวกัน (X6) ยังไม่ได้เทรน
ตัวเลข X4 ที่ใส่ไว้ข้างล่างเป็นแค่ข้อมูลประกอบ (seed 0, GPU คนละรุ่น)

ตัวเลขทุกตัวมาจากไฟล์ใน run folder (`meta.json`, `log.jsonl`, `eval/eval-v2/*/metrics.json`, `eval/attr-v1/dfmm/attr_metrics.json`)
ตัวเลข X4 มาจาก [`2026-11-06-X4-segformer-b2.md`](2026-11-06-X4-segformer-b2.md) ไม่มีค่าประมาณ

## ข้อมูลของการรัน

| รายการ | X5 (run นี้) |
|---|---|
| Run folder | `runs/20261108-1152-2ed6c81-segformer-b2-mt` |
| Commit / dirty | `2ed6c81` / false |
| Config | `configs/segformer_b2_mt.yaml` |
| Config hash | `0d5871d12a05` |
| Data version | `splits-v2-d023252b81d9` |
| ใครรัน / เครื่อง | `metaverse` / `METAVERSE-05` (RTX 4070, 12 GB, driver 560.94) |
| GPU memory cap | 11,094 MiB |
| เริ่ม – จบ | 2026-11-08 11:52:48 – 14:28:37 (+07:00) |
| เวลาเทรน / val ระหว่างเทรน | 8,417.5 / 567.6 วินาที |
| วินาที/iteration | ประมาณ 0.20 – 0.22 (ดู `sec_per_iter` ใน `log.jsonl`) |
| Seed | 1 (`run.overrides_shared: [seed]`) |
| กฎเลือก checkpoint | `dataset_mean_miou` (เฉพาะ segmentation) |
| Checkpoint ที่ถูกเลือก | `best.pt` step 40000 |
| จำนวนพารามิเตอร์ | 27,628,018 (head เพิ่ม 272,166 ตัว) |
| Eval | eval-v2 และ attr-v1, `eval_commit 2ed6c81` |

- shared fields (D5) เหมือน X4 ทุกตัว ต่างกันแค่ `model.*` และ seed
- เทรนบน shared machine โดยให้ `train.py` ถือ GPU lock เอง ไม่มีใครแย่ง GPU ระหว่างรัน
- ไม่มี OOM และไม่มี error ตลอด 40k step

### คำสั่งที่รัน
- เทรน: `uv run python -m cloth_segment.train configs/segformer_b2_mt.yaml` (detach ด้วย `setsid nohup`)
- test segmentation: `uv run python -m cloth_segment.evaluate 20261108-1152-2ed6c81-segformer-b2-mt --wait`
- test attribute: `uv run python -m cloth_segment.attr_eval 20261108-1152-2ed6c81-segformer-b2-mt --wait`
  ทั้งสองคำสั่งจบด้วย exit 0

## ผล: segmentation

### val ระหว่างเทรน
`best.pt` อยู่ที่ step 40000 อีกครั้ง
val `dataset_mean_miou` ตอนจบ = **0.7014** (LIP 0.6203, CIHP 0.6396, DFMM 0.8442)
val ทั้ง 10 ครั้งอยู่ใน `log.jsonl`

### test (`best.pt` step 40000)

| test set | ภาพ | mIoU X5 | mDice X5 | pixel acc X5 | mIoU X4 (ข้อมูลประกอบ) |
|---|---|---|---|---|---|
| LIP | 9,632 | 0.6154 | 0.7482 | 0.8737 | 0.6155 |
| CIHP | 5,000 | 0.6305 | 0.7529 | 0.9056 | 0.6295 |
| DFMM | 1,293 | 0.8394 | 0.9087 | 0.9814 | 0.8425 |

| test mIoU เฉลี่ย 3 ชุด | 12 คลาส | 11 คลาส (ไม่นับ accessory) |
|---|---|---|
| X5 | 0.6951 | 0.7234 |
| X4 (seed 0, RTX 3060) | 0.6959 | 0.7239 |

11 คลาสรายชุดของ X5: LIP 0.6393, CIHP 0.6673, DFMM 0.8637

### test IoU รายคลาส (X5)

| class | LIP | CIHP | DFMM |
|---|---|---|---|
| background | 0.8649 | 0.9315 | 0.9932 |
| hair | 0.7089 | 0.7942 | 0.8677 |
| face | 0.7335 | 0.8595 | 0.9166 |
| skin | 0.7271 | 0.7709 | 0.9152 |
| upper | 0.6982 | 0.6448 | 0.8582 |
| outer | 0.5763 | 0.6166 | 0.7616 |
| dress | 0.3986 | 0.5434 | 0.8644 |
| skirt | 0.3289 | 0.3208 | 0.7063 |
| pants | 0.7391 | 0.6988 | 0.9162 |
| footwear | 0.6018 | 0.5010 | 0.8542 |
| headwear | 0.6547 | 0.6589 | 0.8473 |
| accessory | 0.3526 | 0.2260 | 0.5724 |

## ผล: attribute (DFMM test, `attr-v1`)

หัวข้อ attribute แต่ละตัวมีภาพที่มี label 1,094–1,291 ภาพ
macro-F1 นับเฉพาะคลาสที่มีภาพจริงอย่างน้อย 20 ภาพใน test (`min_count` 20)

| head | ภาพ | accuracy | macro-F1 | accuracy (ทายคลาสหลัก) | macro-F1 (ทายคลาสหลัก) |
|---|---|---|---|---|---|
| sleeve | 1,291 | 0.9357 | 0.7953 | 0.4183 | 0.1475 |
| lower_length | 1,288 | 0.9255 | 0.7779 | 0.5054 | 0.1679 |
| socks | 1,240 | 0.9282 | 0.7265 | 0.8895 | 0.3138 |
| hat | 1,275 | 0.9945 | 0.9884 | 0.8620 | 0.4629 |
| glasses | 1,272 | 0.9969 | 0.9828 | 0.9513 | 0.4875 |
| neckwear | 1,242 | 0.8164 | 0.7872 | 0.6618 | 0.3983 |
| wrist | 1,272 | 0.8994 | 0.8967 | 0.5692 | 0.3627 |
| ring | 1,207 | 0.8003 | 0.7306 | 0.6877 | 0.4075 |
| waist_acc | 1,291 | 0.8552 | 0.6020 | 0.5515 | 0.1777 |
| neckline | 1,094 | 0.7331 | 0.4530 | 0.6188 | 0.1529 |
| outer_cardigan | 1,161 | 0.9733 | 0.9608 | 0.7821 | 0.4389 |
| covers_navel | 1,178 | 0.9686 | 0.9423 | 0.8345 | 0.4549 |
| **เฉลี่ย 12 head** | | **0.9023** | **0.8036** | 0.6943 | 0.3310 |

- val ตอนจบ (`meta.json`, `best_val_attr_mean_macro_f1`) = 0.8187 ส่วน test = 0.8036 ต่างกัน 0.015
- คลาสที่ F1 ต่ำกว่า 0.6 (และมีภาพอย่างน้อย 20 ภาพ):

| head | คลาส | ภาพ | F1 |
|---|---|---|---|
| waist_acc | belt | 42 | 0.0000 |
| neckline | standing | 31 | 0.0000 |
| neckline | v | 142 | 0.1280 |
| sleeve | medium | 65 | 0.3226 |
| socks | socks | 110 | 0.4085 |
| lower_length | three_quarter | 54 | 0.5376 |
| neckline | lapel | 94 | 0.5710 |
| ring | no | 377 | 0.5940 |

## การตีความ
- **ทั้งสองงานเรียนได้ในโมเดลเดียว** segmentation ได้ mIoU ระดับเดียวกับ X4 และ attribute สูงกว่าการทายคลาสหลักทุก head ทั้ง accuracy และ macro-F1
  accuracy อย่างเดียวหลอกได้ (`glasses` ทายคลาสหลักได้ 0.9513 อยู่แล้ว) จึงดู macro-F1 เป็นหลัก
- **head ที่ง่าย** คือ `hat`, `glasses`, `outer_cardigan`, `covers_navel` (macro-F1 ตั้งแต่ 0.94) ส่วน **head ที่ยาก** คือ `neckline` (0.4530) และ `waist_acc` (0.6020)
  คลาส `belt`, `standing` และ `v` แทบทายไม่ถูก ส่วนหนึ่งเพราะมีตัวอย่างน้อยหรือแยกจากรูปทรงยาก (ยังไม่ได้ตรวจสาเหตุ)
- **X5 เทียบกับ X4 ต่างกันไม่เกิน 0.0031 ต่อ test set** (12 คลาสเฉลี่ย −0.0008) แต่เทียบแบบนี้ไม่ได้ตอบคำถามว่า multi-task ช่วยหรือไม่
  เพราะ X4 ใช้ seed 0 บน RTX 3060 ส่วน X5 ใช้ seed 1 บน RTX 4070 ความต่างจาก seed และ GPU ยังไม่ได้วัด
  คำตอบที่ถูกต้องต้องรอ X6 (segformer_b2 ธรรมดา seed 1 บน GPU เดียวกัน)

## สิ่งที่ยังไม่รู้
- multi-task ช่วยหรือทำให้ segmentation แย่ลงเท่าไร (ต้องใช้ X6)
- ความแกว่งระหว่าง seed ของ SegFormer-B2 (จะได้ 2 ค่าเมื่อ X6 เสร็จ: seed 0 จาก X4 กับ seed 1 จาก X6 แต่ GPU ต่างกัน)
- ค่า `loss_weight` 0.1 ดีที่สุดหรือไม่ (ยังไม่ได้จูน)
- เหตุผลที่ `neckline` ทายยาก
- attribute head ช่วยแก้ error ระหว่าง `outer` กับ `upper` หรือไม่ (ยังไม่ได้วิเคราะห์)

## งานต่อ
1. **X6:** เทรน `segformer_b2_s1.yaml` แล้วเทียบ segmentation กับ X5 แบบ seed และ GPU เดียวกัน
2. เขียนรายงาน E1 เทียบสองงาน: segmentation (X5 กับ X6) และ attribute (X5 กับ majority baseline) แยกตาราง ตามที่ rubric ต้องการ
3. (ถ้าต้องการ) sweep `loss_weight` {0.05, 0.1, 0.3} ใช้เวลาประมาณ 2.3 ชั่วโมงต่อ run
