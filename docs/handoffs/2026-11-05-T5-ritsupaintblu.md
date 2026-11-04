# Handoff — T5 Training and evaluation pipeline (ritsupaintblu)

## In plain words
มีคำสั่งเทรนและคำสั่งวัดผลที่ทุกโมเดลใช้ร่วมกันแล้ว ค่าที่ต้องเหมือนกันทุกโมเดล (D5) เช่น
batch 8, 40k iteration, AdamW, CE และ bf16 อยู่ใน `configs/base.yaml` ยังไม่มีการเทรนจริง
เพราะยังไม่มีโมเดลตัวจริง ขั้นต่อไปคือ T6–T8 ลงทะเบียนโมเดลแล้วรัน baseline

## Summary
PR: #9 (https://github.com/lemasc/cloth-segment/pull/9) · branch `T5/train-eval-pipeline`

- **วิธีใช้:**
  ```bash
  uv run python -m cloth_segment.train configs/<model>.yaml [--wait]
  uv run python -m cloth_segment.evaluate <run> [lip cihp dfmm] [--wait]
  ```
- `src/cloth_segment/train.py`: ทำงานตามลำดับนี้
  1. อ่าน config (base + experiment) ถ้ามี run ที่เทรน config เดียวกันจบแล้ว (hash เดียวกัน) จะข้าม (D7)
  2. ขอ GPU lock แล้วสร้างโฟลเดอร์ run
  3. จำกัดหน่วยความจำ GPU ไว้เท่าที่ว่างตอนเริ่ม
  4. เทรน `train.iters` step แล้ววัดผลบน val ภายในทุก `eval.every` step และที่ step สุดท้าย
  5. เก็บ `ckpt/last.pt` (โมเดล + optimizer + scheduler + step) กับ `ckpt/best.pt` (ตัวที่ mIoU บน val ดีที่สุด)
     และเขียน log ลง `log.jsonl`
  6. จบแล้วตั้ง `meta.json` เป็น `finished` ถ้า error หรือกด Ctrl-C จะเป็น `failed` และมี `error.txt`
- `src/cloth_segment/evaluate.py`
  - `predict_confusion`: วัดผลที่ความละเอียดเท่า mask ต้นฉบับ (eval-v1) worker ของ loader เป็นคนอ่าน mask ต้นฉบับ
  - `main`: ใช้ `best.pt` วัดบน test split ของ LIP, CIHP และ DFMM แล้วเขียน
    `eval/eval-v1/<test_set>/metrics.json` (mIoU, Dice, IoU ราย class, confusion matrix,
    step ของ checkpoint, commit) ถ้าไฟล์มีอยู่แล้วจะไม่เขียนทับ
- `src/cloth_segment/runs.py`: จัดการ config และ run (merge, hash, โฟลเดอร์ run, cache, GPU lock)
  และเพิ่ม shared field ใหม่ (ดูหัวข้อถัดไป)
- `src/cloth_segment/models/__init__.py`: T5 เป็นเจ้าของ interface ของโมเดล:
  `@register("<name>")`, `build_model(model_cfg)` และ `REGISTRY` โมเดลรับภาพ (B, 3, S, S)
  และคืน logits (B, NUM_CLASSES, h, w) โดย h, w เล็กกว่า S ได้ (SegFormer ให้ 1/4)
  แล้ว `logits_at` จะขยายให้เท่ากัน
- `scripts/probe_memory.py` + `src/cloth_segment/probe.py`: วัดหน่วยความจำและความเร็ว
  ผลอยู่ใน `docs/reports/2026-11-05-T5-gpu-probe.md`
- Test ของ T5 อยู่ใน `tests/test_train.py`, `test_evaluate.py`, `test_runs.py`, `test_metrics.py`
  และ `test_probe.py` ทั้งหมดรันบน CPU ไม่ต้องใช้ข้อมูลจริงหรือ GPU

## Decisions and why
- **Batch 8, 40k iteration:** batch 8 เป็นค่าที่ใหญ่ที่สุดที่ SegFormer-B2 รับได้บน GPU 12 GB (probe)
  40k iteration ประมาณ 5 epoch จะขยายเป็น 80k ถ้า curve ของ val ยังขึ้นอยู่ตอนจบ
- **AdamW ตัวเดียวกันทุกโมเดล** (`weight_decay 0.01`, warmup 1,500 step แล้วลด LR แบบ poly
  power 1 จนเป็น 0) เพื่อให้เทียบกันได้ (D5) ส่วน `optim.lr` **ไม่ใช่** shared field แต่ละโมเดลตั้งเองได้
- **`lr: 1.0e-4` ยังไม่ได้ปรับจูน** เลือกเพราะ probe ใช้ค่านี้ T6–T8 อาจต้องตั้งค่าเองต่อโมเดล
- **Loss = CE ธรรมดา** ไม่ถ่วงน้ำหนักและไม่มี Dice เพราะ E3 จะเทียบ CE กับ CE + Dice
- **AMP bf16:** probe วัด batch 8 ตอนเปิด AMP ถ้าปิด SegFormer อาจรับ batch 8 ไม่ไหว
- **เลือก checkpoint จาก val ภายในเท่านั้น** เพราะ official LIP/CIHP val เป็น test set (D4)
- **Shared field ใหม่** (เปลี่ยนต้องประกาศใน `run.overrides_shared`): `train.amp`, `optim.name`,
  `optim.weight_decay`, `optim.betas`, `optim.warmup_iters`, `optim.poly_power`, `loss`, `eval`
- **`run.num_workers` และ `run.log_every` ไม่นับเข้า hash** เพราะไม่เปลี่ยนผล
- **T5 กำหนด interface registry เอง:** T5 merge ก่อน แล้ว T6–T8 ปรับตาม ไม่ก๊อปจาก PR #7
  ที่ยังเป็น draft
- **วัดผลบน test ทั้ง 3 ชุดเสมอ** แม้ run จะเทรนแค่ dataset เดียว ตาราง cross-dataset ของ E2
  จึงไม่ต้องรันอะไรเพิ่ม

## Results / findings
- ยังไม่มีผลการเทรน ไม่มีโฟลเดอร์ run บน shared mount
- ruff และ pytest ผ่านทั้งหมด 120 test
- ลองรัน `fit()` บน CPU ด้วยข้อมูลสังเคราะห์และโมเดลจิ๋ว 12 step ในโฟลเดอร์ชั่วคราว ไม่ได้เขียนลง `runs/`
  ผลคือได้ log, val, `best.pt`/`last.pt` และ `meta.json` ครบ (ตัวเลข mIoU ของรอบนี้ไม่มีความหมาย)
- ลองโหลด test set จาก manifest จริงได้ LIP 9,632 ภาพ, CIHP 5,000 ภาพ, DFMM 1,293 ภาพ
- ผลของ probe อยู่ใน `docs/reports/2026-11-05-T5-gpu-probe.md`

## Gotchas
- **WSL ไม่แจ้ง OOM:** `train.py` จึงเรียก `set_per_process_memory_fraction` ตามหน่วยความจำที่ว่างตอนเริ่ม
  ถ้ามี process อื่นใช้ GPU อยู่ เพดานจะต่ำลง ค่าเพดานบันทึกไว้ใน `meta.json` (`gpu_cap_mib`)
- **SegFormer ที่ batch 8 เหลือที่ว่างประมาณ 2.4 GB** การ eval ระหว่างเทรนใช้หน่วยความจำเพิ่ม ยังไม่ได้วัดบน GPU
- **`drop_last=True`:** DeepLabV3+ เทรนด้วย batch 1 ไม่ได้
- **ถ้า loss เป็น NaN หรือ inf** การเทรนจะหยุดทันทีและ run เป็น `failed`
- **run ที่เริ่มตอนโค้ดยังมีการแก้ที่ไม่ได้ commit** ได้ชื่อลงท้าย `-dirty` และไม่นับเป็น cache
  รันซ้ำจะเทรนใหม่เสมอ
- **`ckpt/last.pt` มี optimizer state** แต่ยังไม่มีคำสั่ง resume (ดู Open items)

## Open items
- **Resume:** ถ้า run ล้มกลางทาง ตอนนี้ต้องเริ่มใหม่ มีของใน `last.pt` ครบพอจะทำ resume ได้ เป็น follow-up
- **ทีมต้องตกลงกันเรื่อง LR ต่อโมเดล** ก่อนรัน baseline
- **Freeze contract:** หลัง PR นี้ merge และมี baseline แรกที่ทีมตกลงกัน maintainer freeze ได้ทั้ง
  "Evaluation protocol" และ "Shared comparison fields"
- **PR #7 (T6):** หลัง PR นี้ merge ให้ #7 merge `main` เข้ามา ใช้ `models/__init__.py` ของ T5
  แล้วปรับ builder ของ `unet` ให้รับ `model` config dict (`build_model(model_cfg)`)

## Next
- **T6–T8:** ลงทะเบียนโมเดลด้วย `@register(...)` แล้วรัน `python -m cloth_segment.train configs/<model>.yaml`
  ตามด้วย `python -m cloth_segment.evaluate <run>`
- **T9:** ใช้ `metrics.json` (มี confusion matrix) ทำตารางและกราฟ
- **E1:** ต้องเพิ่ม loss ของ attribute ใน `loss_fn` และ head ใน model
