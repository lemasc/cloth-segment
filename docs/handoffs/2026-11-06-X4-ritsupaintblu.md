# Handoff — X4 SegFormer-B2, 40k (ritsupaintblu)

## In plain words
เทรน SegFormer-B2 ครบ 40k บน RTX 3060 (`lemasc-ubuntu`) และประเมิน test แล้ว
test mIoU: LIP 0.6155, CIHP 0.6295, DFMM 0.8425 สูงกว่า U-Net (X2) ทุกชุดและทุกคลาส (1 run ต่อโมเดล)
`best.pt` อยู่ที่ step 40000 และ val ยังขึ้นอยู่ ต้องให้ maintainer ดูประกอบเงื่อนไขของ M7 หลัง X3

## Summary
- Run folder: `runs/20261105-2220-ac9a26b-segformer-b2` (commit `ac9a26b`, config hash `8e9039649376`, data `splits-v2-d023252b81d9`)
- Report: [`docs/reports/2026-11-06-X4-segformer-b2.md`](../reports/2026-11-06-X4-segformer-b2.md)
- ผล probe หน่วยความจำก่อนเทรน: [`docs/reports/2026-11-06-X4-segformer-b2-probe.json`](../reports/2026-11-06-X4-segformer-b2-probe.json)
- ไม่ได้แก้โค้ดหรือ config มีแค่ report, handoff และแถว X4 ใน `PLAN.md`

## Decisions and why
- **รันบน `lemasc-ubuntu` (RTX 3060) ไม่ใช่ `METAVERSE-05` (RTX 4070):** ผู้ใช้เลือก และยืนยันว่าเครื่องนี้ไม่ต้องใช้ GPU lock
  CONTRIBUTING section 7 ยอมให้ใช้ GPU ของตัวเองได้ ถ้า config hash และ data version ตรงกัน maintainer เป็นคนตัดสินว่าจะใช้ run ไหนในรายงาน
- **วัด probe ซ้ำก่อนเทรน:** probe ของ T5 วัด builder เวอร์ชัน HF บน 4070 และ batch 8 เป็น shared field ลดเฉพาะโมเดลเดียวไม่ได้
  จึงต้องรู้ก่อนว่าพอ ผลคือ batch 8 ผ่าน (peak 9,019 MiB จากเพดาน 11,773 MiB)
- **เทียบกับ X2 โดยใช้ตัวเลขจาก report ของ X2:** run folder ของ X2 ไม่อยู่ในเครื่องนี้ จึงใช้ `scripts/compare_runs.py` ไม่ได้
  ตรวจแล้วว่า 11-class mean ที่คำนวณจากตารางรายคลาสของ X2 (0.6789) ตรงกับที่ report ของ X2 เขียนไว้

## Results / findings

| test mIoU | LIP | CIHP | DFMM | เฉลี่ย 12 คลาส | เฉลี่ย 11 คลาส |
|---|---|---|---|---|---|
| X2 U-Net R34 | 0.5501 | 0.5672 | 0.7570 | 0.6248 | 0.6789 |
| X4 SegFormer-B2 | 0.6155 | 0.6295 | 0.8425 | 0.6959 | 0.7239 |

- `accessory` ของ X4 เรียนได้ตั้งแต่ 4k (test LIP 0.3476, CIHP 0.2041, DFMM 0.6118) ต่างจาก X2 ที่แทบเป็น 0
- val เฉลี่ย 3 ชุดขึ้นทุก check จนถึง 40k (0.5803 → 0.6978)
- เวลา: เทรน 14,937 วินาที, val ระหว่างเทรน 740 วินาที, test 290 วินาที (บน 3060)

รายละเอียดและข้อจำกัดของการเทียบอยู่ใน report

## Gotchas
- **พิมพ์ path ของไฟล์เปล่าๆ ใน shell แล้วได้ `permission denied`:** shell พยายามรันไฟล์เป็นโปรแกรม
  ให้ใช้ `cat runs/<run>/meta.json` หรือ `python -m json.tool runs/<run>/meta.json`
- **`pgrep -f` จับคำสั่งของตัวเองได้:** ถ้า pattern อยู่ในบรรทัดคำสั่งที่เช็กเอง จะได้ "alive" ทั้งที่ process จบแล้ว ให้เช็กด้วย PID (CONTRIBUTING section 7 เตือนไว้แล้ว)
- เครื่องนี้ต้องรันด้วย `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` เพราะ HF token ที่เก็บไว้หมดอายุ (handoff ของ T8)
- nvidia-smi แสดง 10.8 GB ระหว่างเทรน มากกว่า peak ที่ probe วัด (9.0 GB) เพราะนับหน่วยความจำที่ PyTorch จองไว้ล่วงหน้าด้วย ไม่ OOM

## Open items
- **run folder อยู่แค่ใน `lemasc-ubuntu`** เครื่องนี้ไม่ได้ต่อ shared mount (D7) ต้องตกลงกับ maintainer เรื่องการย้าย
- **M7:** `best.pt` = step 40000 และ val ยังขึ้น เงื่อนไขของ M7 เขียนสำหรับ X3 ให้ maintainer พิจารณาเมื่อ X3 เสร็จ
- probe ที่ใช้ builder ตัวจริงยังไม่ได้อยู่ใน `scripts/probe_memory.py` (follow-up)

## Next
- X3 (DeepLabV3+) แล้วจึง T9 (ตารางเทียบ 3 โมเดล) และ T10 (ข้อจำกัด)
