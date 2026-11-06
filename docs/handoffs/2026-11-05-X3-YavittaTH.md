# Handoff — X3 DeepLabV3+ R50, 40k (YavittaTH)

## In plain words
เทรน DeepLabV3+ R50 ครบ 40k และประเมินบน test แล้ว test mIoU: LIP 0.5764, CIHP 0.5875, DFMM 0.7901 สูงกว่า U-Net (X2) ทุกชุด ส่วนใหญ่มาจาก `accessory`
`best.pt` อยู่ที่ step 40000 ซึ่งตรงกับเงื่อนไขของ M7 **lemasc ต้องตัดสินว่าจะเปิด budget ใหม่หรือไม่**

## Summary
- Run: `runs/20261105-2050-f4e2ffe-deeplabv3p-r50` (config `configs/deeplabv3p_r50.yaml`, hash `8d71807da489`, data `splits-v2-d023252b81d9`, commit `f4e2ffe` ไม่ dirty)
- Test eval: `eval/eval-v2/{lip,cihp,dfmm}/metrics.json` ใน run folder
- Report: `docs/reports/2026-11-05-X3-deeplabv3p-r50.md`
- `PLAN.md`: กรอก run folder และ report ในแถว X3 และตั้งเป็น `done`
- ไม่ได้แก้โค้ดหรือ config

## Decisions and why
- **ไม่ได้เปลี่ยน budget เอง** แม้ `best.pt` จะอยู่ที่ step 40000 เพราะ M7 บอกให้แจ้ง lemasc ไม่ให้ตัดสินเอง
- **เทียบกับ X2 เท่านั้น** (ไม่ใช่ X1) เพราะ X2 ใช้ eval-v2 และ seed init เหมือน X3

## Results / findings

| test mIoU | LIP | CIHP | DFMM | เฉลี่ย 12 คลาส | เฉลี่ย 11 คลาส |
|---|---|---|---|---|---|
| X2 U-Net R34 | 0.5501 | 0.5672 | 0.7570 | 0.6248 | 0.6789 |
| X3 DeepLabV3+ R50 | 0.5764 | 0.5875 | 0.7901 | 0.6513 | 0.6859 |

- `accessory` test IoU: X3 0.2394 / 0.1619 / 0.4120 กับ X2 0.0850 / 0.0022 / 0.0007 (LIP / CIHP / DFMM)
- ถ้าไม่นับ `accessory` ต่างกัน +0.0070 ยังสรุปไม่ได้ว่า X3 ดีกว่าจริง (ดู "การตีความ" ใน report)
- ตารางเต็มอยู่ใน report

## Gotchas
- ชื่อ run folder ใช้ `-` แทน `_` (`deeplabv3p-r50`) แต่ชื่อใน GPU lock ใช้ชื่อไฟล์ config (`deeplabv3p_r50`)
- ความเร็ว: ประมาณ 0.105 วินาทีต่อ iteration รวมทั้งหมด 80 นาทีบน RTX 4070 test eval ใช้ 1.5 นาที
- รูปกราฟสร้างใหม่ได้ด้วย `scripts/compare_runs.py` (คำสั่งอยู่ใน report) ไม่ได้ commit

## Open items
- **lemasc:** เงื่อนไขของ M7 เกิดแล้ว (`best_step` 40000) จะคง 40k หรือเปลี่ยน
- ความแกว่งระหว่างรันของ DeepLabV3+ ยังไม่รู้ (มีรันเดียว)

## Next
X4 (SegFormer-B2) หลัง T8 ถ้า lemasc คง 40k และ T9 (ตารางเทียบ 3 โมเดล)
