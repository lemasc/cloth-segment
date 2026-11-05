# Handoff — T6 follow-up: U-Net R34 retrain on eval-v2 (amonwan)

## In plain words
เทรน U-Net ใหม่ด้วย config หลัง M3/M4 แล้ว รายละเอียดทั้งหมดอยู่ใน report:
[`docs/reports/2026-11-05-T6-unet-r34-retrain.md`](../reports/2026-11-05-T6-unet-r34-retrain.md)
สิ่งที่ต้องทำต่อ: ทีมตัดสิน 40k / 80k ก่อน T7/T8 เทรน

## Summary
- PR #14: report ของ run ใหม่ หมายเหตุ stale ใน handoff ของ T6 และ `scripts/compare_runs.py`
- run ใหม่: `runs/20261105-1307-1acb88f-unet-r34` (`configs/unet_r34.yaml`, config hash `d4f03d515ed0`)

## Decisions and why
- ไม่แก้ `PLAN.md`: T6 เป็น `done` อยู่แล้ว และแผนยังไม่มีชนิด task สำหรับการเทรนซ้ำ (reviewer จะเปิด issue แยก)

## Results / findings
ดู report หัวข้อ "ผล" และ "การตีความ" ตัวเลขทั้งหมดสร้างใหม่ได้ด้วย:

```bash
uv run python scripts/compare_runs.py 20261105-0323-7b1798b-unet-r34 20261105-1307-1acb88f-unet-r34 <out-dir>
```

## Gotchas
- `<out-dir>` ควรอยู่นอก repo ไม่ commit รูป
- run เดิมมี test แค่ eval-v1 สคริปต์เลือก eval version ใหม่สุดที่มีครบทุกชุดของแต่ละ run เอง

## Open items
- ตัดสิน 40k / 80k (ดูหัวข้อ "ข้อโต้แย้ง" ใน report)
- สาเหตุที่ `accessory` ลดลง: reviewer กำลังดูแยกตามคลาสดิบ

## Next
- T7, T8 เทรนหลังตัดสิน 40k / 80k
- T9: mIoU 11 คลาส และ `accessory` แยกในตารางเทียบ
