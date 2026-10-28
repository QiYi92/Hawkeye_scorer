"""Run both supported trackers sequentially and save comparable metrics."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description="Demo13 BoT-SORT / ByteTrack 顺序性能对比")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--model", default="yolo26n-seg.pt")
    p.add_argument("--mode", choices=["person", "athlete"], default="person")
    p.add_argument("--device", default="")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output or root / "output" / f"demo13_benchmark_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    report = {}
    for tracker in ("botsort", "bytetrack"):
        folder = output / tracker
        subprocess.run([sys.executable, str(root / "scripts/run_demo13.py"), "--video", str(args.video),
                        "--model", args.model, "--mode", args.mode, "--device", args.device,
                        "--imgsz", str(args.imgsz), "--max-frames", str(args.max_frames),
                        "--tracker", tracker, "--output", str(folder)], check=True)
        report[tracker] = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    (output / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for tracker, metrics in report.items():
        print(f"{tracker}：{metrics['processing_fps']:.2f} 帧/秒，主目标覆盖率 "
              f"{metrics['primary_coverage']:.1%}，主目标编号变化 {metrics['primary_id_changes']} 次")
    print(output / "benchmark.json")


if __name__ == "__main__":
    main()
