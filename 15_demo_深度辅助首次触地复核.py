"""VS Code 中文入口；自动使用项目的 .venv-demo15 独立环境。"""
from pathlib import Path
import os
import sys

项目根目录=Path(__file__).resolve().parent
输入视频='training_video/video29.mp4'
批量目录=None  # 设为 'training_video' 分析全部视频
深度模式='fusion'  # depth / fusion / visual；三种模式会共用证据进行对比
关闭深度=False
推理设备='mps'  # CPU 使用 'cpu'r
显示预览=True
显示深度视图=True  # True 开启独立深度窗口；False 隐藏，仍进行深度计算
预览缩放=.75
调试并冻结参数=False  # 需要批量目录，且仅用 video18–video23 调参
冻结参数文件='configs/demo15_contact.json'  # 调试并冻结参数开启时自动忽略
复用证据目录=None
脱靶短路=True  # 明确脱靶后跳过鞋底精修与靶环计算

if __name__=='__main__':
    独立环境=项目根目录/'.venv-demo15'
    if Path(sys.prefix).resolve()!=独立环境.resolve():
        独立解释器=独立环境/'bin'/'python'
        if not 独立解释器.is_file():
            raise SystemExit('Demo15 失败：缺少 .venv-demo15 独立环境，请按 docs/demo15.md 安装。')
        print('Demo15：自动切换到 .venv-demo15 独立环境。',flush=True)
        os.execv(str(独立解释器),[str(独立解释器),'-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    sys.path.insert(0,str(项目根目录/'scripts'))
    from run_demo15 import main
    参数=['--mode',深度模式,'--device',推理设备,'--display-scale',str(预览缩放)]
    参数+=['--video-dir',批量目录] if 批量目录 else ['--video',输入视频]
    if not 脱靶短路:参数+=['--no-miss-shortcut']
    if 关闭深度:参数+=['--no-depth']
    if not 显示预览:参数+=['--no-preview']
    if not 显示深度视图:参数+=['--no-depth-view']
    if 调试并冻结参数:参数+=['--tune']
    if 冻结参数文件 and not 调试并冻结参数:参数+=['--config',冻结参数文件]
    if 复用证据目录:参数+=['--reuse-extraction',复用证据目录]
    raise SystemExit(main(参数))
