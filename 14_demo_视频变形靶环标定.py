"""演示14：全自动追踪实际变形靶环与靶心；仅输出二维像素位置。

空格暂停；n 单步；r 重新开始；o 切换叠加；q / Esc 退出。
实线为图像支持的位置，橙色虚线为短时预测，未知部分不绘制。
"""
from pathlib import Path
import runpy
import sys

项目根目录 = Path(__file__).resolve().parent
# ===== 可调试配置 =====
输入视频路径 = 项目根目录 / 'training_video/video10.mp4'
预览缩放比例 = 0.75
靶环数量先验 = 10
输出标注视频 = None  # 例如：项目根目录 / 'output/demo14/video6.mp4'
输出坐标文件 = None  # 例如：项目根目录 / 'output/demo14/video6.jsonl'
# =====================

if __name__ == '__main__':
    sys.path.insert(0,str(项目根目录/'scripts'))
    sys.argv = ['run_demo14.py','--video',str(输入视频路径),'--display-scale',str(预览缩放比例),
                '--ring-count',str(靶环数量先验)]
    if 输出标注视频: sys.argv += ['--output-video',str(输出标注视频)]
    if 输出坐标文件: sys.argv += ['--output-jsonl',str(输出坐标文件)]
    runpy.run_path(str(项目根目录/'scripts/run_demo14.py'),run_name='__main__')
