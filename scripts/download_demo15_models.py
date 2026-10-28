"""Download Demo15's three external weights and verify recorded SHA256s."""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def verified(path, metadata):
    if not path.is_file() or path.stat().st_size != metadata['bytes']:
        return False
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest() == metadata['sha256']


def download(destination, metadata, opener):
    if destination.exists():
        if not verified(destination, metadata):
            raise ValueError(f'文件已存在但校验不匹配，不覆盖：{destination}')
        print(f'已校验：{destination.relative_to(ROOT)}', flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.download')
    primary = metadata['source'].replace('/blob/', '/resolve/')
    urls = list(dict.fromkeys([primary, metadata.get('download_url', primary)]))
    for attempt in range(6):
        if verified(partial, metadata):
            partial.replace(destination)
            print(f'下载及校验完成：{destination.relative_to(ROOT)}', flush=True)
            return
        offset = partial.stat().st_size if partial.exists() else 0
        if offset >= metadata['bytes']:
            raise ValueError(f'断点文件校验失败，请检查并移走后重试：{partial}')
        headers = {'User-Agent': 'Hawkeye-scorer-model-download'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        request = urllib.request.Request(urls[attempt % len(urls)], headers=headers)
        print(f'下载 {destination.name}：{offset / 1024**2:.1f}/{metadata["bytes"] / 1024**2:.1f} MiB，尝试 {attempt + 1}/6', flush=True)
        try:
            with opener.open(request, timeout=60) as response:
                resumed = offset > 0 and response.status == 206
                if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('服务器返回了不匹配的断点位置')
                total = offset if resumed else 0
                reported = total
                with partial.open('ab' if resumed else 'wb') as stream:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        total += len(block)
                        if total > metadata['bytes']:
                            raise ValueError('下载数据超过记录的文件大小')
                        stream.write(block)
                        if total - reported >= 20 * 1024**2:
                            print(f'  {total / 1024**2:.1f} MiB', flush=True)
                            reported = total
        except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
            print(f'连接中断（{type(error).__name__}），保留断点重试。', flush=True)
    if verified(partial, metadata):
        partial.replace(destination)
        print(f'下载及校验完成：{destination.relative_to(ROOT)}', flush=True)
        return
    raise RuntimeError(f'下载或校验未完成，断点保留在：{partial}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proxy', help='可选代理，例如 http://127.0.0.1:7897；不写入项目配置')
    args = parser.parse_args()
    opener = (urllib.request.build_opener(urllib.request.ProxyHandler({'http': args.proxy, 'https': args.proxy}))
              if args.proxy else urllib.request.build_opener())
    for manifest, relative_path in (
        ('YOLO26_SEG_SOURCE.json', 'yolo26n-seg.pt'),
        ('YOLO26_DEPTH_SOURCE.json', 'models/yolo26n-depth.pt'),
        ('SCHP_LIP_SOURCE.json', 'models/exp-schp-201908261155-lip.pth'),
    ):
        metadata = json.loads((ROOT / 'models' / manifest).read_text(encoding='utf-8'))
        download(ROOT / relative_path, metadata, opener)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f'权重准备失败：{error}')
