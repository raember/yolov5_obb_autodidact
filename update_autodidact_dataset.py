import datetime
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas
import requests
from PIL import Image, ImageDraw, ImageFont
from bs4 import BeautifulSoup
from pandas import DataFrame
from requests import HTTPError
from tqdm import tqdm, trange

CVAT_URL = 'localhost:9080'
ORG = 'AutoDidact'
FORMAT = 'CVAT for video 1.1'
AUTH_TOKEN = os.environ['AUTH_TOKEN']
CSRF_TOKEN = os.environ['CSRF_TOKEN']
SESSION_ID = os.environ['SESSION_ID']
OUT = Path('out')
LABELS = 'Bed', 'Staff', 'Devices', 'Patient'


def download_dataset(task_id: int, folder: Path) -> Tuple[Path, bool]:
    out = folder / f"{task_id}"
    done_flag = out / '.done'
    if done_flag.exists():
        return out, False
    zip_path = out.with_name(f"{task_id}.zip")
    if zip_path.exists():
        zip_path.unlink()
    url = f"http://{CVAT_URL}/api/tasks/{task_id}?org={ORG}"
    headers0 = {
        "Host": "localhost:9080",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:105.0) Gecko/20100101 Firefox/105.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": url,
        "Authorization": f"Token {AUTH_TOKEN}",
        "X-CSRFTOKEN": CSRF_TOKEN,
        "DNT": "1",
        "Connection": "keep-alive",
        "Cookie": f"csrftoken={CSRF_TOKEN}; sessionid={SESSION_ID}",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }
    resp0 = requests.get(url, headers=headers0)
    resp0.raise_for_status()
    meta = resp0.json()
    if meta['status'] != 'validation':
        shutil.rmtree(str(out), ignore_errors=True)
        raise Exception(" -> Task not done yet")
    if out.exists():
        done_flag.touch(exist_ok=True)
        return out, False
    url = f"http://{CVAT_URL}/api/tasks/{task_id}/dataset?org={ORG}&format={FORMAT}"
    headers = {
        "Host": "localhost:9080",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:105.0) Gecko/20100101 Firefox/105.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": url,
        "Authorization": f"Token {AUTH_TOKEN}",
        "X-CSRFTOKEN": CSRF_TOKEN,
        "DNT": "1",
        "Connection": "keep-alive",
        "Cookie": f"csrftoken={CSRF_TOKEN}; sessionid={SESSION_ID}",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }
    requests.get(url, headers=headers).raise_for_status()
    print("Requesting dataset export:")
    for _ in range(19):
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        if resp.status_code == 202:
            # Accepted
            print('.', end='')
        elif resp.status_code == 201:
            # Created
            print()
            break
        time.sleep(1)
    else:
        raise Exception("Failed to get the dataset within 20s")

    url2 = f"{url}&action=download"
    headers2 = {
        "Host": "localhost:9080",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:105.0) Gecko/20100101 Firefox/105.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": url,
        "DNT": "1",
        "Connection": "keep-alive",
        "Cookie": f"csrftoken={CSRF_TOKEN}; sessionid={SESSION_ID}",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
    }
    print("Downloading dataset:")
    resp2 = requests.get(url2, headers=headers2)
    try:
        resp2.raise_for_status()
    except Exception as e:
        if resp2.status_code == 401:
            shutil.rmtree(str(out), ignore_errors=True)
        raise e
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(zip_path, 'wb') as fp:
        fp.write(resp2.content)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out)
    zip_path.unlink()
    done_flag.touch(exist_ok=True)
    return out, True


def read_data_subset(subset_folder: Path, task_id: int) -> dict:
    assert subset_folder.is_dir()
    xml = BeautifulSoup(open(subset_folder / 'annotations.xml', 'r'), "xml")
    labels = {}
    for label in xml.select('annotations meta task labels label'):
        labels[label.select_one('name').text] = label.select_one('color').text
    assert labels.keys() == set(LABELS)
    source = xml.select_one('annotations meta source').text
    data = {
        'version': xml.select_one('annotations version').text,
        'labels': labels,
        'size': int(xml.select_one('annotations meta task size').text),
    }
    assert len(list((subset_folder / 'images').rglob('*.PNG'))) == int(data['size'])
    gt = []
    for track in xml.select('annotations track'):
        assert track['label'] in LABELS
        for box in track.select('box'):
            x0 = float(box['xtl'])
            y0 = float(box['ytl'])
            x1 = float(box['xbr'])
            y1 = float(box['ybr'])
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            w = x1 - x0
            h = y1 - y0
            rot = float(box.get('rotation', "0.0"))
            if h > w:
                w, h = h, w
                rot += 90
            rot %= 180
            date_str, bed, fps, dur = source.rsplit('.', maxsplit=1)[0].rsplit('-', maxsplit=3)
            gt.append({
                'task': task_id,
                'frame': int(box['frame']),
                'source': source,
                'date': date_str.rsplit('-', maxsplit=1)[0],
                'bed': int(bed),
                'fps': int(fps[1:]),
                'duration': datetime.timedelta(seconds=float(dur[1:]) / 100),
                'label': LABELS.index(track['label']),
                'cx': cx,
                'cy': cy,
                'w': w,
                'h': h,
                'rotation': rot,
            })
    if len(gt) > 0:
        data['gt'] = DataFrame(gt).set_index(['task', 'frame']).sort_index()
    else:
        data['gt'] = DataFrame(
            columns=['task', 'frame', 'source', 'date', 'bed', 'fps', 'duration', 'label', 'cx', 'cy', 'w', 'h',
                     'rotation']
        ).set_index(['task', 'frame'])
        shutil.rmtree(str(subset_folder), ignore_errors=True)
    return data


def concat_data_subsets(dataset: dict, extension: dict) -> dict:
    if len(dataset) == 0:
        dataset = {
            'version': extension['version'],
            'labels': extension['labels'],
            'size': 0,
            'gt': DataFrame(columns=extension['gt'].columns),
        }
    assert dataset['labels'] == extension['labels']
    dataset['size'] += extension['size']
    dataset['gt'] = pandas.concat([dataset['gt'], extension['gt']])
    return dataset


def rotate(arr: np.ndarray, angle: float) -> np.ndarray:
    ar = arr.copy()
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    center = np.array([np.mean(arr.T[0]), np.mean(arr.T[1])])
    ar = ar.reshape((4, 2)) - center
    ar = ar.dot(R) + center
    return ar


def draw_bbox(draw: ImageDraw.ImageDraw, cx: float, cy: float, w: float, h: float, rot: float, color: str,
              size: Tuple[int, int], label: str = None):
    xtl = cx - w / 2
    ytl = cy - h / 2
    xbr = cx + w / 2
    ybr = cy + h / 2
    bbox = rotate(np.array([(xtl, ytl), (xbr, ytl), (xbr, ybr), (xtl, ybr)]), rot)
    draw.line(list(map(tuple, np.concatenate([bbox, bbox[0:1]]).tolist())), fill=color, width=3)
    if label is not None:
        x0 = np.min(bbox.T[0])
        y0 = np.min(bbox.T[1])
        _, _, x1, y1 = ImageFont.load_default().getbbox(label)
        x1 += x0 + 4
        y1 += y0 + 4
        if x1 > size[0]:
            diff = x1 - size[0]
            x0 -= diff
        if y1 > size[1]:
            diff = y1 - size[1]
            y0 -= diff
        draw.rectangle((x0, y0, x1, y1), fill='#303030')
        draw.text((x0 + 2, y0 + 2), label, color)


def idx_to_img_path(task: int, frame: int) -> Path:
    return Path(f"frame_{frame:06}.PNG")


def visualize_sample(idx: Tuple[int, int], data: dict, folder: Path, save_to_disk: bool = False) -> Image.Image:
    task, frame = idx
    out = folder / str(task) / 'visualized'
    out.mkdir(parents=True, exist_ok=True)
    img_path = folder / str(task) / 'images' / idx_to_img_path(*idx)
    assert img_path.exists()
    vis_path = out / img_path.name
    img = Image.open(img_path)
    # img = img.convert('RGBA')
    draw = ImageDraw.Draw(img)
    ann = data['gt'].loc[[idx]]
    for (task, frame), (source, label, cx, cy, w, h, rot) in ann.iterrows():
        cls = LABELS[int(label)]
        col = data['labels'][cls]
        draw_bbox(draw, cx, cy, w, h, rot, col, img.size, cls)
    if save_to_disk:
        img.save(vis_path)
    return img


def visualize_dataset(data: dict, folder: Path, save_to_disk: bool = True):
    print('Visualizing dataset')
    index = data['gt'].index.unique()
    for idx in tqdm(index):
        visualize_sample(idx, data, folder, save_to_disk=save_to_disk)


def export_dataset(data: dict, folder: Path):
    index = data['gt'].index.unique()
    total = len(index)
    files = {
        'train': 0.70,
        'val': 0.15,
        'test': 0.15,
    }
    idx = 0
    for dataset_name, ratio in files.items():
        count = int(total * ratio)
        files[dataset_name] = (idx, idx + count)
        idx += count
    # noinspection PyUnboundLocalVariable
    files[dataset_name] = (files[dataset_name][0], total)
    with tqdm(total=total) as bar:
        for dataset_name, (idx0, idx1) in files.items():
            df = data['gt'].loc[index[idx0:idx1]]
            export_subset(df, data, folder, dataset_name, bar)


def export_subset(df: DataFrame, data: dict, folder: Path, name: str, bar: tqdm = None) -> bool:
    base_dir = Path('..', 'datasets', ORG.lower(), name.lower())
    img_dir = base_dir / 'images'
    label_dir = base_dir / 'labelTxt'
    vis_dir = base_dir / 'visualizations'
    img_dir.mkdir(exist_ok=True, parents=True)
    label_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    check_if_sample_exists = False
    added_samples = False
    index = df.index.unique()
    for idx in index:
        task, frame = idx
        sample_name = f"task_{task:03}_frame_{frame:06}"
        orig_img_path = folder / str(task) / 'images' / idx_to_img_path(*idx)
        assert orig_img_path.exists(), f"Source file does not exist: {orig_img_path}"
        img_fp = img_dir / f"{sample_name}.png"
        if check_if_sample_exists and img_fp.exists():
            # Don't add to the set - it already exists
            if bar is not None:
                bar.update()
            continue

        lbl_fp = label_dir / f"{sample_name}.txt"
        labels = {}
        with open(lbl_fp, 'w') as fp:
            for i, ((task, frame), (source, date_str, bed, fps, duration, label, cx, cy, w, h, rot)) in enumerate(
                    df.loc[[idx]].iterrows()):
                xtl = cx - w / 2
                ytl = cy - h / 2
                xbr = cx + w / 2
                ybr = cy + h / 2
                bbox = rotate(np.array([(xtl, ytl), (xbr, ytl), (xbr, ybr), (xtl, ybr)]), rot)
                elements = [*map(lambda flt: f"{flt:.2f}", bbox.reshape((8,)).tolist()), label, 1.0]
                fp.write(f"{', '.join(map(str, elements))}\n")
        if bar is not None:
            bar.update()
        added_samples = True

        visualize_sample(idx, data, vis_dir, True)

        # Only link now, so in case something breaks, we don't skip recreating this sample
        os.link(orig_img_path, img_fp)
    return added_samples


if __name__ == '__main__':
    dataset = {}
    for task_id in trange(200):
        is_new = False
        try:
            path, is_new = download_dataset(task_id, OUT)
        except HTTPError as err:
            if err.response.status_code == 401:
                tqdm.write("Cookies expired! Please update the tokens!")
                exit(1)
            continue
        except Exception as err:
            tqdm.write(f"Task {task_id}: {str(err)}")
            continue
        if is_new or True:
            dataset = concat_data_subsets(dataset, read_data_subset(path, task_id))
    if len(dataset) == 0:
        print("Nothing to do")
        exit(0)
    export_dataset(dataset, OUT)
