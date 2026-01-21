
import cv2
import base64
import numpy as np
import math
import re
import requests
from collections import namedtuple
from typing import Optional

RotationResult = namedtuple(
    "RotationResult", "similar, inner_rotate_angle, total_rotate_angle"
)

IMAGE_TYPE_BASE64 = 0
IMAGE_TYPE_URL = 1
IMAGE_TYPE_FILE = 2


def double_rotate_identify(
    small_circle: str,
    big_circle: str,
    image_type: int = IMAGE_TYPE_BASE64,
    check_pixel: int = 10,
    speed_ratio: float = 1.0,
    grayscale: bool = False,
    standard_deviation: int = 0,
    cut_pixel_value: int = 0,
    proxies: Optional[dict] = None,
) -> RotationResult:
    """
    双图、双向、同时旋转类型滑块验证码识别
    :param small_circle: 小圈图片
    :param big_circle: 大圈图片
    :param image_type: 图片类型: 0: 图片base64; 1: 图片url; 2: 图片文件地址
    :param check_pixel: 进行图片验证的像素宽度
    :param speed_ratio: 内外圈转动速率比: 内圈转速 / 外圈转速
    :param grayscale: 是否需要灰度化处理: True: 是; False: 否
    :param standard_deviation: 计算行/列像素点值的标准差
    :param cut_pixel_value: 需要裁切的像素点值; 保留值区间range(cut_pixel_value, 255 - cut_pixel_value)
    :param proxies: 代理
    :return: namedtuple -> RotationResult(similar, inner_angle, inner_angle)
    """
    inner_image, outer_image = _load_images(
        small_circle, big_circle, image_type, grayscale, proxies
    )

    cut_inner_image = _cut_image(inner_image, standard_deviation, cut_pixel_value)
    cut_inner_radius = cut_inner_image.shape[0] // 2
    cut_outer_image = _cut_image(
        outer_image, standard_deviation, cut_pixel_value, cut_inner_radius, check_pixel
    )

    mask_inner_image = _mask_image(cut_inner_image, check_pixel)
    mask_outer_image = _mask_image(cut_outer_image, check_pixel)

    match_result = _find_best_rotation_angle(
        mask_inner_image, mask_outer_image, speed_ratio
    )

    return match_result


def _load_images(
    small_circle_src: str,
    big_circle_src: str,
    image_type: int,
    grayscale: bool,
    proxies: Optional[dict],
):
    inner_image = _decode_image(small_circle_src, image_type, grayscale, proxies)
    outer_image = _decode_image(big_circle_src, image_type, grayscale, proxies)
    return inner_image, outer_image


def _mask_image(origin_array, check_pixel):
    radius = origin_array.shape[0] // 2
    
    # 创建掩码
    center_point = (radius, radius)
    mask = np.zeros((radius * 2, radius * 2), dtype=np.uint8)
    mask = cv2.circle(mask, center_point, radius, (255, 255, 255), -1)
    mask = cv2.circle(mask, center_point, radius - check_pixel, (0, 0, 0), -1)
    
    src_array = np.zeros(origin_array.shape, dtype=np.uint8)
    mask_result = cv2.add(origin_array, src_array, mask=mask)
    return mask_result


def _cut_image(origin_array, std, cut_value, radius=None, check_pixel=None):
    cut_pixel_list = []  # 上, 左, 下, 右
    height, width = origin_array.shape[:2]
    if not radius:
        for rotate_count in range(4):
            cut_pixel = 0
            rotate_array = np.rot90(origin_array, rotate_count)
            for line in rotate_array:
                if len(line.shape) == 1:
                    pixel_set = set(list(line)) - {0, 255}
                else:
                    pixel_set = set(map(tuple, line)) - {(0, 0, 0), (255, 255, 255)}

                if not pixel_set:
                    cut_pixel += 1
                    continue

                if len(line.shape) == 1:
                    pixels = tuple(
                        pixel
                        for pixel in tuple(pixel_set)
                        if cut_value <= pixel <= 255 - cut_value
                    )
                    pixel_std = np.std(pixels)
                    if pixel_std > std:
                        break
                else:
                    count = 0
                    pixels = [[], [], []]
                    for b, g, r in pixel_set:
                        if cut_value <= min(b, g, r) <= max(b, g, r) <= 255 - cut_value:
                            pixels[0].append(b)
                            pixels[1].append(g)
                            pixels[2].append(r)

                    bgr_std = tuple(
                        np.std(pixels[i]) if pixels[i] else 0 for i in range(3)
                    )
                    for pixel_std in bgr_std:
                        if pixel_std >= std:
                            count += 1
                    if count == 3:
                        break
                cut_pixel += 1
            cut_pixel_list.append(cut_pixel)
        cut_pixel_list[2] = height - cut_pixel_list[2]
        cut_pixel_list[3] = width - cut_pixel_list[3]

    elif check_pixel:
        y, x = height // 2, width // 2
        resize_check_pixel = math.ceil(radius / (radius - check_pixel) * check_pixel)
        for i in -1, 1:
            for p in y, x:
                pos = p + i * radius
                for _ in range(p - radius):
                    p_x, p_y = (pos, y) if len(cut_pixel_list) % 2 else (x, pos)
                    pixel_point = origin_array[p_y][p_x]
                    pixel_set = (
                        {pixel_point} - {0, 255}
                        if isinstance(pixel_point, np.uint8)
                        else set(tuple(pixel_point)) - {(0, 0, 0), (255, 255, 255)}
                    )
                    if not pixel_set:
                        pos += i
                        continue
                    status = True
                    for pixel in pixel_set:
                        if pixel <= cut_value or pixel >= 255 - cut_value:
                            status = False
                            break
                    if status:
                        break
                    pos += i
                cut_pixel_list.append(pos + i * resize_check_pixel)

    up, left, down, right = cut_pixel_list
    cut_array = origin_array[up:down, left:right]
    diameter = (radius or min(cut_array.shape[:2]) // 2) * 2
    cut_result = cv2.resize(cut_array, dsize=(diameter, diameter))
    return cut_result


def _find_best_rotation_angle(
    inner_image: np.ndarray, outer_image: np.ndarray, speed_ratio: float
) -> RotationResult:
    def _rotate_and_match(angle):
        h, w = inner_image.shape[:2]
        center = (w / 2, h / 2)
        mat_inner = cv2.getRotationMatrix2D(center, -angle, 1)
        rot_inner = cv2.warpAffine(inner_image, mat_inner, (w, h))
        outer_angle = angle / speed_ratio
        mat_outer = cv2.getRotationMatrix2D(center, outer_angle, 1)
        rot_outer = cv2.warpAffine(outer_image, mat_outer, (w, h))
        res = cv2.matchTemplate(rot_outer, rot_inner, cv2.TM_CCOEFF_NORMED)
        _, sim, _, _ = cv2.minMaxLoc(res)
        return sim

    # 粗略搜索
    best_angle_coarse = 0
    max_sim_coarse = -1.0
    for angle in range(0, 360, 5):  # 步长为5度
        sim = _rotate_and_match(angle)
        if sim > max_sim_coarse:
            max_sim_coarse = sim
            best_angle_coarse = angle

    # 精细搜索
    best_angle_fine = best_angle_coarse
    max_sim_fine = max_sim_coarse
    start_angle = best_angle_coarse - 5
    end_angle = best_angle_coarse + 5
    for angle in range(start_angle, end_angle + 1):
        sim = _rotate_and_match(angle)
        if sim > max_sim_fine:
            max_sim_fine = sim
            best_angle_fine = angle

    return RotationResult(max_sim_fine, best_angle_fine, best_angle_fine)


def _decode_image(
    image_source: str, image_type: int, grayscale: bool, proxies=None
) -> np.ndarray:
    assert image_type in [IMAGE_TYPE_BASE64, IMAGE_TYPE_URL, IMAGE_TYPE_FILE]
    if image_type == IMAGE_TYPE_BASE64:
        search_base64 = re.search("base64,(.*?)$", image_source)
        base64_image = search_base64.group(1) if search_base64 else image_source
        image_array = np.asarray(
            bytearray(base64.b64decode(base64_image)), dtype="uint8"
        )
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    elif image_type == IMAGE_TYPE_URL:
        image_content = _fetch_image_from_url(image_source, proxies)
        if not image_content:
            raise Exception("请求图片链接失败！")
        image_array = np.array(bytearray(image_content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    else:  # IMAGE_TYPE_FILE
        image = cv2.imread(image_source)

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return image


def _fetch_image_from_url(
    image_url: str, proxies: Optional[dict] = None
) -> Optional[bytes]:
    headers = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(image_url, headers=headers, proxies=proxies, timeout=5)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from {image_url}: {e}")
        return None
