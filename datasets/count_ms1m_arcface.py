#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ms1m-arcface 디렉토리를 탐색하여 하위 폴더 개수와 이미지 파일 개수를 출력하는 스크립트
"""

import os
from pathlib import Path


def count_directories_and_images(root_dir):
    """
    디렉토리를 탐색하여 하위 폴더 개수와 이미지 파일 개수를 반환

    Args:
        root_dir: 탐색할 루트 디렉토리 경로

    Returns:
        tuple: (하위 폴더 개수, 총 이미지 파일 개수)
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"오류: 디렉토리가 존재하지 않습니다: {root_dir}")
        return 0, 0

    # 이미지 파일 확장자
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".tiff",
        ".tif",
        ".webp",
    }

    subdir_count = 0
    total_image_count = 0

    # 루트 디렉토리의 직접 하위 폴더들을 탐색
    for item in root_path.iterdir():
        if item.is_dir():
            subdir_count += 1
            # 각 하위 폴더에서 이미지 파일 개수 세기
            image_count = 0
            try:
                for file in item.iterdir():
                    if file.is_file() and file.suffix.lower() in image_extensions:
                        image_count += 1
                total_image_count += image_count
            except PermissionError:
                print(f"경고: 접근 권한이 없습니다: {item}")
            except Exception as e:
                print(f"경고: {item} 처리 중 오류 발생: {e}")

    return subdir_count, total_image_count


def main():
    # ms1m-arcface 디렉토리 경로
    base_dir = "/purestorage/AILAB/AI_2/yjhwang/work/face/torch-insightface/datasets"
    target_dir = os.path.join(base_dir, "ms1m-arcface")

    print(f"디렉토리 탐색 중: {target_dir}")
    print("-" * 60)

    subdir_count, image_count = count_directories_and_images(target_dir)

    print(f"하위 폴더 개수: {subdir_count:,}")
    print(f"총 이미지 파일 개수: {image_count:,}")
    print("-" * 60)


if __name__ == "__main__":
    main()
