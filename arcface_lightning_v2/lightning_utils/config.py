"""
Config loading utility for ArcFace training
"""

import importlib
import os.path as osp
import sys

from easydict import EasyDict as edict


def get_config(config_file: str) -> edict:
    """
    Load config from Python file

    Args:
        config_file: Config file path (e.g., 'configs/ms1mv3_r50.py' or 절대 경로)

    Returns:
        Config dictionary (EasyDict)
    """
    # Config 파일 경로 처리
    if osp.isabs(config_file):
        # 절대 경로인 경우
        config_path = config_file
        config_dir = osp.dirname(config_path)
        config_file = osp.basename(config_path)
        # configs 디렉토리의 부모 디렉토리를 sys.path에 추가
        parent_dir = osp.dirname(config_dir)
    elif config_file.startswith("configs/"):
        # configs/로 시작하는 경우 (상대 경로)
        # arcface_lightning_v2 기준으로 찾기
        base_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
        config_path = osp.join(base_dir, config_file)
        config_dir = osp.dirname(config_path)
        config_file = osp.basename(config_file)
        parent_dir = base_dir
    else:
        # 파일명만 주어진 경우 (configs/ 디렉토리에서 찾기)
        base_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
        config_path = osp.join(base_dir, "configs", config_file)
        config_dir = osp.join(base_dir, "configs")
        config_file = osp.basename(config_file)
        parent_dir = base_dir

    # 모듈 이름 추출
    temp_module_name = osp.splitext(config_file)[0]

    # Base config 로드
    # configs.base를 import하려면 configs의 부모 디렉토리가 sys.path에 있어야 함
    sys_path_backup = sys.path.copy()
    try:
        sys.path.insert(0, parent_dir)
        config_base = importlib.import_module("configs.base")
        # copy()는 일반 dict를 반환하므로 edict()로 감싸야 함
        cfg = edict(config_base.config.copy())

        # Specific config 로드
        config_specific = importlib.import_module(f"configs.{temp_module_name}")
        job_cfg = config_specific.config
        # job_cfg가 EasyDict이면 dict로 변환 후 update
        if isinstance(job_cfg, edict):
            cfg.update(dict(job_cfg))
        else:
            cfg.update(job_cfg)
        # update 후에도 EasyDict 유지
        cfg = edict(cfg)
    finally:
        sys.path = sys_path_backup

    # Output 경로 설정
    if cfg.output is None:
        cfg.output = osp.join("work_dirs", temp_module_name)

    return cfg


def config_to_dict(cfg: edict) -> dict:
    """EasyDict를 일반 dict로 변환"""
    result = {}
    for key, value in cfg.items():
        if isinstance(value, edict):
            result[key] = config_to_dict(value)
        else:
            result[key] = value
    return result
