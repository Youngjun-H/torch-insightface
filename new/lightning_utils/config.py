"""
Config loading utility for EdgeFace training
"""

import importlib
import os.path as osp
import sys

from easydict import EasyDict as edict


def get_config(config_file: str) -> edict:
    """
    Load config from Python file

    Args:
        config_file: Config file path (e.g., 'configs/edgeface_xs_gamma_06.py' or 절대 경로)

    Returns:
        Config dictionary (EasyDict)
    """
    # Config 파일 경로 처리
    if osp.isabs(config_file):
        # 절대 경로인 경우
        config_path = config_file
        config_dir = osp.dirname(config_path)
        config_file = osp.basename(config_path)
        parent_dir = osp.dirname(config_dir)
    elif config_file.startswith("configs/"):
        # configs/로 시작하는 경우 (상대 경로)
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

    # Base config 로드 (없으면 기본값 사용)
    sys_path_backup = sys.path.copy()
    try:
        sys.path.insert(0, parent_dir)

        # base config가 있으면 로드, 없으면 기본값 사용
        try:
            # python -m new.train으로 실행할 때 new.configs로 시도
            try:
                config_base = importlib.import_module(f"new.configs.base")
            except ImportError:
                config_base = importlib.import_module(f"configs.base")
            cfg = edict(config_base.config.copy())
        except ImportError:
            # base config가 없으면 기본값 사용
            cfg = edict()

        # Specific config 로드
        try:
            # python -m new.train으로 실행할 때 new.configs로 시도
            config_specific = importlib.import_module(f"new.configs.{temp_module_name}")
        except ImportError:
            config_specific = importlib.import_module(f"configs.{temp_module_name}")
        job_cfg = config_specific.config
        if isinstance(job_cfg, edict):
            cfg.update(dict(job_cfg))
        else:
            cfg.update(job_cfg)
        cfg = edict(cfg)
    finally:
        sys.path = sys_path_backup

    # Output 경로 설정
    if not hasattr(cfg, "output") or cfg.output is None:
        cfg.output = osp.join("work_dirs", temp_module_name)

    return cfg
