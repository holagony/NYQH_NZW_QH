import json
import yaml
from typing import Dict, Any
from pathlib import Path

class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
    
    def load_crop_config(self, crop_code: str) -> Dict[str, Any]:
        """加载农作物配置"""
        config_path = self.config_dir / "crops" / f"{crop_code}.yaml"
        return self._load_config(config_path)
    
    def load_province_config(self, province_code: str) -> Dict[str, Any]:
        """加载省份配置"""
        config_path = self.config_dir / "provinces" / f"{province_code}.yaml"
        return self._load_config(config_path)
    
    def load_zoning_config(self, zoning_type: str) -> Dict[str, Any]:
        """加载区划类型配置"""
        config_path = self.config_dir / "zoning_types" / f"{zoning_type}.yaml"
        return self._load_config(config_path)
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """加载配置文件"""
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        if config_path.suffix == ".json":
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif config_path.suffix in [".yaml", ".yml"]:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")