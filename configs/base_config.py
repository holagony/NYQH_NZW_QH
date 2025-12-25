import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

# configs/base_config.py

class BaseConfig:
    """基础配置类 - 支持跨区域配置"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_cache = {}
    
    def get_config(self, config_key: str, crop_code: str, zoning_type: str,element:str) -> Dict[str, Any]:
        """获取配置参数 - 支持通过配置键直接获取"""
        
        # 先从缓存中获取
        if config_key in self.config_cache:
            return self.config_cache[config_key]
        
        # 从文件加载
        config_file = self.config_dir / f"{config_key}.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.config_cache[config_key] = config
                return config
        else:
            # 如果指定配置不存在，尝试使用默认命名规则
            fallback_key = f"zoning_{config_key.split('_')[1]}_{crop_code}_{zoning_type}_{element}"
            fallback_file = self.config_dir / f"{fallback_key}.json"
            
            if fallback_file.exists():
                with open(fallback_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.config_cache[config_key] = config
                    return config
            else:
                # 返回空配置
                return {}
    
    def merge_configs(self, user_config: Dict[str, Any], 
                     config_key: str, crop_code: str, zoning_type: str,element:str) -> Dict[str, Any]:
        """合并用户配置和默认配置"""
        default_config = self.get_config(config_key, crop_code, zoning_type,element)
        
        # 深度合并配置
        merged_config = self._deep_merge_fixed(default_config, user_config)
        return merged_config
    
    
    def _deep_merge_fixed(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """
        修复的深度合并函数
        user_config中的参数优先，没有的话再从default_config中获取
        """
        result = default.copy()  # 先复制默认配置
        
        for key, user_value in user.items():
            if key in result:
                # 如果键在默认配置中存在
                default_value = result[key]
                
                if isinstance(default_value, dict) and isinstance(user_value, dict):
                    # 如果两者都是字典，递归合并
                    result[key] = self._deep_merge_fixed(default_value, user_value)
                else:
                    # 否则使用用户配置的值（优先）
                    result[key] = user_value
            else:
                # 如果键在默认配置中不存在，直接添加用户配置
                result[key] = user_value
        
        return result
    
    def _deep_merge_alternative(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """
        另一种实现方式 - 更清晰的深度合并
        """
        result = {}
        
        # 处理所有默认配置的键
        for key in set(default.keys()) | set(user.keys()):
            if key in user and key in default:
                # 键在两者中都存在
                default_val = default[key]
                user_val = user[key]
                
                if isinstance(default_val, dict) and isinstance(user_val, dict):
                    # 递归合并字典
                    result[key] = self._deep_merge_alternative(default_val, user_val)
                else:
                    # 用户配置优先
                    result[key] = user_val
            elif key in user:
                # 只在用户配置中存在
                result[key] = user[key]
            else:
                # 只在默认配置中存在
                result[key] = default[key]
        
        return result

