"""
Configuration file for QGNN Layer Study
"""
import os

# Base directory
BASE_DIR = '/Users/zhulinghua/Dropbox/UW_research/QML/QGCNN_Molecule/Layer_system'

# Data configuration
DATA_CONFIG = {
    'csv_file': os.path.join(BASE_DIR, 'qm9/qm9_2000.csv'),
    'max_molecules': 2000,
}

# Model configuration
MODEL_CONFIG = {
    'n_qubits': 12,
    'use_master': False,
}

def get_training_config(n_layers):
    """根据层数获取训练配置"""
    # 基础配置
    base_config = {
        'batch_size': 16,
        'learning_rate': 0.01,
        'weight_decay': 1e-5,
        'lr_schedule': True,
        'lr_patience': 5,
        'lr_factor': 0.5,
        'lr_min': 1e-5,
        'early_stopping': True,
        'test_split': 0.2,
        'val_split': 0.2,
        'gradient_clip': 2.0,
        'save_every_epoch': 10,
    }
    
    # Phase 2 的动态配置
    if n_layers == 1:
        base_config['n_epochs'] = 40
        base_config['patience'] = 10
    elif n_layers == 2:
        base_config['n_epochs'] = 45
        base_config['patience'] = 10
    elif n_layers == 3:
        base_config['n_epochs'] = 48
        base_config['patience'] = 12
    elif n_layers == 4:
        base_config['n_epochs'] = 35  # 减少epochs
        base_config['patience'] = 8   # 更激进的早停
    else:  # n_layers >= 5
        base_config['n_epochs'] = 25
        base_config['patience'] = 5
    
    return base_config

# EXPERIMENT_CONFIG
EXPERIMENT_CONFIG = {
    # Phase 1: Initial exploration
    'phase1': {
        'layers': [1, 2, 3, 4, 5],
        'seeds': [42],
        'skip_if_poor': True,
        'poor_threshold': 0.5,
    },
    
    # Phase 2: Multi-seed validation
    'phase2': {
        'layers': [1, 2, 3, 4],  # 只测试1-4层
        'seeds': [456],  # 3个种子
    },
    
    # Phase 3: Deep analysis
    'phase3': {
        'layers': None,
        'seeds': [42, 123, 456, 789, 999],
        'additional_metrics': True,
    },
    
    # Output settings
    'output_base_dir': os.path.join(BASE_DIR, 'experiments'),
    'save_intermediate': True,
    'verbose': True,
}

# 合并所有配置
def get_config(n_layers=3, seed=42):
    """获取特定实验的配置"""
    config = {}
    config.update(DATA_CONFIG)
    config.update(MODEL_CONFIG)
    config.update(get_training_config(n_layers))  # 使用动态训练配置
    
    # 更新特定参数
    config['n_layers'] = n_layers
    config['seed'] = seed
    
    return config
    