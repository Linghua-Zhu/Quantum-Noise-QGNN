import sys
sys.path.append('.')
from config import get_config
from qgnn_base import train_qgnn_model
import numpy as np
import torch
from datetime import datetime
import pandas as pd
import os
import argparse

# === 命令行参数 ===
parser = argparse.ArgumentParser(description='Screen for strong noise cases')
parser.add_argument('--batch', type=int, required=True, help='Batch number (1 or 2)')
parser.add_argument('--seeds', type=int, nargs='+', default=None, 
                    help='Seeds to test (default: auto-generate)')
args = parser.parse_args()

# === 实验配置 ===
batch_num = args.batch

# 生成或使用指定的种子
if args.seeds:
    test_seeds = args.seeds
else:
    # 为两个batch生成不同的种子
    np.random.seed(42 + batch_num)  # 确保可重复性
    test_seeds = np.random.randint(1000, 9999, 10).tolist()

noise_levels = [0, 0.005, 0.01, 0.015]  # 你指定的噪声水平
n_layers = 1
n_epochs_override = 20
patience_override = 8

# 文件路径 - 每个batch独立的文件
base_dir = '/Users/zhulinghua/Dropbox/UW_research/QML/QGCNN_Molecule/Layer_system/experiments/select_strong_cases'
os.makedirs(base_dir, exist_ok=True)

results_file = os.path.join(base_dir, f'batch{batch_num}_layer{n_layers}_screening.txt')
csv_file = os.path.join(base_dir, f'batch{batch_num}_layer{n_layers}_screening.csv')
summary_file = os.path.join(base_dir, f'batch{batch_num}_summary.txt')

# 初始化
all_results = []
positive_cases = []  # 追踪positive cases

# === 写入实验头信息 ===
with open(results_file, 'w') as f:
    f.write(f"{'='*80}\n")
    f.write(f"SCREENING FOR STRONG CASES - Batch {batch_num}\n")
    f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Configuration:\n")
    f.write(f"  - Batch: {batch_num}\n")
    f.write(f"  - Layers: {n_layers}\n")
    f.write(f"  - Seeds: {test_seeds}\n")
    f.write(f"  - Noise levels: {noise_levels}\n")
    f.write(f"  - Epochs: {n_epochs_override}\n")
    f.write(f"  - Smart early stop: YES (stop if noise=0.005 degrades)\n")
    f.write(f"{'='*80}\n\n")

print(f"\n{'='*60}")
print(f"Starting Batch {batch_num} - Screening for Strong Cases")
print(f"Seeds to test: {test_seeds}")
print(f"Smart early stop enabled")
print(f"{'='*60}\n")

# === 主实验循环 ===
for seed_idx, seed in enumerate(test_seeds, 1):
    print(f"\n{'='*50}")
    print(f"[Batch {batch_num}] Testing Seed {seed} ({seed_idx}/{len(test_seeds)})")
    print(f"{'='*50}")
    
    seed_results = {}
    baseline_r2 = None
    skip_remaining = False  # 早停标志
    
    with open(results_file, 'a') as f:
        f.write(f"\n--- Seed {seed} ---\n")
    
    for noise_idx, noise_level in enumerate(noise_levels):
        if skip_remaining and noise_level > 0.005:
            print(f"  ⚠ Skipping noise={noise_level} (early stop triggered)")
            with open(results_file, 'a') as f:
                f.write(f"  Noise={noise_level}: SKIPPED (early stop)\n")
            continue
        
        print(f"\n  Testing noise={noise_level}")
        
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # 获取配置
        config = get_config(n_layers=n_layers, seed=seed)
        config['n_epochs'] = n_epochs_override
        config['patience'] = patience_override
        config['noise_level'] = noise_level
        
        # 记录时间
        start_time = datetime.now()
        
        try:
            # 运行训练
            output_dir = f"{base_dir}/runs/seed_{seed}_noise_{noise_level}"
            results = train_qgnn_model(config, output_dir)
            
            # 计算时间
            train_time = (datetime.now() - start_time).total_seconds() / 60
            
            # 记录结果
            seed_results[noise_level] = {
                'r2': results['test_r2'],
                'mae': results['test_mae'],
                'time': train_time,
                'epochs': results['epochs_trained']
            }
            
            # 记录基准
            if noise_level == 0:
                baseline_r2 = results['test_r2']
            
            # 计算改善
            if baseline_r2:
                improvement = ((results['test_r2'] - baseline_r2) / baseline_r2 * 100)
            else:
                improvement = 0
            
            print(f"    R²={results['test_r2']:.4f}, Improvement={improvement:+.1f}%")
            
            # 写入文件
            with open(results_file, 'a') as f:
                f.write(f"  Noise={noise_level:.3f}: R²={results['test_r2']:.4f}, "
                       f"MAE={results['test_mae']:.4f}, Δ={improvement:+.1f}%\n")
            
            # === 智能早停逻辑 ===
            if noise_level == 0.005 and baseline_r2:
                if improvement < -2:  # 如果0.005时性能下降超过2%
                    skip_remaining = True
                    print(f"  ⚡ Early stop triggered! Performance degraded at noise=0.005")
                    with open(results_file, 'a') as f:
                        f.write(f"  → Early stop: performance degraded\n")
                elif improvement > 2:  # 如果有改善，标记为positive
                    print(f"  ⭐ Positive case found! Improvement={improvement:.1f}%")
                    
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            with open(results_file, 'a') as f:
                f.write(f"  Noise={noise_level}: ERROR - {str(e)[:50]}\n")
            continue
    
    # === 分析该种子的结果 ===
    if baseline_r2 and len(seed_results) > 1:
        best_noise = max(seed_results.items(), key=lambda x: x[1]['r2'])
        best_improvement = ((best_noise[1]['r2'] - baseline_r2) / baseline_r2 * 100)
        
        # 判断是否是positive case
        is_positive = best_improvement > 2 and best_noise[0] > 0
        
        summary = {
            'seed': seed,
            'baseline_r2': baseline_r2,
            'best_noise': best_noise[0],
            'best_r2': best_noise[1]['r2'],
            'improvement': best_improvement,
            'is_positive': is_positive,
            'pattern': 'Noise Helps' if is_positive else 'Noise Hurts'
        }
        
        all_results.append(summary)
        
        if is_positive:
            positive_cases.append(summary)
            print(f"\n  ✅ POSITIVE CASE CONFIRMED!")
        
        with open(results_file, 'a') as f:
            f.write(f"  Summary: Best @ noise={best_noise[0]:.3f}, "
                   f"R²={best_noise[1]['r2']:.4f} ({best_improvement:+.1f}%)\n")
            f.write(f"  Pattern: {summary['pattern']}\n")
    
    # 实时保存CSV
    if all_results:
        pd.DataFrame(all_results).to_csv(csv_file, index=False)

# === 生成最终总结 ===
print(f"\n{'='*60}")
print(f"Batch {batch_num} Complete - Generating Summary")
print(f"{'='*60}")

with open(summary_file, 'w') as f:
    f.write(f"BATCH {batch_num} SUMMARY\n")
    f.write(f"{'='*50}\n")
    f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    if positive_cases:
        f.write(f"POSITIVE CASES FOUND: {len(positive_cases)}\n")
        f.write(f"{'='*50}\n")
        
        # 排序positive cases
        positive_cases.sort(key=lambda x: x['improvement'], reverse=True)
        
        f.write(f"{'Rank':<6} {'Seed':<8} {'Best Noise':<12} {'R²':<10} {'Improvement':<12}\n")
        f.write(f"{'-'*50}\n")
        
        for idx, case in enumerate(positive_cases, 1):
            f.write(f"{idx:<6} {case['seed']:<8} {case['best_noise']:<12.3f} "
                   f"{case['best_r2']:<10.4f} {case['improvement']:<+12.1f}%\n")
        
        # 统计
        f.write(f"\n{'='*50}\n")
        f.write(f"Success Rate: {len(positive_cases)}/{len(test_seeds)} "
               f"({len(positive_cases)/len(test_seeds)*100:.1f}%)\n")
        
        improvements = [c['improvement'] for c in positive_cases]
        f.write(f"Mean Improvement: {np.mean(improvements):.2f}%\n")
        f.write(f"Max Improvement: {max(improvements):.2f}%\n")
        
        optimal_noises = [c['best_noise'] for c in positive_cases]
        f.write(f"Optimal Noise Range: {min(optimal_noises):.3f} - {max(optimal_noises):.3f}\n")
        
    else:
        f.write("No positive cases found in this batch.\n")
    
    f.write(f"\n{'='*50}\n")
    f.write(f"Total seeds tested: {len(test_seeds)}\n")
    f.write(f"Results saved to: {results_file}\n")

print(f"\n✅ Batch {batch_num} Complete!")
print(f"Positive cases found: {len(positive_cases)}")
if positive_cases:
    print(f"Best improvement: {max(c['improvement'] for c in positive_cases):.1f}%")
print(f"Results saved to: {base_dir}")

