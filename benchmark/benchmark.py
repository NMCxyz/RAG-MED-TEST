"""
Benchmark Runner
----------------
Script để chạy benchmark hệ thống RAG y khoa trên nhiều bộ dữ liệu
và tổng hợp kết quả.
"""

import os
import json
import time
import argparse
import pandas as pd
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Thêm thư mục gốc vào path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các evaluator
from hotpotqa_evaluator import HotpotQAEvaluator

def run_all_benchmarks(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Chạy benchmark trên tất cả các bộ dữ liệu được chỉ định
    
    Args:
        args: Tham số dòng lệnh
        
    Returns:
        Kết quả benchmark tổng hợp
    """
    results = {}
    
    # Tạo thư mục kết quả
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Benchmark trên HotpotQA
    if "hotpotqa" in args.datasets:
        print("\n=== Running HotpotQA benchmark ===")
        hotpotqa_results = run_hotpotqa_benchmark(args)
        results["hotpotqa"] = hotpotqa_results
    
    # Có thể thêm các bộ dữ liệu khác ở đây
    
    # Tổng hợp và hiển thị kết quả
    summarize_results(results, args.output_dir)
    
    return results

def run_hotpotqa_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Chạy benchmark trên tập dữ liệu HotpotQA
    
    Args:
        args: Tham số dòng lệnh
        
    Returns:
        Kết quả benchmark
    """
    # Khởi tạo evaluator
    hotpotqa_evaluator = HotpotQAEvaluator(
        device=args.device,
        output_dir=os.path.join(args.output_dir, "hotpotqa")
    )
    
    # Tải dữ liệu
    filter_domain = args.filter_domain.split(",") if args.filter_domain else None
    samples = hotpotqa_evaluator.load_hotpotqa_data(
        split=args.split,
        subset=args.num_samples,
        filter_domain=filter_domain
    )
    
    # Chạy benchmark
    results = hotpotqa_evaluator.run_benchmark(
        samples=samples,
        timeout=args.timeout,
        save_results=True
    )
    
    return results

def summarize_results(results: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    """
    Tổng hợp và hiển thị kết quả benchmark
    
    Args:
        results: Kết quả benchmark từ các dataset
        output_dir: Thư mục lưu kết quả
    """
    print("\n=== BENCHMARK SUMMARY ===")
    
    # Thu thập metrics từ tất cả các dataset
    all_metrics = []
    for dataset_name, dataset_results in results.items():
        metrics = dataset_results["metrics"]
        metrics["dataset"] = dataset_name
        all_metrics.append(metrics)
    
    # Tạo DataFrame từ metrics
    metrics_df = pd.DataFrame(all_metrics)
    
    # Hiển thị bảng kết quả tóm tắt
    summary_columns = ["dataset", "success_rate", "f1", "rougeL", "bleu", "avg_latency"]
    summary_df = metrics_df[summary_columns].sort_values("f1", ascending=False)
    
    print("\nResults Summary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Lưu kết quả tổng hợp
    summary_path = os.path.join(output_dir, f"benchmark_summary_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    metrics_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    
    # Visualize results if there are multiple datasets
    if len(results) > 1:
        create_visualization(metrics_df, output_dir)

def create_visualization(metrics_df: pd.DataFrame, output_dir: str) -> None:
    """
    Tạo các biểu đồ trực quan hóa kết quả benchmark
    
    Args:
        metrics_df: DataFrame chứa metrics
        output_dir: Thư mục lưu biểu đồ
    """
    try:
        # Tạo thư mục cho biểu đồ
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Set style
        sns.set(style="whitegrid")
        
        # 1. F1 Score Comparison
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x="dataset", y="f1", data=metrics_df)
        ax.set_title("F1 Score by Dataset", fontsize=15)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_xlabel("Dataset", fontsize=12)
        
        # Add value labels on bars
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.4f}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "f1_comparison.png"))
        
        # 2. Multiple Metrics Comparison
        plt.figure(figsize=(12, 8))
        metrics_to_plot = ["f1", "rougeL", "bleu", "exact_match"]
        
        # Melt the dataframe for easier plotting
        plot_df = pd.melt(metrics_df, 
                          id_vars=["dataset"], 
                          value_vars=metrics_to_plot,
                          var_name="Metric", 
                          value_name="Score")
        
        ax = sns.barplot(x="dataset", y="Score", hue="Metric", data=plot_df)
        ax.set_title("Performance Metrics by Dataset", fontsize=15)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_xlabel("Dataset", fontsize=12)
        ax.legend(title="Metric")
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "metrics_comparison.png"))
        
        # 3. Latency Comparison
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x="dataset", y="avg_latency", data=metrics_df)
        ax.set_title("Average Latency by Dataset", fontsize=15)
        ax.set_ylabel("Latency (seconds)", fontsize=12)
        ax.set_xlabel("Dataset", fontsize=12)
        
        # Add value labels on bars
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}s", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "latency_comparison.png"))
        
        print(f"Visualizations saved to {viz_dir}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks for the medical RAG system")
    
    # Tham số chung
    parser.add_argument("--datasets", type=str, default="hotpotqa", 
                        help="Comma-separated list of datasets to benchmark (hotpotqa)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to evaluate per dataset")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout per query in seconds")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run models (cuda or cpu)")
    
    # Tham số cho HotpotQA
    parser.add_argument("--split", type=str, default="validation",
                        help="Dataset split to use (train, validation)")
    parser.add_argument("--filter_domain", type=str, default="medicine",
                        help="Filter by domain (comma-separated)")
    
    args = parser.parse_args()
    
    # Chuyển đổi danh sách datasets
    args.datasets = args.datasets.split(",")
    
    # Chạy tất cả benchmarks
    start_time = time.time()
    results = run_all_benchmarks(args)
    total_time = time.time() - start_time
    
    print(f"\nTotal benchmark time: {total_time:.2f} seconds")
    
if __name__ == "__main__":
    main() 