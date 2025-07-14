import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support, 
    multilabel_confusion_matrix,
    hamming_loss,
    jaccard_score,
    accuracy_score,
    classification_report
)
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

COLORS = {
    'blue': '#4169E1',
    'teal': '#008B8B', 
    'green': '#32CD32',
    'purple': '#9370DB', 
    'navy': '#191970', 
    'sea_green': '#2E8B57',
    'slate_blue': '#6A5ACD',
    'dark_green': '#006400',
    'indigo': '#4B0082',
    'steel_blue': '#4682B4'
}

PALETTE_MAIN = [COLORS['blue'], COLORS['green'], COLORS['purple']]
PALETTE_EXTENDED = [COLORS['blue'], COLORS['teal'], COLORS['green'], 
                   COLORS['purple'], COLORS['navy'], COLORS['sea_green']]
PALETTE_MULTI = [COLORS['sea_green'], COLORS['purple'], COLORS['blue'], 
                COLORS['teal'], COLORS['slate_blue']]

def create_custom_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    colors = [COLORS['blue'], COLORS['teal'], COLORS['green'], COLORS['purple']]
    return LinearSegmentedColormap.from_list('custom_bgp', colors)

class MedicalDiagnosisEvaluator:
    def __init__(self, results_files: List[str], disease_vocabulary_file: str, model_names: List[str] = None):
        self.results_files = results_files
        self.disease_vocabulary_file = disease_vocabulary_file
        self.disease_vocab = self._load_disease_vocabulary()
        self.model_names = model_names or [f"Model_{i+1}" for i in range(len(results_files))]
        self.data = self._load_and_process_data()
        
    def _load_disease_vocabulary(self) -> List[str]:
        with open(self.disease_vocabulary_file, 'r') as f:
            return json.load(f)
    
    def _load_and_process_data(self) -> Dict[str, pd.DataFrame]:
        all_data = {}
        
        for i, results_file in enumerate(self.results_files):
            model_name = self.model_names[i]
            
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            processed_data = []
            for item in data:
                processed_item = {
                    'case_id': item['case_id'],
                    'input_finding': item['input_finding'],
                    'ground_truth_raw': item['ground_truth'],
                    'predicted_labels_raw': item['predicted_labels'],
                    'raw_response': item['raw_response'],
                    'model_name': model_name
                }
                
                processed_item['ground_truth'] = self._parse_labels(item['ground_truth'])
                processed_item['predicted_labels'] = self._parse_labels(item['predicted_labels'])
                processed_data.append(processed_item)
            
            all_data[model_name] = pd.DataFrame(processed_data)
        
        return all_data
    
    def _parse_labels(self, label_string: str) -> List[str]:
        if not label_string or pd.isna(label_string):
            return []
        

        labels = [label.strip() for label in label_string.split(',')]
        seen = set()
        clean_labels = []
        for label in labels:
            if label and label not in seen and label in self.disease_vocab:
                clean_labels.append(label)
                seen.add(label)
        
        return clean_labels
    
    def _get_combined_data(self) -> pd.DataFrame:
        return pd.concat(self.data.values(), ignore_index=True)
    
    def _create_binary_matrix(self, model_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        if model_name:
            data = self.data[model_name]
        else:
            data = self._get_combined_data()
            
        mlb = MultiLabelBinarizer(classes=self.disease_vocab)
        y_true = mlb.fit_transform(data['ground_truth'])
        y_pred = mlb.transform(data['predicted_labels'])
        
        return y_true, y_pred, mlb
    
    def calculate_classification_metrics(self, model_name: str = None) -> Dict:
        y_true, y_pred, mlb = self._create_binary_matrix(model_name)
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        return {
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }
    
    def calculate_multilabel_metrics(self, model_name: str = None) -> Dict:
        """Calculate multi-label specific metrics"""
        y_true, y_pred, mlb = self._create_binary_matrix(model_name)
        
        metrics = {}
        exact_match = accuracy_score(y_true, y_pred)
        metrics['exact_match_ratio'] = exact_match
        hamming_loss_val = hamming_loss(y_true, y_pred)
        metrics['hamming_loss'] = hamming_loss_val
        jaccard_samples = jaccard_score(y_true, y_pred, average='samples', zero_division=0)
        jaccard_micro = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
        jaccard_macro = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics['jaccard_samples'] = jaccard_samples
        metrics['jaccard_micro'] = jaccard_micro
        metrics['jaccard_macro'] = jaccard_macro
        
        return metrics
    
    def calculate_medical_specific_metrics(self, model_name: str = None) -> Dict:
        if model_name:
            data = self.data[model_name]
        else:
            data = self._get_combined_data()
            
        metrics = {}

        case_jaccard_scores = []
        case_f1_scores = []
        case_precision_scores = []
        case_recall_scores = []
        
        for idx, row in data.iterrows():
            gt_set = set(row['ground_truth'])
            pred_set = set(row['predicted_labels'])
            
            if len(gt_set) == 0 and len(pred_set) == 0:
                case_jaccard_scores.append(1.0)
                case_f1_scores.append(1.0)
                case_precision_scores.append(1.0)
                case_recall_scores.append(1.0)
            elif len(gt_set) == 0:
                case_jaccard_scores.append(0.0)
                case_f1_scores.append(0.0)
                case_precision_scores.append(0.0)
                case_recall_scores.append(0.0)
            elif len(pred_set) == 0:
                case_jaccard_scores.append(0.0)
                case_f1_scores.append(0.0)
                case_precision_scores.append(0.0)
                case_recall_scores.append(0.0)
            else:
                intersection = len(gt_set.intersection(pred_set))
                union = len(gt_set.union(pred_set))
                jaccard = intersection / union if union > 0 else 0
                case_jaccard_scores.append(jaccard)
                precision = intersection / len(pred_set) if len(pred_set) > 0 else 0
                recall = intersection / len(gt_set) if len(gt_set) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                case_precision_scores.append(precision)
                case_recall_scores.append(recall)
                case_f1_scores.append(f1)
        
        metrics['avg_case_jaccard'] = np.mean(case_jaccard_scores)
        metrics['avg_case_f1'] = np.mean(case_f1_scores)
        metrics['avg_case_precision'] = np.mean(case_precision_scores)
        metrics['avg_case_recall'] = np.mean(case_recall_scores)

        for k in [1, 3, 5]:
            top_k_acc = self._calculate_top_k_accuracy(k, model_name)
            metrics[f'top_{k}_accuracy'] = top_k_acc
        
        return metrics
    
    def _calculate_top_k_accuracy(self, k: int, model_name: str = None) -> float:
        if model_name:
            data = self.data[model_name]
        else:
            data = self._get_combined_data()
            
        correct = 0
        total = 0
        
        for idx, row in data.iterrows():
            gt_set = set(row['ground_truth'])
            pred_list = row['predicted_labels'][:k]  # Take top k predictions
            
            if len(gt_set) > 0:
                total += 1
                if any(pred in gt_set for pred in pred_list):
                    correct += 1
        
        return correct / total if total > 0 else 0
    
    def analyze_label_distribution(self, model_name: str = None) -> Dict:
        """Analyze label distribution and frequency"""
        if model_name:
            data = self.data[model_name]
        else:
            data = self._get_combined_data()

        gt_counter = Counter()
        for labels in data['ground_truth']:
            gt_counter.update(labels)

        pred_counter = Counter()
        for labels in data['predicted_labels']:
            pred_counter.update(labels)

        label_metrics = {}
        y_true, y_pred, mlb = self._create_binary_matrix(model_name)
        
        for i, label in enumerate(mlb.classes_):
            y_true_label = y_true[:, i]
            y_pred_label = y_pred[:, i]
            
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true_label, y_pred_label, average='binary', zero_division=0
            )
            
            label_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support,
                'gt_frequency': gt_counter.get(label, 0),
                'pred_frequency': pred_counter.get(label, 0)
            }
        
        return {
            'label_metrics': label_metrics,
            'gt_distribution': dict(gt_counter),
            'pred_distribution': dict(pred_counter)
        }
    
    def error_analysis(self, model_name: str = None) -> Dict:
        if model_name:
            data = self.data[model_name]
        else:
            data = self._get_combined_data()
            
        error_analysis = {
            'false_positives': defaultdict(int),
            'false_negatives': defaultdict(int),
            'correct_predictions': defaultdict(int),
            'cases_with_errors': []
        }
        
        for idx, row in data.iterrows():
            gt_set = set(row['ground_truth'])
            pred_set = set(row['predicted_labels'])
            fp = pred_set - gt_set
            for label in fp:
                error_analysis['false_positives'][label] += 1
            fn = gt_set - pred_set
            for label in fn:
                error_analysis['false_negatives'][label] += 1
            correct = gt_set.intersection(pred_set)
            for label in correct:
                error_analysis['correct_predictions'][label] += 1
            
            if fp or fn:
                error_analysis['cases_with_errors'].append({
                    'case_id': row['case_id'],
                    'ground_truth': list(gt_set),
                    'predictions': list(pred_set),
                    'false_positives': list(fp),
                    'false_negatives': list(fn),
                    'input_finding': row['input_finding'][:200] + '...' if len(row['input_finding']) > 200 else row['input_finding']
                })
        
        return error_analysis
    
    def report(self) -> Dict:
        print("Calculating evaluation metrics...")
        
        all_reports = {}
        
        for model_name in self.model_names:
            data = self.data[model_name]
            
            report = {
                'dataset_info': {
                    'total_cases': len(data),
                    'total_diseases': len(self.disease_vocab),
                    'avg_labels_per_case_gt': np.mean([len(labels) for labels in data['ground_truth']]),
                    'avg_labels_per_case_pred': np.mean([len(labels) for labels in data['predicted_labels']])
                },
                'classification_metrics': self.calculate_classification_metrics(model_name),
                'multilabel_metrics': self.calculate_multilabel_metrics(model_name),
                'avg_metrics': self.calculate_medical_specific_metrics(model_name),
                'label_analysis': self.analyze_label_distribution(model_name),
                'error_analysis': self.error_analysis(model_name)
            }
            
            all_reports[model_name] = report
        
        return all_reports
    
    
    def save_report(self, all_reports: Dict, output_file: str):
        reports_copy = {}
        
        for model_name, report in all_reports.items():
            report_copy = report.copy()
            report_copy['error_analysis']['false_positives'] = dict(report_copy['error_analysis']['false_positives'])
            report_copy['error_analysis']['false_negatives'] = dict(report_copy['error_analysis']['false_negatives'])
            report_copy['error_analysis']['correct_predictions'] = dict(report_copy['error_analysis']['correct_predictions'])
            reports_copy[model_name] = report_copy
        
        with open(output_file, 'w') as f:
            json.dump(reports_copy, f, indent=2)
        
        print(f"\nreport saved to: {output_file}")
    
    def plot(self, all_reports: Dict, output_dir: str = "evaluation_plots"):
        import os
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('default')
        sns.set_palette("husl")
        
        self._create_comparison_plots(all_reports, output_dir)
        for model_name, report in all_reports.items():
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            self._create_individual_plots(report, model_dir, model_name)
        
        print(f"Visualizations saved to: {output_dir}/")
    
    def _create_comparison_plots(self, all_reports: Dict, output_dir: str):
        sns.set_palette(PALETTE_EXTENDED)
        plt.figure(figsize=(12, 8))
        metrics_data = []
        for model_name, report in all_reports.items():
            classification = report['classification_metrics']
            metrics_data.extend([
                {'Model': model_name, 'Metric': 'Precision', 'Score': classification['precision_micro'], 'Type': 'Micro'},
                {'Model': model_name, 'Metric': 'Recall', 'Score': classification['recall_micro'], 'Type': 'Micro'},
                {'Model': model_name, 'Metric': 'F1-Score', 'Score': classification['f1_micro'], 'Type': 'Micro'},
                {'Model': model_name, 'Metric': 'Precision', 'Score': classification['precision_macro'], 'Type': 'Macro'},
                {'Model': model_name, 'Metric': 'Recall', 'Score': classification['recall_macro'], 'Type': 'Macro'},
                {'Model': model_name, 'Metric': 'F1-Score', 'Score': classification['f1_macro'], 'Type': 'Macro'}
            ])
        
        metrics_df = pd.DataFrame(metrics_data)
        sns.barplot(data=metrics_df, x='Metric', y='Score', hue='Model')
        plt.title('Classification Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/classification_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        plt.figure(figsize=(12, 8))
        multilabel_data = []
        for model_name, report in all_reports.items():
            multilabel = report['multilabel_metrics']
            multilabel_data.extend([
                {'Model': model_name, 'Metric': 'Exact Match', 'Score': multilabel['exact_match_ratio']},
                {'Model': model_name, 'Metric': 'Jaccard (Samples)', 'Score': multilabel['jaccard_samples']},
                {'Model': model_name, 'Metric': 'Jaccard (Micro)', 'Score': multilabel['jaccard_micro']},
                {'Model': model_name, 'Metric': 'Jaccard (Macro)', 'Score': multilabel['jaccard_macro']}
            ])
        
        multilabel_df = pd.DataFrame(multilabel_data)
        sns.barplot(data=multilabel_df, x='Metric', y='Score', hue='Model')
        plt.title('Multi-label Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/multilabel_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        

        plt.figure(figsize=(10, 8))
        k_vals = [1, 3, 5]
        colors = PALETTE_EXTENDED[:len(all_reports)]
        
        for i, (model_name, report) in enumerate(all_reports.items()):
            medical = report['avg_metrics']
            topk_vals = [medical['top_1_accuracy'], medical['top_3_accuracy'], medical['top_5_accuracy']]
            plt.plot(k_vals, topk_vals, marker='o', linewidth=3, markersize=8, label=model_name, color=colors[i])
        
        plt.title('Top-k Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Top-k', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(k_vals)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/topk_accuracy_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        

        plt.figure(figsize=(12, 8))
        medical_data = []
        for model_name, report in all_reports.items():
            medical = report['avg_metrics']
            medical_data.extend([
                {'Model': model_name, 'Metric': 'Avg Case Jaccard', 'Score': medical['avg_case_jaccard']},
                {'Model': model_name, 'Metric': 'Avg Case F1-Score', 'Score': medical['avg_case_f1']},
                {'Model': model_name, 'Metric': 'Avg Case Precision', 'Score': medical['avg_case_precision']},
                {'Model': model_name, 'Metric': 'Avg Case Recall', 'Score': medical['avg_case_recall']}
            ])
        
        medical_df = pd.DataFrame(medical_data)
        sns.barplot(data=medical_df, x='Metric', y='Score', hue='Model')
        plt.title('Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/avg_metrics_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        plt.figure(figsize=(10, 6))
        hamming_data = []
        for model_name, report in all_reports.items():
            multilabel = report['multilabel_metrics']
            hamming_data.append({'Model': model_name, 'Hamming Loss': multilabel['hamming_loss']})
        
        hamming_df = pd.DataFrame(hamming_data)
        bars = plt.bar(hamming_df['Model'], hamming_df['Hamming Loss'], color=COLORS['purple'], alpha=0.8)
        plt.title('Hamming Loss Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Hamming Loss', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        

        for bar, val in zip(bars, hamming_df['Hamming Loss']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/hamming_loss_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _create_individual_plots(self, report: Dict, output_dir: str, model_name: str):
        sns.set_palette(PALETTE_MAIN)
        plt.figure(figsize=(10, 6))
        classification_metrics = report['classification_metrics']
        metrics_names = ['Precision', 'Recall', 'F1-Score']
        micro_vals = [classification_metrics['precision_micro'], classification_metrics['recall_micro'], classification_metrics['f1_micro']]
        macro_vals = [classification_metrics['precision_macro'], classification_metrics['recall_macro'], classification_metrics['f1_macro']]
        
        metrics_df = pd.DataFrame({
            'Metric': metrics_names * 2,
            'Score': micro_vals + macro_vals,
            'Average Type': ['Micro (Overall)'] * 3 + ['Macro (Per Class)'] * 3
        })
        
        sns.barplot(data=metrics_df, x='Metric', y='Score', hue='Average Type')
        plt.title(f'{model_name} - Classification Performance Metrics', fontsize=14, fontweight='bold', pad=20)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{model_name}_classification.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        plt.figure(figsize=(12, 6))
        multilabel_metrics = report['multilabel_metrics']
        ml_names = ['Exact Match\nRatio', 'Hamming Loss\n(Error Rate)', 'Jaccard\n(Samples)', 'Jaccard\n(Micro)', 'Jaccard\n(Macro)']
        ml_vals = [multilabel_metrics['exact_match_ratio'], multilabel_metrics['hamming_loss'], 
                   multilabel_metrics['jaccard_samples'], multilabel_metrics['jaccard_micro'], 
                   multilabel_metrics['jaccard_macro']]
        
        colors = PALETTE_MULTI
        bars = plt.bar(ml_names, ml_vals, color=colors, alpha=0.8)
        plt.title(f'{model_name} - Multi-label Classification Metrics', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, ml_vals):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{model_name}_multilabel.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        plt.figure(figsize=(8, 6))
        avg_metrics = report['avg_metrics']
        k_vals = [1, 3, 5]
        topk_vals = [avg_metrics['top_1_accuracy'], avg_metrics['top_3_accuracy'], avg_metrics['top_5_accuracy']]
        
        colors_topk = PALETTE_MAIN
        bars = plt.bar([f'Top-{k}' for k in k_vals], topk_vals, color=colors_topk, alpha=0.8)
        plt.title(f'{model_name} - Top-k Accuracy Performance', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, topk_vals):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{model_name}_topk_accuracy.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        medical_data = {
            'Avg Case Jaccard': avg_metrics['avg_case_jaccard'],
            'Avg Case F1-Score': avg_metrics['avg_case_f1'],
            'Avg Case Precision': avg_metrics['avg_case_precision'],
            'Avg Case Recall': avg_metrics['avg_case_recall']
        }
        
        bars = plt.bar(medical_data.keys(), medical_data.values(), color='#9370DB', alpha=0.8)
        plt.title(f'{model_name} - Average case Metrics', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        for bar, val in zip(bars, medical_data.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{model_name}_avg_metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        

        plt.figure(figsize=(12, 8))
        error_analysis = report['error_analysis']
        fp_top = sorted(error_analysis['false_positives'].items(), key=lambda x: x[1], reverse=True)[:10]
        
        if fp_top:
            labels, counts = zip(*fp_top)
            y_pos = np.arange(len(labels))
            bars = plt.barh(y_pos, counts, color=COLORS['teal'], alpha=0.8)
            plt.yticks(y_pos, labels)
            plt.xlabel('Error Count')
            plt.title(f'{model_name} - Top 10 False Positive Errors', fontsize=14, fontweight='bold', pad=20)
            plt.grid(True, alpha=0.3, axis='x')
            
            for bar, count in zip(bars, counts):
                plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                        f'{count}', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{model_name}_false_positives.png', dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        

        plt.figure(figsize=(12, 8))
        fn_top = sorted(error_analysis['false_negatives'].items(), key=lambda x: x[1], reverse=True)[:10]
        
        if fn_top:
            labels, counts = zip(*fn_top)
            y_pos = np.arange(len(labels))
            bars = plt.barh(y_pos, counts, color=COLORS['slate_blue'], alpha=0.8)
            plt.yticks(y_pos, labels)
            plt.xlabel('Error Count')
            plt.title(f'{model_name} - Top 10 False Negative Errors', fontsize=14, fontweight='bold', pad=20)
            plt.grid(True, alpha=0.3, axis='x')
            
            for bar, count in zip(bars, counts):
                plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                        f'{count}', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{model_name}_false_negatives.png', dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

        self._create_confusion_heatmap(report, output_dir)
    
    def _create_confusion_heatmap(self, report: Dict, output_dir: str):
        label_metrics = report['label_analysis']['label_metrics']
        

        top_diseases = sorted(label_metrics.items(), 
                            key=lambda x: x[1]['gt_frequency'], 
                            reverse=True)[:15]
        
        if not top_diseases:
            return
        
        diseases = [disease for disease, _ in top_diseases]
        precision_vals = [metrics['precision'] for _, metrics in top_diseases]
        recall_vals = [metrics['recall'] for _, metrics in top_diseases]
        f1_vals = [metrics['f1'] for _, metrics in top_diseases]
        support_vals = [metrics['gt_frequency'] for _, metrics in top_diseases]
        
        heatmap_data = pd.DataFrame({
            'Disease': diseases,
            'Precision': precision_vals,
            'Recall': recall_vals,
            'F1-Score': f1_vals,
            'Frequency': support_vals
        })
        
        plt.figure(figsize=(14, 8))
        perf_data = heatmap_data[['Precision', 'Recall', 'F1-Score']].T
        perf_data.columns = diseases
        
        sns.heatmap(perf_data, annot=True, fmt='.3f', cmap=create_custom_cmap(), 
                   cbar_kws={'label': 'Score'})
        plt.title('Disease-Specific Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Diseases', fontsize=12)
        plt.ylabel('Metrics', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/disease_performance_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        plt.figure(figsize=(10, 8))
        sorted_freq = heatmap_data.sort_values('Frequency', ascending=True)
        colors = PALETTE_EXTENDED * ((len(sorted_freq) // len(PALETTE_EXTENDED)) + 1)
        colors = colors[:len(sorted_freq)]
        bars = plt.barh(range(len(sorted_freq)), sorted_freq['Frequency'], 
                       color=colors)
        plt.yticks(range(len(sorted_freq)), sorted_freq['Disease'])
        plt.xlabel('Frequency in Ground Truth', fontsize=12)
        plt.title('Disease Frequency Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='x')
        
        for bar, freq in zip(bars, sorted_freq['Frequency']):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{freq}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/disease_frequency_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        plt.figure(figsize=(10, 8))
        colors = PALETTE_EXTENDED * ((len(heatmap_data) // len(PALETTE_EXTENDED)) + 1)
        colors = colors[:len(heatmap_data)]  # Trim to exact length
        
        plt.scatter(heatmap_data['Frequency'], heatmap_data['F1-Score'], 
                   s=100, c=colors, alpha=0.7, edgecolors='black')
        
        for i, disease in enumerate(diseases):
            plt.annotate(disease, 
                        (heatmap_data['Frequency'].iloc[i], heatmap_data['F1-Score'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, alpha=0.8)
        
        plt.xlabel('Disease Frequency', fontsize=12)
        plt.ylabel('F1-Score', fontsize=12)
        plt.title('F1-Score vs Disease Frequency', fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_vs_frequency.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <json_file1> [json_file2] [json_file3] ...")
        print("Example: python evaluate.py results/Qwen3-4B_qlora_results.json results/Qwen3-8B_results.json")
        sys.exit(1)
    
    results_files = sys.argv[1:]
    model_names = []
    for file_path in results_files:
        import os
        filename = os.path.basename(file_path)
        model_name = filename.replace('_results.json', '').replace('.json', '')
        model_names.append(model_name)
    evaluator = MedicalDiagnosisEvaluator(
        results_files,
        'processed_data/disease_vocabulary.json',
        model_names
    )
    
    all_reports = evaluator.report()
    evaluator.save_report(all_reports, 'evaluation_report.json')
    evaluator.plot(all_reports)
    
    print("\n Evaluation completed!")


if __name__ == "__main__":
    main() 