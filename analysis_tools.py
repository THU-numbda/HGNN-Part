"""
Visualization and analysis tools for hyperparameter optimization results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class HyperoptAnalyzer:
    """Comprehensive analyzer for hyperparameter optimization results"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.optuna_db = self.results_dir / "optuna_study.db"
        self.results_json = self.results_dir / "optimization_results.json"
        self.best_config_json = self.results_dir / "best_config.json"
        
        # Load data
        self.trials_df = self._load_trials_data()
        self.results_data = self._load_results_data()
        
    def _load_trials_data(self) -> pd.DataFrame:
        """Load trials data from Optuna database"""
        if not self.optuna_db.exists():
            print(f"Warning: Optuna database not found at {self.optuna_db}")
            return pd.DataFrame()
        
        try:
            conn = sqlite3.connect(self.optuna_db)
            
            # Get trial data with parameters and objectives
            query = """
            SELECT 
                t.trial_id,
                t.state,
                t.value as objective_0,
                tv1.value as objective_1,
                tv2.value as objective_2, 
                tv3.value as objective_3,
                t.datetime_start,
                t.datetime_complete
            FROM trials t
            LEFT JOIN trial_values tv1 ON t.trial_id = tv1.trial_id AND tv1.objective = 1
            LEFT JOIN trial_values tv2 ON t.trial_id = tv2.trial_id AND tv2.objective = 2
            LEFT JOIN trial_values tv3 ON t.trial_id = tv3.trial_id AND tv3.objective = 3
            WHERE t.state = 'COMPLETE'
            ORDER BY t.trial_id
            """
            
            trials_df = pd.read_sql_query(query, conn)
            
            # Get parameters
            params_query = """
            SELECT trial_id, param_name, param_value
            FROM trial_params
            """
            params_df = pd.read_sql_query(params_query, conn)
            
            conn.close()
            
            # Pivot parameters to columns
            if not params_df.empty:
                params_pivot = params_df.pivot(index='trial_id', columns='param_name', values='param_value')
                trials_df = trials_df.merge(params_pivot, left_on='trial_id', right_index=True, how='left')
            
            # Rename objectives for clarity
            trials_df.rename(columns={
                'objective_0': 'cut_loss',
                'objective_1': 'balance_loss',
                'objective_2': 'neg_quality',
                'objective_3': 'neg_stability'
            }, inplace=True)
            
            # Convert negative objectives back to positive
            if 'neg_quality' in trials_df.columns:
                trials_df['quality_score'] = -trials_df['neg_quality']
            if 'neg_stability' in trials_df.columns:
                trials_df['stability_score'] = -trials_df['neg_stability']
            
            return trials_df
            
        except Exception as e:
            print(f"Error loading trials data: {e}")
            return pd.DataFrame()
    
    def _load_results_data(self) -> Dict:
        """Load results from JSON files"""
        results = {}
        
        if self.results_json.exists():
            try:
                with open(self.results_json, 'r') as f:
                    results['optimization'] = json.load(f)
            except Exception as e:
                print(f"Error loading optimization results: {e}")
        
        if self.best_config_json.exists():
            try:
                with open(self.best_config_json, 'r') as f:
                    results['best_config'] = json.load(f)
            except Exception as e:
                print(f"Error loading best config: {e}")
        
        return results
    
    def create_convergence_plot(self, save_path: Optional[str] = None) -> go.Figure:
        """Create interactive convergence plot"""
        if self.trials_df.empty:
            print("No trial data available for plotting")
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cut Loss Convergence', 'Balance Loss Convergence',
                          'Quality Score Progress', 'Stability Score Progress'),
            vertical_spacing=0.12
        )
        
        # Calculate running minimum for losses and maximum for scores
        df = self.trials_df.sort_values('trial_id')
        
        if 'cut_loss' in df.columns:
            df['best_cut_loss'] = df['cut_loss'].cummin()
            fig.add_trace(
                go.Scatter(x=df['trial_id'], y=df['best_cut_loss'],
                          name='Best Cut Loss', line=dict(color='red', width=2)),
                row=1, col=1
            )
            # Add target line
            fig.add_hline(y=0.02, line_dash="dash", line_color="red", 
                         annotation_text="Target: 0.02", row=1, col=1)
        
        if 'balance_loss' in df.columns:
            df['best_balance_loss'] = df['balance_loss'].cummin()
            fig.add_trace(
                go.Scatter(x=df['trial_id'], y=df['best_balance_loss'],
                          name='Best Balance Loss', line=dict(color='blue', width=2)),
                row=1, col=2
            )
            # Add target line
            fig.add_hline(y=1e-3, line_dash="dash", line_color="blue", 
                         annotation_text="Target: 1e-3", row=1, col=2)
        
        if 'quality_score' in df.columns:
            df['best_quality'] = df['quality_score'].cummax()
            fig.add_trace(
                go.Scatter(x=df['trial_id'], y=df['best_quality'],
                          name='Best Quality', line=dict(color='green', width=2)),
                row=2, col=1
            )
        
        if 'stability_score' in df.columns:
            df['best_stability'] = df['stability_score'].cummax()
            fig.add_trace(
                go.Scatter(x=df['trial_id'], y=df['best_stability'],
                          name='Best Stability', line=dict(color='orange', width=2)),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Hyperparameter Optimization Convergence",
            height=800,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Trial Number")
        fig.update_yaxes(title_text="Cut Loss", row=1, col=1, type="log")
        fig.update_yaxes(title_text="Balance Loss", row=1, col=2, type="log")
        fig.update_yaxes(title_text="Quality Score", row=2, col=1)
        fig.update_yaxes(title_text="Stability Score", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_parameter_importance_plot(self, save_path: Optional[str] = None) -> go.Figure:
        """Create parameter importance visualization"""
        if self.trials_df.empty:
            print("No trial data available for parameter analysis")
            return None
        
        # Calculate parameter importance using correlation with objectives
        numeric_params = []
        param_importance = {}
        
        for col in self.trials_df.columns:
            if col not in ['trial_id', 'state', 'cut_loss', 'balance_loss', 
                          'quality_score', 'stability_score', 'datetime_start', 
                          'datetime_complete', 'neg_quality', 'neg_stability']:
                try:
                    self.trials_df[col] = pd.to_numeric(self.trials_df[col], errors='coerce')
                    if not self.trials_df[col].isna().all():
                        numeric_params.append(col)
                except:
                    continue
        
        # Calculate importance scores
        for param in numeric_params:
            importance_score = 0
            valid_data = self.trials_df[[param, 'cut_loss', 'balance_loss']].dropna()
            
            if len(valid_data) > 5:  # Need minimum data points
                # Correlation with cut loss (negative because we want to minimize)
                cut_corr = abs(valid_data[param].corr(valid_data['cut_loss']))
                balance_corr = abs(valid_data[param].corr(valid_data['balance_loss']))
                importance_score = (cut_corr * 0.6 + balance_corr * 0.4)  # Weight cut loss more
                
            param_importance[param] = importance_score
        
        # Sort by importance
        sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_params:
            print("No parameter importance data available")
            return None
        
        params, scores = zip(*sorted_params[:15])  # Top 15 parameters
        
        fig = go.Figure(data=[
            go.Bar(x=list(scores), y=list(params), orientation='h',
                  marker_color='skyblue', text=[f'{s:.3f}' for s in scores],
                  textposition='auto')
        ])
        
        fig.update_layout(
            title="Parameter Importance for Loss Minimization",
            xaxis_title="Importance Score (Correlation with Objectives)",
            yaxis_title="Parameters",
            height=600,
            margin=dict(l=150)
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_pareto_front_plot(self, save_path: Optional[str] = None) -> go.Figure:
        """Create Pareto front visualization for multi-objective optimization"""
        if self.trials_df.empty or 'cut_loss' not in self.trials_df.columns:
            print("No trial data available for Pareto front")
            return None
        
        df = self.trials_df[['cut_loss', 'balance_loss', 'quality_score', 'trial_id']].dropna()
        
        # Find Pareto front (minimize cut_loss and balance_loss, maximize quality)
        pareto_mask = np.ones(len(df), dtype=bool)
        
        for i, (_, row) in enumerate(df.iterrows()):
            for j, (_, other_row) in enumerate(df.iterrows()):
                if i != j:
                    # Check if point j dominates point i
                    if (other_row['cut_loss'] <= row['cut_loss'] and
                        other_row['balance_loss'] <= row['balance_loss'] and
                        other_row['quality_score'] >= row['quality_score'] and
                        (other_row['cut_loss'] < row['cut_loss'] or 
                         other_row['balance_loss'] < row['balance_loss'] or
                         other_row['quality_score'] > row['quality_score'])):
                        pareto_mask[i] = False
                        break
        
        pareto_points = df[pareto_mask]
        non_pareto_points = df[~pareto_mask]
        
        fig = go.Figure()
        
        # Non-Pareto points
        fig.add_trace(go.Scatter3d(
            x=non_pareto_points['cut_loss'],
            y=non_pareto_points['balance_loss'], 
            z=non_pareto_points['quality_score'],
            mode='markers',
            marker=dict(size=4, color='lightblue', opacity=0.6),
            name='Non-Pareto Points',
            text=[f'Trial {t}' for t in non_pareto_points['trial_id']],
            hovertemplate='Trial: %{text}<br>Cut Loss: %{x:.4f}<br>Balance Loss: %{y:.6f}<br>Quality: %{z:.3f}<extra></extra>'
        ))
        
        # Pareto front points
        fig.add_trace(go.Scatter3d(
            x=pareto_points['cut_loss'],
            y=pareto_points['balance_loss'],
            z=pareto_points['quality_score'],
            mode='markers',
            marker=dict(size=8, color='red', opacity=0.8),
            name='Pareto Front',
            text=[f'Trial {t}' for t in pareto_points['trial_id']],
            hovertemplate='Trial: %{text}<br>Cut Loss: %{x:.4f}<br>Balance Loss: %{y:.6f}<br>Quality: %{z:.3f}<extra></extra>'
        ))
        
        # Add target planes
        cut_target = 0.02
        balance_target = 1e-3
        
        fig.update_layout(
            title="Multi-Objective Optimization: Pareto Front",
            scene=dict(
                xaxis_title="Cut Loss (minimize)",
                yaxis_title="Balance Loss (minimize)", 
                zaxis_title="Quality Score (maximize)",
                xaxis=dict(type='log'),
                yaxis=dict(type='log')
            ),
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_correlation_heatmap(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create correlation heatmap of parameters and objectives"""
        if self.trials_df.empty:
            print("No trial data available for correlation analysis")
            return None
        
        # Select numeric columns
        numeric_cols = []
        for col in self.trials_df.columns:
            try:
                if col not in ['trial_id', 'state', 'datetime_start', 'datetime_complete']:
                    pd.to_numeric(self.trials_df[col], errors='coerce')
                    numeric_cols.append(col)
            except:
                continue
        
        if len(numeric_cols) < 2:
            print("Not enough numeric data for correlation analysis")
            return None
        
        # Compute correlation matrix
        corr_data = self.trials_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        correlation_matrix = corr_data.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        
        plt.title('Parameter and Objective Correlations')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def create_loss_evolution_animation(self, save_path: Optional[str] = None):
        """Create animated plot showing loss evolution over trials"""
        if self.trials_df.empty:
            return None
        
        df = self.trials_df.sort_values('trial_id')
        
        # Create frames for animation
        frames = []
        for i in range(10, len(df), 5):  # Every 5 trials after trial 10
            frame_data = df.iloc[:i]
            
            frame = go.Frame(
                data=[
                    go.Scatter(x=frame_data['trial_id'], y=frame_data['cut_loss'],
                             mode='markers+lines', name='Cut Loss', yaxis='y',
                             marker=dict(color='red')),
                    go.Scatter(x=frame_data['trial_id'], y=frame_data['balance_loss'],
                             mode='markers+lines', name='Balance Loss', yaxis='y2',
                             marker=dict(color='blue'))
                ],
                name=f'Trial {i}'
            )
            frames.append(frame)
        
        # Create initial plot
        fig = go.Figure(
            data=[
                go.Scatter(x=df['trial_id'][:10], y=df['cut_loss'][:10],
                          mode='markers+lines', name='Cut Loss', yaxis='y',
                          marker=dict(color='red')),
                go.Scatter(x=df['trial_id'][:10], y=df['balance_loss'][:10],
                          mode='markers+lines', name='Balance Loss', yaxis='y2', 
                          marker=dict(color='blue'))
            ],
            frames=frames
        )
        
        # Add play button
        fig.update_layout(
            title="Loss Evolution Over Optimization Trials",
            xaxis_title="Trial Number",
            yaxis=dict(title="Cut Loss", side="left", color="red", type="log"),
            yaxis2=dict(title="Balance Loss", side="right", overlaying="y", 
                       color="blue", type="log"),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': 'Play', 'method': 'animate', 
                     'args': [None, {'frame': {'duration': 200, 'redraw': True}}]},
                    {'label': 'Pause', 'method': 'animate', 
                     'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 
                                     'mode': 'immediate'}]}
                ]
            }]
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def generate_comprehensive_report(self, output_dir: Optional[str] = None):
        """Generate comprehensive analysis report"""
        if output_dir is None:
            output_dir = self.results_dir / "analysis"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        print("📊 Generating comprehensive analysis report...")
        
        # Create all visualizations
        print("  🎯 Creating convergence plot...")
        conv_fig = self.create_convergence_plot(str(output_dir / "convergence.html"))
        
        print("  📈 Creating parameter importance plot...")
        imp_fig = self.create_parameter_importance_plot(str(output_dir / "parameter_importance.html"))
        
        print("  🎲 Creating Pareto front plot...")
        pareto_fig = self.create_pareto_front_plot(str(output_dir / "pareto_front.html"))
        
        print("  🔥 Creating correlation heatmap...")
        corr_fig = self.create_correlation_heatmap(str(output_dir / "correlations.png"))
        
        print("  🎬 Creating evolution animation...")
        anim_fig = self.create_loss_evolution_animation(str(output_dir / "evolution_animation.html"))
        
        # Generate summary statistics
        summary = self._generate_summary_stats()
        
        # Create HTML report
        html_report = self._create_html_report(summary, output_dir)
        
        with open(output_dir / "analysis_report.html", "w") as f:
            f.write(html_report)
        
        print(f"✅ Analysis report generated: {output_dir}")
        print(f"📖 View report: {output_dir}/analysis_report.html")
        
        return output_dir
    
    def _generate_summary_stats(self) -> Dict:
        """Generate summary statistics"""
        if self.trials_df.empty:
            return {}
        
        summary = {
            'total_trials': len(self.trials_df),
            'targets_achieved': {
                'cut_loss': 0,
                'balance_loss': 0,
                'both': 0
            },
            'best_trials': {},
            'parameter_stats': {}
        }
        
        # Count target achievements
        if 'cut_loss' in self.trials_df.columns:
            cut_achieved = (self.trials_df['cut_loss'] <= 0.02).sum()
            summary['targets_achieved']['cut_loss'] = cut_achieved
            summary['best_trials']['cut_loss'] = {
                'value': self.trials_df['cut_loss'].min(),
                'trial_id': self.trials_df.loc[self.trials_df['cut_loss'].idxmin(), 'trial_id']
            }
        
        if 'balance_loss' in self.trials_df.columns:
            balance_achieved = (self.trials_df['balance_loss'] <= 1e-3).sum()
            summary['targets_achieved']['balance_loss'] = balance_achieved
            summary['best_trials']['balance_loss'] = {
                'value': self.trials_df['balance_loss'].min(),
                'trial_id': self.trials_df.loc[self.trials_df['balance_loss'].idxmin(), 'trial_id']
            }
        
        # Both targets
        if 'cut_loss' in self.trials_df.columns and 'balance_loss' in self.trials_df.columns:
            both_achieved = ((self.trials_df['cut_loss'] <= 0.02) & 
                           (self.trials_df['balance_loss'] <= 1e-3)).sum()
            summary['targets_achieved']['both'] = both_achieved
        
        return summary
    
    def _create_html_report(self, summary: Dict, output_dir: Path) -> str:
        """Create comprehensive HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GraphPart Hyperparameter Optimization Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1, h2 {{ color: #333; }}
                .metric {{ display: inline-block; margin: 10px; padding: 20px; background: #f8f9fa; border-radius: 8px; text-align: center; min-width: 150px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
                .success {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .danger {{ color: #dc3545; }}
                .visualization {{ margin: 20px 0; text-align: center; }}
                .iframe-container {{ width: 100%; height: 600px; border: 1px solid #ddd; border-radius: 8px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; font-weight: bold; }}
                .highlight {{ background-color: #fff3cd; }}
                .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
                .badge-success {{ background: #d4edda; color: #155724; }}
                .badge-danger {{ background: #f8d7da; color: #721c24; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 GraphPart Hyperparameter Optimization Analysis</h1>
                <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>📊 Summary Statistics</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">{summary.get('total_trials', 0)}</div>
                        <div class="metric-label">Total Trials</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value {'success' if summary.get('targets_achieved', {}).get('both', 0) > 0 else 'danger'}">{summary.get('targets_achieved', {}).get('both', 0)}</div>
                        <div class="metric-label">Both Targets Achieved</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value {'success' if summary.get('targets_achieved', {}).get('cut_loss', 0) > 0 else 'warning'}">{summary.get('targets_achieved', {}).get('cut_loss', 0)}</div>
                        <div class="metric-label">Cut Target (≤0.02)</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value {'success' if summary.get('targets_achieved', {}).get('balance_loss', 0) > 0 else 'warning'}">{summary.get('targets_achieved', {}).get('balance_loss', 0)}</div>
                        <div class="metric-label">Balance Target (≤1e-3)</div>
                    </div>
                </div>
                
                <h2>🎯 Best Results</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Best Value</th>
                        <th>Trial ID</th>
                        <th>Target Achievement</th>
                    </tr>
        """
        
        # Add best results table
        if 'best_trials' in summary:
            for metric, data in summary['best_trials'].items():
                if metric == 'cut_loss':
                    target = 0.02
                    achieved = data['value'] <= target
                    badge_class = 'badge-success' if achieved else 'badge-danger'
                    badge_text = '✅ Achieved' if achieved else '❌ Missed'
                elif metric == 'balance_loss':
                    target = 1e-3  
                    achieved = data['value'] <= target
                    badge_class = 'badge-success' if achieved else 'badge-danger'
                    badge_text = '✅ Achieved' if achieved else '❌ Missed'
                else:
                    badge_class = 'badge-success'
                    badge_text = '✅'
                
                html += f"""
                    <tr>
                        <td>{metric.replace('_', ' ').title()}</td>
                        <td>{data['value']:.6f}</td>
                        <td>{data['trial_id']}</td>
                        <td><span class="badge {badge_class}">{badge_text}</span></td>
                    </tr>
                """
        
        html += """
                </table>
                
                <h2>📈 Visualizations</h2>
                
                <div class="visualization">
                    <h3>Convergence Analysis</h3>
                    <iframe src="convergence.html" class="iframe-container"></iframe>
                </div>
                
                <div class="visualization">
                    <h3>Parameter Importance</h3>
                    <iframe src="parameter_importance.html" class="iframe-container"></iframe>
                </div>
                
                <div class="visualization">
                    <h3>Pareto Front (Multi-Objective)</h3>
                    <iframe src="pareto_front.html" class="iframe-container"></iframe>
                </div>
                
                <div class="visualization">
                    <h3>Parameter Correlations</h3>
                    <img src="correlations.png" style="max-width: 100%; height: auto;" alt="Correlation Heatmap">
                </div>
                
                <div class="visualization">
                    <h3>Loss Evolution Animation</h3>
                    <iframe src="evolution_animation.html" class="iframe-container"></iframe>
                </div>
                
                <h2>🎉 Conclusions & Recommendations</h2>
        """
        
        # Add conclusions based on results
        if summary.get('targets_achieved', {}).get('both', 0) > 0:
            html += """
                <div style="background: #d4edda; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <h4 style="color: #155724; margin-top: 0;">🎯 Success! Targets Achieved</h4>
                    <p>Excellent! The optimization successfully found configurations that achieve both target losses:</p>
                    <ul>
                        <li>Cut loss ≤ 0.02</li>
                        <li>Balance loss ≤ 1e-3</li>
                    </ul>
                    <p><strong>Next steps:</strong></p>
                    <ul>
                        <li>Deploy the best configuration for production use</li>
                        <li>Validate results on additional test datasets</li>
                        <li>Consider ensemble methods for even better performance</li>
                    </ul>
                </div>
            """
        else:
            html += """
                <div style="background: #fff3cd; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <h4 style="color: #856404; margin-top: 0;">⚠️ Targets Not Yet Achieved</h4>
                    <p>The optimization made progress but hasn't fully achieved both targets. Consider:</p>
                    <ul>
                        <li>Running additional trials with refined search space</li>
                        <li>Adjusting loss function weights (higher β and γ)</li>
                        <li>Experimenting with different normalization methods</li>
                        <li>Using longer training epochs for promising configurations</li>
                    </ul>
                </div>
            """
        
        html += """
                <h2>📁 Files Generated</h2>
                <ul>
                    <li><code>convergence.html</code> - Interactive convergence plots</li>
                    <li><code>parameter_importance.html</code> - Parameter importance analysis</li>
                    <li><code>pareto_front.html</code> - Multi-objective Pareto front</li>
                    <li><code>correlations.png</code> - Parameter correlation heatmap</li>
                    <li><code>evolution_animation.html</code> - Animated loss evolution</li>
                    <li><code>analysis_report.html</code> - This comprehensive report</li>
                </ul>
                
                <hr style="margin: 40px 0;">
                <p style="text-align: center; color: #666; font-size: 14px;">
                    Generated by GraphPart Hyperparameter Optimization System
                </p>
            </div>
        </body>
        </html>
        """
        
        return html

def main():
    """Main analysis entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze hyperparameter optimization results")
    parser.add_argument('--results-dir', type=str, required=True, help='Directory with optimization results')
    parser.add_argument('--output-dir', type=str, help='Output directory for analysis (default: results_dir/analysis)')
    
    args = parser.parse_args()
    
    analyzer = HyperoptAnalyzer(args.results_dir)
    output_dir = analyzer.generate_comprehensive_report(args.output_dir)
    
    print(f"\n🎉 Analysis complete! Open {output_dir}/analysis_report.html to view results.")

if __name__ == '__main__':
    main()