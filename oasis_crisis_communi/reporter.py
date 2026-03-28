"""Report generation (PDF and HTML)."""
from __future__ import annotations

import os
import json
import datetime
from typing import List

from .simulation import SimulationResult
from .analyzer import ResultsAnalyzer


class ReportGenerator:
    """Generates HTML and JSON reports from simulation results."""

    def __init__(self, results: List[SimulationResult], output_dir: str = "outputs"):
        self.results = results
        self.analyzer = ResultsAnalyzer(results)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_json_report(self) -> str:
        """Save detailed JSON results."""
        save_path = os.path.join(self.output_dir, "results.json")
        winner = self.analyzer.get_winner()

        report = {
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "simulation_parameters": {
                "num_agents": self.results[0].num_agents if self.results else 0,
                "num_timesteps": self.results[0].timesteps if self.results else 0,
                "num_strategies": len(self.results),
            },
            "winning_strategy": winner.strategy.name if winner else None,
            "strategies": [r.to_summary_dict() for r in self.results],
            "effect_sizes": self.analyzer.compute_effect_sizes(),
            "time_to_60pct_threshold": self.analyzer.compute_time_to_threshold(0.6),
            "recommendation": self.analyzer.generate_recommendation(),
            "sample_interview_responses": self.analyzer.get_interview_sample(3),
        }

        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)

        return save_path

    def save_html_report(self) -> str:
        """Save a styled HTML report."""
        save_path = os.path.join(self.output_dir, "report.html")
        winner = self.analyzer.get_winner()
        summary_df = self.analyzer.get_summary_table()
        recommendation = self.analyzer.generate_recommendation()
        interview_samples = self.analyzer.get_interview_sample(5)
        effect_sizes = self.analyzer.compute_effect_sizes()

        # Build the HTML
        table_rows = ""
        for _, row in summary_df.iterrows():
            winning_badge = " \U0001f3c6" if row.get("winning") else ""
            table_rows += f"""
            <tr>
                <td>{row['rank']}</td>
                <td><strong>{row['strategy']}{winning_badge}</strong></td>
                <td>{float(row['final_alignment']):.1%}</td>
                <td>{float(row['peak_alignment']):.1%}</td>
                <td>{float(row['avg_alignment']):.1%}</td>
                <td>{int(row['total_gov_reposts'])}</td>
            </tr>"""

        interview_html = ""
        for resp in interview_samples:
            score = resp.get("belief_score", 0)
            sentiment = resp.get("sentiment", "neutral")
            color = "#4CAF50" if sentiment == "positive" else "#F44336" if sentiment == "negative" else "#FF9800"
            interview_html += f"""
            <div class="interview-card">
                <div class="interview-header">
                    Agent {resp['agent_id']} \u00b7 Timestep {resp['timestep']} \u00b7
                    <span style="color:{color}">\u25a0 {sentiment}</span> \u00b7
                    Belief: <strong>{score:.2f}</strong>
                </div>
                <div class="interview-response">{resp['raw_response']}</div>
            </div>"""

        effect_html = ""
        for pair, d in effect_sizes.items():
            magnitude = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
            effect_html += f"<li><strong>{pair}</strong>: d={d:.3f} ({magnitude})</li>"

        rec_html = recommendation.replace("\n", "<br>")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OASIS Crisis Communication Optimizer \u2014 Results Report</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; color: #333; }}
        .container {{ max-width: 960px; margin: 0 auto; padding: 24px; }}
        h1 {{ color: #1a237e; font-size: 2em; margin-bottom: 8px; }}
        h2 {{ color: #283593; font-size: 1.3em; margin: 24px 0 12px; border-bottom: 2px solid #e3e3e3; padding-bottom: 6px; }}
        .subtitle {{ color: #666; margin-bottom: 24px; }}
        .winner-box {{ background: linear-gradient(135deg, #1a237e, #283593); color: white; border-radius: 12px; padding: 20px; margin-bottom: 24px; }}
        .winner-box h2 {{ color: white; border-bottom-color: rgba(255,255,255,0.3); }}
        .winner-score {{ font-size: 3em; font-weight: bold; margin: 12px 0; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        th {{ background: #1a237e; color: white; padding: 12px; text-align: left; font-size: 0.9em; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #eee; }}
        tr:last-child td {{ border-bottom: none; }}
        tr:hover {{ background: #f8f9ff; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .recommendation {{ background: #E8F5E9; border-left: 4px solid #4CAF50; padding: 16px; border-radius: 4px; font-family: monospace; font-size: 0.9em; white-space: pre-wrap; }}
        .interview-card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 14px; margin-bottom: 12px; }}
        .interview-header {{ font-size: 0.85em; color: #666; margin-bottom: 8px; }}
        .interview-response {{ font-size: 0.9em; color: #333; line-height: 1.5; }}
        .footer {{ text-align: center; color: #999; font-size: 0.8em; margin-top: 40px; padding: 20px; }}
        img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
<div class="container">
    <h1>OASIS Crisis Communication Optimizer</h1>
    <div class="subtitle">
        Public Health Emergency Communication Strategy Analysis<br>
        Generated: {datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}
    </div>

    {"" if not winner else f'''
    <div class="winner-box">
        <h2>\U0001f3c6 Winning Strategy</h2>
        <div class="winner-score">{winner.strategy.name}</div>
        <div>Final Alignment: <strong>{winner.final_alignment:.1%}</strong> of population aligned with official guidance</div>
        <div style="margin-top:8px;opacity:0.9">{winner.strategy.description}</div>
    </div>
    '''}

    <h2>Strategy Comparison</h2>
    <div class="card">
        <table>
            <thead><tr>
                <th>Rank</th><th>Strategy</th><th>Final Alignment</th>
                <th>Peak</th><th>Average</th><th>Gov Reposts</th>
            </tr></thead>
            <tbody>{table_rows}</tbody>
        </table>
    </div>

    <h2>Visualization</h2>
    <div class="card">
        <img src="strategy_comparison.png" alt="Strategy Comparison Chart">
    </div>

    <h2>Effect Sizes (Cohen's d)</h2>
    <div class="card">
        <ul style="padding-left:20px;line-height:2">{effect_html}</ul>
    </div>

    <h2>Sample Interview Responses (Belief Measurement)</h2>
    <div class="card">
        <p style="color:#666;margin-bottom:12px;font-size:0.9em">
            INTERVIEW actions probe each agent's current belief. Below are sample responses from the winning strategy.
        </p>
        {interview_html}
    </div>

    <h2>Recommendation</h2>
    <div class="card">
        <div class="recommendation">{rec_html}</div>
    </div>

    <div class="footer">
        OASIS Crisis Communication Optimizer \u00b7 Built with CAMEL-AI OASIS Framework<br>
        Made autonomously using <a href="https://heyneo.so">NEO</a>
    </div>
</div>
</body>
</html>"""

        with open(save_path, "w") as f:
            f.write(html)

        return save_path
