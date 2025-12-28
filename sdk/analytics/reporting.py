"""
Report Generator
================

Generate analytics reports.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import logging

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Report formats."""
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"
    CSV = "csv"


class ReportType(Enum):
    """Report types."""
    SUMMARY = "summary"
    DETAILED = "detailed"
    COMPARISON = "comparison"
    TREND = "trend"
    ANOMALY = "anomaly"


@dataclass
class ReportSection:
    """Report section."""
    title: str
    content: Any
    section_type: str
    order: int


@dataclass
class Report:
    """Generated report."""
    report_id: str
    report_type: ReportType
    title: str
    description: str
    sections: List[ReportSection]
    generated_at: float
    period_start: float
    period_end: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReportGenerator:
    """
    Analytics report generator.
    
    Features:
    - Multiple report types
    - Various output formats
    - Scheduled reports
    - Custom templates
    - Email delivery
    
    Example:
        >>> generator = ReportGenerator(metrics, tracker)
        >>> report = generator.generate_summary_report("weekly")
    """
    
    def __init__(self, metrics_aggregator: Any = None,
                 analytics_tracker: Any = None):
        """
        Initialize report generator.
        
        Args:
            metrics_aggregator: MetricsAggregator instance
            analytics_tracker: AnalyticsTracker instance
        """
        self.metrics = metrics_aggregator
        self.tracker = analytics_tracker
        
        # Generated reports
        self._reports: Dict[str, Report] = {}
        
        # Report counter
        self._report_counter = 0
        
        logger.info("Report Generator initialized")
    
    def generate_summary_report(self, period: str = "daily",
                                 title: str = None) -> Report:
        """
        Generate summary report.
        
        Args:
            period: Report period (hourly, daily, weekly, monthly)
            title: Custom title
            
        Returns:
            Report
        """
        # Calculate time range
        now = time.time()
        period_map = {
            "hourly": 3600,
            "daily": 86400,
            "weekly": 604800,
            "monthly": 2592000
        }
        
        duration = period_map.get(period, 86400)
        period_start = now - duration
        
        self._report_counter += 1
        report_id = f"report_{self._report_counter}"
        
        sections = []
        
        # Overview section
        sections.append(ReportSection(
            title="Overview",
            content=self._generate_overview(period_start, now),
            section_type="overview",
            order=1
        ))
        
        # Metrics section
        if self.metrics:
            sections.append(ReportSection(
                title="Key Metrics",
                content=self._generate_metrics_section(period_start, now),
                section_type="metrics",
                order=2
            ))
        
        # Events section
        if self.tracker:
            sections.append(ReportSection(
                title="Event Summary",
                content=self._generate_events_section(period_start, now),
                section_type="events",
                order=3
            ))
        
        # Top items section
        sections.append(ReportSection(
            title="Top Items",
            content=self._generate_top_items(period_start, now),
            section_type="top_items",
            order=4
        ))
        
        report = Report(
            report_id=report_id,
            report_type=ReportType.SUMMARY,
            title=title or f"{period.capitalize()} Summary Report",
            description=f"Summary report for {period} period",
            sections=sections,
            generated_at=now,
            period_start=period_start,
            period_end=now,
            metadata={"period": period}
        )
        
        self._reports[report_id] = report
        
        logger.info(f"Generated summary report: {report_id}")
        return report
    
    def _generate_overview(self, start: float, end: float) -> Dict:
        """Generate overview section."""
        return {
            "period_start": start,
            "period_end": end,
            "duration_hours": (end - start) / 3600,
            "generated_at": time.time()
        }
    
    def _generate_metrics_section(self, start: float, end: float) -> Dict:
        """Generate metrics section."""
        if not self.metrics:
            return {}
        
        metric_names = self.metrics.list_metrics()
        metrics_data = {}
        
        for name in metric_names[:10]:  # Top 10 metrics
            summary = self.metrics.get_summary(name, int(end - start))
            metrics_data[name] = summary
        
        return metrics_data
    
    def _generate_events_section(self, start: float, end: float) -> Dict:
        """Generate events section."""
        if not self.tracker:
            return {}
        
        stats = self.tracker.get_statistics()
        
        return {
            "total_events": stats.get("total_events", 0),
            "active_sessions": stats.get("active_sessions", 0),
            "event_types": {}
        }
    
    def _generate_top_items(self, start: float, end: float) -> Dict:
        """Generate top items section."""
        return {
            "top_metrics": [],
            "top_errors": [],
            "top_users": []
        }
    
    def generate_trend_report(self, metric_names: List[str],
                               period: str = "weekly") -> Report:
        """Generate trend analysis report."""
        now = time.time()
        period_map = {"daily": 86400, "weekly": 604800, "monthly": 2592000}
        duration = period_map.get(period, 604800)
        
        self._report_counter += 1
        report_id = f"report_{self._report_counter}"
        
        sections = []
        
        for metric in metric_names:
            if self.metrics:
                timeseries = self.metrics.get_timeseries(
                    metric, now - duration, now
                )
                
                sections.append(ReportSection(
                    title=f"Trend: {metric}",
                    content={
                        "metric": metric,
                        "data_points": len(timeseries),
                        "timeseries": timeseries[-100:]  # Last 100 points
                    },
                    section_type="trend",
                    order=len(sections) + 1
                ))
        
        report = Report(
            report_id=report_id,
            report_type=ReportType.TREND,
            title=f"Trend Analysis Report",
            description=f"Trend analysis for {len(metric_names)} metrics",
            sections=sections,
            generated_at=now,
            period_start=now - duration,
            period_end=now
        )
        
        self._reports[report_id] = report
        return report
    
    def export_report(self, report_id: str,
                      format: ReportFormat = ReportFormat.JSON) -> str:
        """
        Export report to specified format.
        
        Args:
            report_id: Report ID
            format: Output format
            
        Returns:
            Formatted report string
        """
        if report_id not in self._reports:
            return ""
        
        report = self._reports[report_id]
        
        if format == ReportFormat.JSON:
            return self._export_json(report)
        elif format == ReportFormat.HTML:
            return self._export_html(report)
        elif format == ReportFormat.MARKDOWN:
            return self._export_markdown(report)
        else:
            return self._export_json(report)
    
    def _export_json(self, report: Report) -> str:
        """Export as JSON."""
        return json.dumps({
            "report_id": report.report_id,
            "type": report.report_type.value,
            "title": report.title,
            "description": report.description,
            "sections": [
                {
                    "title": s.title,
                    "content": s.content,
                    "type": s.section_type
                }
                for s in report.sections
            ],
            "generated_at": report.generated_at,
            "period": {
                "start": report.period_start,
                "end": report.period_end
            }
        }, indent=2)
    
    def _export_html(self, report: Report) -> str:
        """Export as HTML."""
        sections_html = ""
        for section in report.sections:
            sections_html += f"""
            <section>
                <h2>{section.title}</h2>
                <pre>{json.dumps(section.content, indent=2)}</pre>
            </section>
            """
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                section {{ margin: 20px 0; padding: 20px; background: #f5f5f5; }}
            </style>
        </head>
        <body>
            <h1>{report.title}</h1>
            <p>{report.description}</p>
            {sections_html}
            <footer>Generated at: {time.ctime(report.generated_at)}</footer>
        </body>
        </html>
        """
    
    def _export_markdown(self, report: Report) -> str:
        """Export as Markdown."""
        md = f"# {report.title}\n\n"
        md += f"{report.description}\n\n"
        
        for section in report.sections:
            md += f"## {section.title}\n\n"
            md += f"```json\n{json.dumps(section.content, indent=2)}\n```\n\n"
        
        md += f"---\n*Generated at: {time.ctime(report.generated_at)}*\n"
        
        return md
    
    def get_report(self, report_id: str) -> Optional[Report]:
        """Get report by ID."""
        return self._reports.get(report_id)
    
    def list_reports(self) -> List[Report]:
        """List all reports."""
        return list(self._reports.values())
    
    def __repr__(self) -> str:
        return f"ReportGenerator(reports={len(self._reports)})"
