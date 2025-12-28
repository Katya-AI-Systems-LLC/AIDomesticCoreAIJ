"""
Dashboard API
=============

Dashboard and visualization API.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import logging

logger = logging.getLogger(__name__)


class WidgetType(Enum):
    """Dashboard widget types."""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    GAUGE = "gauge"
    TABLE = "table"
    STAT = "stat"
    HEATMAP = "heatmap"
    MAP = "map"


@dataclass
class Widget:
    """Dashboard widget."""
    widget_id: str
    widget_type: WidgetType
    title: str
    metric: str
    config: Dict[str, Any]
    position: Dict[str, int]  # x, y, width, height
    refresh_interval: int = 30


@dataclass
class Dashboard:
    """Dashboard configuration."""
    dashboard_id: str
    name: str
    description: str
    widgets: List[Widget]
    created_at: float
    updated_at: float
    owner: str
    shared: bool = False
    tags: List[str] = field(default_factory=list)


class DashboardAPI:
    """
    Dashboard management API.
    
    Features:
    - Dashboard CRUD
    - Widget management
    - Real-time updates
    - Sharing and collaboration
    - Export/Import
    
    Example:
        >>> api = DashboardAPI()
        >>> dashboard = api.create_dashboard("AI Metrics")
        >>> api.add_widget(dashboard.dashboard_id, widget)
    """
    
    def __init__(self, metrics_aggregator: Any = None):
        """
        Initialize Dashboard API.
        
        Args:
            metrics_aggregator: MetricsAggregator instance
        """
        self.metrics = metrics_aggregator
        
        # Dashboards storage
        self._dashboards: Dict[str, Dashboard] = {}
        
        # Widget counter
        self._widget_counter = 0
        
        logger.info("Dashboard API initialized")
    
    def create_dashboard(self, name: str,
                         description: str = "",
                         owner: str = "system") -> Dashboard:
        """
        Create new dashboard.
        
        Args:
            name: Dashboard name
            description: Dashboard description
            owner: Dashboard owner
            
        Returns:
            Dashboard
        """
        import hashlib
        
        dashboard_id = hashlib.sha256(
            f"{name}_{owner}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            name=name,
            description=description,
            widgets=[],
            created_at=time.time(),
            updated_at=time.time(),
            owner=owner
        )
        
        self._dashboards[dashboard_id] = dashboard
        
        logger.info(f"Dashboard created: {name} ({dashboard_id})")
        return dashboard
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard by ID."""
        return self._dashboards.get(dashboard_id)
    
    def list_dashboards(self, owner: str = None) -> List[Dashboard]:
        """List all dashboards."""
        dashboards = list(self._dashboards.values())
        
        if owner:
            dashboards = [d for d in dashboards if d.owner == owner]
        
        return dashboards
    
    def update_dashboard(self, dashboard_id: str,
                         name: str = None,
                         description: str = None) -> Optional[Dashboard]:
        """Update dashboard."""
        if dashboard_id not in self._dashboards:
            return None
        
        dashboard = self._dashboards[dashboard_id]
        
        if name:
            dashboard.name = name
        if description:
            dashboard.description = description
        
        dashboard.updated_at = time.time()
        
        return dashboard
    
    def delete_dashboard(self, dashboard_id: str) -> bool:
        """Delete dashboard."""
        if dashboard_id in self._dashboards:
            del self._dashboards[dashboard_id]
            return True
        return False
    
    def add_widget(self, dashboard_id: str,
                   widget_type: WidgetType,
                   title: str,
                   metric: str,
                   config: Dict = None,
                   position: Dict = None) -> Optional[Widget]:
        """
        Add widget to dashboard.
        
        Args:
            dashboard_id: Dashboard ID
            widget_type: Widget type
            title: Widget title
            metric: Metric to display
            config: Widget configuration
            position: Widget position
            
        Returns:
            Widget
        """
        if dashboard_id not in self._dashboards:
            return None
        
        self._widget_counter += 1
        widget_id = f"widget_{self._widget_counter}"
        
        widget = Widget(
            widget_id=widget_id,
            widget_type=widget_type,
            title=title,
            metric=metric,
            config=config or {},
            position=position or {"x": 0, "y": 0, "width": 4, "height": 3}
        )
        
        self._dashboards[dashboard_id].widgets.append(widget)
        self._dashboards[dashboard_id].updated_at = time.time()
        
        return widget
    
    def remove_widget(self, dashboard_id: str,
                      widget_id: str) -> bool:
        """Remove widget from dashboard."""
        if dashboard_id not in self._dashboards:
            return False
        
        dashboard = self._dashboards[dashboard_id]
        dashboard.widgets = [w for w in dashboard.widgets if w.widget_id != widget_id]
        dashboard.updated_at = time.time()
        
        return True
    
    def get_widget_data(self, dashboard_id: str,
                        widget_id: str) -> Optional[Dict]:
        """
        Get data for widget.
        
        Args:
            dashboard_id: Dashboard ID
            widget_id: Widget ID
            
        Returns:
            Widget data
        """
        if dashboard_id not in self._dashboards:
            return None
        
        dashboard = self._dashboards[dashboard_id]
        widget = next((w for w in dashboard.widgets if w.widget_id == widget_id), None)
        
        if not widget:
            return None
        
        # Get data based on widget type
        if self.metrics:
            if widget.widget_type == WidgetType.LINE_CHART:
                data = self.metrics.get_timeseries(widget.metric)
                return {"series": data}
            
            elif widget.widget_type == WidgetType.STAT:
                summary = self.metrics.get_summary(widget.metric)
                return summary
            
            elif widget.widget_type == WidgetType.GAUGE:
                agg = self.metrics.aggregate(widget.metric, "avg")
                return {"value": agg.value if agg else 0}
        
        # Simulated data
        return {
            "widget_id": widget_id,
            "metric": widget.metric,
            "data": [],
            "timestamp": time.time()
        }
    
    def export_dashboard(self, dashboard_id: str) -> Optional[str]:
        """Export dashboard as JSON."""
        if dashboard_id not in self._dashboards:
            return None
        
        dashboard = self._dashboards[dashboard_id]
        
        export_data = {
            "version": "1.0",
            "dashboard": {
                "name": dashboard.name,
                "description": dashboard.description,
                "widgets": [
                    {
                        "type": w.widget_type.value,
                        "title": w.title,
                        "metric": w.metric,
                        "config": w.config,
                        "position": w.position
                    }
                    for w in dashboard.widgets
                ],
                "tags": dashboard.tags
            },
            "exported_at": time.time()
        }
        
        return json.dumps(export_data, indent=2)
    
    def import_dashboard(self, json_data: str,
                         owner: str = "system") -> Optional[Dashboard]:
        """Import dashboard from JSON."""
        try:
            data = json.loads(json_data)
            dashboard_data = data["dashboard"]
            
            dashboard = self.create_dashboard(
                dashboard_data["name"],
                dashboard_data.get("description", ""),
                owner
            )
            
            for widget_data in dashboard_data.get("widgets", []):
                self.add_widget(
                    dashboard.dashboard_id,
                    WidgetType(widget_data["type"]),
                    widget_data["title"],
                    widget_data["metric"],
                    widget_data.get("config"),
                    widget_data.get("position")
                )
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return None
    
    def clone_dashboard(self, dashboard_id: str,
                        new_name: str) -> Optional[Dashboard]:
        """Clone existing dashboard."""
        if dashboard_id not in self._dashboards:
            return None
        
        exported = self.export_dashboard(dashboard_id)
        
        if exported:
            data = json.loads(exported)
            data["dashboard"]["name"] = new_name
            return self.import_dashboard(json.dumps(data))
        
        return None
    
    def share_dashboard(self, dashboard_id: str,
                        shared: bool = True) -> bool:
        """Set dashboard sharing status."""
        if dashboard_id in self._dashboards:
            self._dashboards[dashboard_id].shared = shared
            return True
        return False
    
    def __repr__(self) -> str:
        return f"DashboardAPI(dashboards={len(self._dashboards)})"
