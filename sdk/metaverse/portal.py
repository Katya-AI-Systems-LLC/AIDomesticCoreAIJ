"""
Cross-Metaverse Portal
======================

Interoperability between metaverses.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time
import logging

logger = logging.getLogger(__name__)


class MetaversePlatform(Enum):
    """Supported metaverse platforms."""
    AIPLATFORM = "aiplatform"
    DECENTRALAND = "decentraland"
    SANDBOX = "sandbox"
    CRYPTOVOXELS = "cryptovoxels"
    SOMNIUM = "somnium"
    SPATIAL = "spatial"
    HORIZONS = "horizons"


@dataclass
class PortalLink:
    """Link between metaverses."""
    portal_id: str
    source_metaverse: MetaversePlatform
    target_metaverse: MetaversePlatform
    source_location: Dict[str, Any]
    target_location: Dict[str, Any]
    bidirectional: bool
    active: bool
    owner: str
    created_at: float


@dataclass
class TravelSession:
    """Cross-metaverse travel session."""
    session_id: str
    traveler: str
    avatar_id: str
    source_metaverse: MetaversePlatform
    target_metaverse: MetaversePlatform
    portal_id: str
    status: str
    started_at: float
    completed_at: Optional[float] = None
    carried_assets: List[str] = field(default_factory=list)


class CrossMetaversePortal:
    """
    Cross-metaverse travel system.
    
    Features:
    - Portal creation and management
    - Avatar portability
    - NFT asset transfer
    - Identity preservation
    - State synchronization
    
    Example:
        >>> portal = CrossMetaversePortal()
        >>> link = portal.create_portal(
        ...     MetaversePlatform.AIPLATFORM,
        ...     MetaversePlatform.DECENTRALAND,
        ...     source_loc, target_loc
        ... )
        >>> session = await portal.travel(traveler, link.portal_id)
    """
    
    # Protocol versions for each platform
    PROTOCOL_VERSIONS = {
        MetaversePlatform.AIPLATFORM: "1.0",
        MetaversePlatform.DECENTRALAND: "0.9",
        MetaversePlatform.SANDBOX: "0.8",
        MetaversePlatform.CRYPTOVOXELS: "0.7",
        MetaversePlatform.SOMNIUM: "0.6",
        MetaversePlatform.SPATIAL: "0.5",
        MetaversePlatform.HORIZONS: "0.4"
    }
    
    def __init__(self):
        """Initialize portal system."""
        self._portals: Dict[str, PortalLink] = {}
        self._sessions: Dict[str, TravelSession] = {}
        
        # Platform connectors (simulated)
        self._connectors: Dict[MetaversePlatform, bool] = {
            p: True for p in MetaversePlatform
        }
        
        logger.info("Cross-Metaverse Portal initialized")
    
    def create_portal(self, source: MetaversePlatform,
                      target: MetaversePlatform,
                      source_location: Dict[str, Any],
                      target_location: Dict[str, Any],
                      owner: str,
                      bidirectional: bool = True) -> PortalLink:
        """
        Create portal between metaverses.
        
        Args:
            source: Source platform
            target: Target platform
            source_location: Location in source
            target_location: Location in target
            owner: Portal owner
            bidirectional: Allow two-way travel
            
        Returns:
            PortalLink
        """
        portal_id = hashlib.sha256(
            f"{source.value}_{target.value}_{owner}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        portal = PortalLink(
            portal_id=portal_id,
            source_metaverse=source,
            target_metaverse=target,
            source_location=source_location,
            target_location=target_location,
            bidirectional=bidirectional,
            active=True,
            owner=owner,
            created_at=time.time()
        )
        
        self._portals[portal_id] = portal
        
        logger.info(f"Portal created: {source.value} <-> {target.value}")
        return portal
    
    async def travel(self, traveler: str,
                     portal_id: str,
                     avatar_id: str,
                     carry_assets: List[str] = None) -> TravelSession:
        """
        Travel through portal.
        
        Args:
            traveler: Traveler identity
            portal_id: Portal to use
            avatar_id: Avatar ID
            carry_assets: Assets to bring
            
        Returns:
            TravelSession
        """
        if portal_id not in self._portals:
            raise ValueError("Portal not found")
        
        portal = self._portals[portal_id]
        
        if not portal.active:
            raise ValueError("Portal is inactive")
        
        session_id = hashlib.sha256(
            f"{traveler}_{portal_id}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        session = TravelSession(
            session_id=session_id,
            traveler=traveler,
            avatar_id=avatar_id,
            source_metaverse=portal.source_metaverse,
            target_metaverse=portal.target_metaverse,
            portal_id=portal_id,
            status="initiating",
            started_at=time.time(),
            carried_assets=carry_assets or []
        )
        
        self._sessions[session_id] = session
        
        # Execute travel
        await self._execute_travel(session, portal)
        
        return session
    
    async def _execute_travel(self, session: TravelSession,
                               portal: PortalLink):
        """Execute cross-metaverse travel."""
        try:
            # Step 1: Validate source state
            session.status = "validating_source"
            source_valid = await self._validate_platform(portal.source_metaverse)
            
            if not source_valid:
                session.status = "failed"
                return
            
            # Step 2: Export avatar
            session.status = "exporting_avatar"
            avatar_data = await self._export_avatar(
                session.avatar_id,
                portal.source_metaverse
            )
            
            # Step 3: Transfer assets
            session.status = "transferring_assets"
            for asset_id in session.carried_assets:
                await self._transfer_asset(
                    asset_id,
                    portal.source_metaverse,
                    portal.target_metaverse
                )
            
            # Step 4: Import to target
            session.status = "importing"
            await self._import_avatar(
                avatar_data,
                portal.target_metaverse,
                portal.target_location
            )
            
            # Step 5: Complete
            session.status = "completed"
            session.completed_at = time.time()
            
            logger.info(f"Travel completed: {session.session_id}")
            
        except Exception as e:
            session.status = "failed"
            logger.error(f"Travel failed: {e}")
    
    async def _validate_platform(self, platform: MetaversePlatform) -> bool:
        """Validate platform connectivity."""
        return self._connectors.get(platform, False)
    
    async def _export_avatar(self, avatar_id: str,
                              platform: MetaversePlatform) -> Dict:
        """Export avatar from platform."""
        return {
            "avatar_id": avatar_id,
            "source": platform.value,
            "protocol_version": self.PROTOCOL_VERSIONS[platform],
            "exported_at": time.time()
        }
    
    async def _import_avatar(self, avatar_data: Dict,
                              platform: MetaversePlatform,
                              location: Dict):
        """Import avatar to platform."""
        logger.debug(f"Importing avatar to {platform.value}")
    
    async def _transfer_asset(self, asset_id: str,
                               source: MetaversePlatform,
                               target: MetaversePlatform):
        """Transfer NFT asset between platforms."""
        logger.debug(f"Transferring asset {asset_id}")
    
    def get_portal(self, portal_id: str) -> Optional[PortalLink]:
        """Get portal by ID."""
        return self._portals.get(portal_id)
    
    def get_portals_from(self, platform: MetaversePlatform) -> List[PortalLink]:
        """Get portals originating from platform."""
        return [
            p for p in self._portals.values()
            if p.source_metaverse == platform or
               (p.bidirectional and p.target_metaverse == platform)
        ]
    
    def get_session(self, session_id: str) -> Optional[TravelSession]:
        """Get travel session."""
        return self._sessions.get(session_id)
    
    def deactivate_portal(self, portal_id: str, owner: str) -> bool:
        """Deactivate portal."""
        if portal_id not in self._portals:
            return False
        
        portal = self._portals[portal_id]
        
        if portal.owner != owner:
            return False
        
        portal.active = False
        return True
    
    def get_statistics(self) -> Dict:
        """Get portal statistics."""
        return {
            "total_portals": len(self._portals),
            "active_portals": sum(1 for p in self._portals.values() if p.active),
            "total_sessions": len(self._sessions),
            "completed_travels": sum(
                1 for s in self._sessions.values() if s.status == "completed"
            )
        }
    
    def __repr__(self) -> str:
        return f"CrossMetaversePortal(portals={len(self._portals)})"
