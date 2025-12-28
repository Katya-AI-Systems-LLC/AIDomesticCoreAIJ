"""
Avatar System
=============

Cross-metaverse avatar management.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import hashlib
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class AvatarAppearance:
    """Avatar visual appearance."""
    body_type: str
    skin_color: str
    hair_style: str
    hair_color: str
    eye_color: str
    height: float
    custom_features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AvatarWearable:
    """Wearable item."""
    wearable_id: str
    name: str
    slot: str  # head, body, legs, feet, accessory
    rarity: str
    nft_address: Optional[str] = None
    nft_chain: Optional[str] = None
    metadata_uri: Optional[str] = None


@dataclass
class Avatar:
    """User avatar."""
    avatar_id: str
    owner: str  # DID or wallet address
    display_name: str
    appearance: AvatarAppearance
    wearables: List[AvatarWearable]
    animations: List[str]
    voice_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class AvatarSystem:
    """
    Cross-metaverse avatar system.
    
    Features:
    - Portable avatars across metaverses
    - NFT wearables
    - Decentralized identity
    - Animation library
    - Voice synthesis
    
    Example:
        >>> system = AvatarSystem()
        >>> avatar = system.create_avatar(owner, "CoolName")
        >>> system.equip_wearable(avatar.avatar_id, wearable)
    """
    
    SUPPORTED_METAVERSES = [
        "decentraland", "sandbox", "somnium",
        "cryptovoxels", "aiplatform"
    ]
    
    WEARABLE_SLOTS = [
        "head", "face", "upper_body", "lower_body",
        "feet", "hand_left", "hand_right", "accessory"
    ]
    
    def __init__(self):
        """Initialize avatar system."""
        self._avatars: Dict[str, Avatar] = {}
        self._wearables: Dict[str, AvatarWearable] = {}
        
        logger.info("Avatar System initialized")
    
    def create_avatar(self, owner: str,
                      display_name: str,
                      appearance: AvatarAppearance = None) -> Avatar:
        """
        Create new avatar.
        
        Args:
            owner: Owner DID/address
            display_name: Display name
            appearance: Visual appearance
            
        Returns:
            Created avatar
        """
        avatar_id = hashlib.sha256(
            f"{owner}_{display_name}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        default_appearance = appearance or AvatarAppearance(
            body_type="humanoid",
            skin_color="#f5d0c5",
            hair_style="short",
            hair_color="#4a3c31",
            eye_color="#5b8c6e",
            height=1.75
        )
        
        avatar = Avatar(
            avatar_id=avatar_id,
            owner=owner,
            display_name=display_name,
            appearance=default_appearance,
            wearables=[],
            animations=["idle", "walk", "run", "jump", "wave"]
        )
        
        self._avatars[avatar_id] = avatar
        
        logger.info(f"Avatar created: {display_name} ({avatar_id})")
        return avatar
    
    def get_avatar(self, avatar_id: str) -> Optional[Avatar]:
        """Get avatar by ID."""
        return self._avatars.get(avatar_id)
    
    def get_avatars_by_owner(self, owner: str) -> List[Avatar]:
        """Get all avatars owned by address."""
        return [a for a in self._avatars.values() if a.owner == owner]
    
    def update_appearance(self, avatar_id: str,
                          appearance: AvatarAppearance):
        """Update avatar appearance."""
        if avatar_id in self._avatars:
            self._avatars[avatar_id].appearance = appearance
    
    def register_wearable(self, name: str,
                          slot: str,
                          rarity: str = "common",
                          nft_address: str = None,
                          nft_chain: str = None) -> AvatarWearable:
        """
        Register wearable item.
        
        Args:
            name: Wearable name
            slot: Equipment slot
            rarity: Rarity level
            nft_address: NFT contract address
            nft_chain: Blockchain
            
        Returns:
            Registered wearable
        """
        wearable_id = hashlib.sha256(
            f"{name}_{slot}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        wearable = AvatarWearable(
            wearable_id=wearable_id,
            name=name,
            slot=slot,
            rarity=rarity,
            nft_address=nft_address,
            nft_chain=nft_chain
        )
        
        self._wearables[wearable_id] = wearable
        return wearable
    
    def equip_wearable(self, avatar_id: str,
                       wearable_id: str) -> bool:
        """
        Equip wearable to avatar.
        
        Args:
            avatar_id: Avatar ID
            wearable_id: Wearable ID
            
        Returns:
            True if equipped
        """
        if avatar_id not in self._avatars:
            return False
        
        if wearable_id not in self._wearables:
            return False
        
        avatar = self._avatars[avatar_id]
        wearable = self._wearables[wearable_id]
        
        # Remove existing item in same slot
        avatar.wearables = [
            w for w in avatar.wearables if w.slot != wearable.slot
        ]
        
        avatar.wearables.append(wearable)
        
        logger.info(f"Equipped {wearable.name} to {avatar.display_name}")
        return True
    
    def unequip_wearable(self, avatar_id: str, slot: str) -> bool:
        """Unequip wearable from slot."""
        if avatar_id not in self._avatars:
            return False
        
        avatar = self._avatars[avatar_id]
        avatar.wearables = [w for w in avatar.wearables if w.slot != slot]
        return True
    
    def export_avatar(self, avatar_id: str,
                      target_metaverse: str) -> Dict:
        """
        Export avatar for use in another metaverse.
        
        Args:
            avatar_id: Avatar ID
            target_metaverse: Target platform
            
        Returns:
            Export data
        """
        if avatar_id not in self._avatars:
            raise ValueError("Avatar not found")
        
        avatar = self._avatars[avatar_id]
        
        # Generate portable format
        export_data = {
            "version": "1.0",
            "format": "aiplatform_avatar",
            "avatar_id": avatar.avatar_id,
            "owner": avatar.owner,
            "display_name": avatar.display_name,
            "target_metaverse": target_metaverse,
            "appearance": {
                "body_type": avatar.appearance.body_type,
                "skin_color": avatar.appearance.skin_color,
                "hair_style": avatar.appearance.hair_style,
                "hair_color": avatar.appearance.hair_color,
                "eye_color": avatar.appearance.eye_color,
                "height": avatar.appearance.height,
                "custom": avatar.appearance.custom_features
            },
            "wearables": [
                {
                    "name": w.name,
                    "slot": w.slot,
                    "nft": {
                        "address": w.nft_address,
                        "chain": w.nft_chain
                    } if w.nft_address else None
                }
                for w in avatar.wearables
            ],
            "animations": avatar.animations,
            "exported_at": time.time()
        }
        
        logger.info(f"Avatar exported for {target_metaverse}")
        return export_data
    
    def import_avatar(self, import_data: Dict,
                      new_owner: str = None) -> Avatar:
        """
        Import avatar from another metaverse.
        
        Args:
            import_data: Export data
            new_owner: New owner (if different)
            
        Returns:
            Imported avatar
        """
        appearance = AvatarAppearance(
            body_type=import_data["appearance"]["body_type"],
            skin_color=import_data["appearance"]["skin_color"],
            hair_style=import_data["appearance"]["hair_style"],
            hair_color=import_data["appearance"]["hair_color"],
            eye_color=import_data["appearance"]["eye_color"],
            height=import_data["appearance"]["height"],
            custom_features=import_data["appearance"].get("custom", {})
        )
        
        avatar = self.create_avatar(
            new_owner or import_data["owner"],
            import_data["display_name"],
            appearance
        )
        
        # Import wearables
        for w_data in import_data.get("wearables", []):
            wearable = self.register_wearable(
                w_data["name"],
                w_data["slot"],
                nft_address=w_data.get("nft", {}).get("address"),
                nft_chain=w_data.get("nft", {}).get("chain")
            )
            self.equip_wearable(avatar.avatar_id, wearable.wearable_id)
        
        return avatar
    
    def __repr__(self) -> str:
        return f"AvatarSystem(avatars={len(self._avatars)})"
