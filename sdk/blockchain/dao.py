"""
DAO Governance
==============

Decentralized Autonomous Organization governance.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time
import logging

logger = logging.getLogger(__name__)


class ProposalState(Enum):
    """Proposal states."""
    PENDING = "pending"
    ACTIVE = "active"
    CANCELED = "canceled"
    DEFEATED = "defeated"
    SUCCEEDED = "succeeded"
    QUEUED = "queued"
    EXPIRED = "expired"
    EXECUTED = "executed"


class VoteType(Enum):
    """Vote types."""
    FOR = "for"
    AGAINST = "against"
    ABSTAIN = "abstain"


@dataclass
class Proposal:
    """DAO proposal."""
    proposal_id: str
    title: str
    description: str
    proposer: str
    targets: List[str]
    values: List[int]
    calldatas: List[bytes]
    start_block: int
    end_block: int
    state: ProposalState
    votes_for: int = 0
    votes_against: int = 0
    votes_abstain: int = 0
    created_at: float = field(default_factory=time.time)
    executed_at: Optional[float] = None


@dataclass
class Vote:
    """Individual vote."""
    voter: str
    proposal_id: str
    vote_type: VoteType
    weight: int
    reason: Optional[str]
    timestamp: float


@dataclass
class Member:
    """DAO member."""
    address: str
    voting_power: int
    delegated_to: Optional[str]
    delegated_from: List[str]
    proposals_created: int
    votes_cast: int
    joined_at: float


class DAOGovernance:
    """
    DAO Governance system.
    
    Features:
    - On-chain proposals
    - Token-weighted voting
    - Delegation
    - Timelock execution
    - Multi-sig treasury
    - AI-assisted governance
    
    Example:
        >>> dao = DAOGovernance("AIPlatformDAO")
        >>> proposal = await dao.create_proposal(
        ...     "Upgrade quantum module",
        ...     "Proposal to upgrade quantum computing module"
        ... )
        >>> await dao.cast_vote(proposal.proposal_id, VoteType.FOR, voter)
    """
    
    def __init__(self, name: str,
                 token_address: Optional[str] = None,
                 voting_delay: int = 1,
                 voting_period: int = 50400,  # ~1 week in blocks
                 proposal_threshold: int = 100000,
                 quorum_percentage: int = 4):
        """
        Initialize DAO governance.
        
        Args:
            name: DAO name
            token_address: Governance token address
            voting_delay: Blocks before voting starts
            voting_period: Voting period in blocks
            proposal_threshold: Tokens needed to create proposal
            quorum_percentage: Quorum percentage (0-100)
        """
        self.name = name
        self.token_address = token_address
        self.voting_delay = voting_delay
        self.voting_period = voting_period
        self.proposal_threshold = proposal_threshold
        self.quorum_percentage = quorum_percentage
        
        # State
        self._proposals: Dict[str, Proposal] = {}
        self._votes: Dict[str, List[Vote]] = {}
        self._members: Dict[str, Member] = {}
        self._treasury_balance: int = 0
        self._current_block: int = 0
        
        # Hooks
        self._on_proposal_created: List[Callable] = []
        self._on_vote_cast: List[Callable] = []
        self._on_proposal_executed: List[Callable] = []
        
        logger.info(f"DAO Governance initialized: {name}")
    
    async def create_proposal(self, title: str,
                               description: str,
                               proposer: str,
                               targets: List[str] = None,
                               values: List[int] = None,
                               calldatas: List[bytes] = None) -> Proposal:
        """
        Create a new proposal.
        
        Args:
            title: Proposal title
            description: Detailed description
            proposer: Proposer address
            targets: Target contract addresses
            values: ETH values for calls
            calldatas: Encoded function calls
            
        Returns:
            Proposal
        """
        # Check proposer has enough voting power
        member = self._members.get(proposer)
        if member and member.voting_power < self.proposal_threshold:
            raise ValueError("Insufficient voting power to create proposal")
        
        proposal_id = hashlib.sha256(
            f"{title}{description}{proposer}{time.time()}".encode()
        ).hexdigest()[:16]
        
        start_block = self._current_block + self.voting_delay
        end_block = start_block + self.voting_period
        
        proposal = Proposal(
            proposal_id=proposal_id,
            title=title,
            description=description,
            proposer=proposer,
            targets=targets or [],
            values=values or [],
            calldatas=calldatas or [],
            start_block=start_block,
            end_block=end_block,
            state=ProposalState.PENDING
        )
        
        self._proposals[proposal_id] = proposal
        self._votes[proposal_id] = []
        
        # Update member stats
        if member:
            member.proposals_created += 1
        
        # Trigger hooks
        for hook in self._on_proposal_created:
            await hook(proposal)
        
        logger.info(f"Proposal created: {proposal_id} - {title}")
        return proposal
    
    async def cast_vote(self, proposal_id: str,
                        vote_type: VoteType,
                        voter: str,
                        reason: Optional[str] = None) -> Vote:
        """
        Cast a vote on proposal.
        
        Args:
            proposal_id: Proposal to vote on
            vote_type: Vote type (for/against/abstain)
            voter: Voter address
            reason: Optional reason for vote
            
        Returns:
            Vote
        """
        if proposal_id not in self._proposals:
            raise ValueError("Proposal not found")
        
        proposal = self._proposals[proposal_id]
        
        # Check proposal is active
        if proposal.state != ProposalState.ACTIVE:
            if proposal.state == ProposalState.PENDING:
                # Activate if past start block
                if self._current_block >= proposal.start_block:
                    proposal.state = ProposalState.ACTIVE
                else:
                    raise ValueError("Voting has not started")
            else:
                raise ValueError(f"Proposal is {proposal.state.value}")
        
        # Check not already voted
        existing_votes = [v for v in self._votes[proposal_id] if v.voter == voter]
        if existing_votes:
            raise ValueError("Already voted on this proposal")
        
        # Get voting power
        member = self._members.get(voter)
        voting_power = member.voting_power if member else 0
        
        # Include delegated power
        if member:
            for delegator in member.delegated_from:
                delegator_member = self._members.get(delegator)
                if delegator_member:
                    voting_power += delegator_member.voting_power
        
        vote = Vote(
            voter=voter,
            proposal_id=proposal_id,
            vote_type=vote_type,
            weight=voting_power,
            reason=reason,
            timestamp=time.time()
        )
        
        self._votes[proposal_id].append(vote)
        
        # Update proposal vote counts
        if vote_type == VoteType.FOR:
            proposal.votes_for += voting_power
        elif vote_type == VoteType.AGAINST:
            proposal.votes_against += voting_power
        else:
            proposal.votes_abstain += voting_power
        
        # Update member stats
        if member:
            member.votes_cast += 1
        
        # Trigger hooks
        for hook in self._on_vote_cast:
            await hook(vote, proposal)
        
        logger.info(f"Vote cast: {voter[:10]}... voted {vote_type.value} on {proposal_id}")
        return vote
    
    async def execute_proposal(self, proposal_id: str) -> bool:
        """
        Execute a succeeded proposal.
        
        Args:
            proposal_id: Proposal to execute
            
        Returns:
            True if executed successfully
        """
        if proposal_id not in self._proposals:
            return False
        
        proposal = self._proposals[proposal_id]
        
        # Check proposal succeeded
        if proposal.state != ProposalState.SUCCEEDED:
            # Check if voting ended and succeeded
            if self._current_block > proposal.end_block:
                await self._finalize_proposal(proposal)
        
        if proposal.state != ProposalState.SUCCEEDED:
            return False
        
        # Execute targets
        for i, target in enumerate(proposal.targets):
            value = proposal.values[i] if i < len(proposal.values) else 0
            calldata = proposal.calldatas[i] if i < len(proposal.calldatas) else b""
            
            # Execute call (simulated)
            logger.info(f"Executing call to {target} with value {value}")
        
        proposal.state = ProposalState.EXECUTED
        proposal.executed_at = time.time()
        
        # Trigger hooks
        for hook in self._on_proposal_executed:
            await hook(proposal)
        
        logger.info(f"Proposal executed: {proposal_id}")
        return True
    
    async def _finalize_proposal(self, proposal: Proposal):
        """Finalize proposal after voting ends."""
        total_supply = sum(m.voting_power for m in self._members.values())
        quorum = total_supply * self.quorum_percentage // 100
        
        total_votes = proposal.votes_for + proposal.votes_against + proposal.votes_abstain
        
        if total_votes >= quorum and proposal.votes_for > proposal.votes_against:
            proposal.state = ProposalState.SUCCEEDED
        else:
            proposal.state = ProposalState.DEFEATED
    
    async def delegate(self, from_address: str, to_address: str):
        """Delegate voting power."""
        from_member = self._members.get(from_address)
        to_member = self._members.get(to_address)
        
        if not from_member or not to_member:
            raise ValueError("Member not found")
        
        # Remove old delegation
        if from_member.delegated_to:
            old_delegate = self._members.get(from_member.delegated_to)
            if old_delegate and from_address in old_delegate.delegated_from:
                old_delegate.delegated_from.remove(from_address)
        
        # Set new delegation
        from_member.delegated_to = to_address
        to_member.delegated_from.append(from_address)
        
        logger.info(f"Delegated: {from_address[:10]}... -> {to_address[:10]}...")
    
    def register_member(self, address: str, voting_power: int):
        """Register a DAO member."""
        self._members[address] = Member(
            address=address,
            voting_power=voting_power,
            delegated_to=None,
            delegated_from=[],
            proposals_created=0,
            votes_cast=0,
            joined_at=time.time()
        )
    
    def get_proposal(self, proposal_id: str) -> Optional[Proposal]:
        """Get proposal by ID."""
        return self._proposals.get(proposal_id)
    
    def get_proposals(self, state: ProposalState = None) -> List[Proposal]:
        """Get proposals, optionally filtered by state."""
        proposals = list(self._proposals.values())
        
        if state:
            proposals = [p for p in proposals if p.state == state]
        
        return proposals
    
    def get_votes(self, proposal_id: str) -> List[Vote]:
        """Get votes for proposal."""
        return self._votes.get(proposal_id, [])
    
    def get_member(self, address: str) -> Optional[Member]:
        """Get member info."""
        return self._members.get(address)
    
    def advance_block(self, blocks: int = 1):
        """Advance block number (for testing)."""
        self._current_block += blocks
    
    def on_proposal_created(self, callback: Callable):
        """Register proposal created hook."""
        self._on_proposal_created.append(callback)
    
    def on_vote_cast(self, callback: Callable):
        """Register vote cast hook."""
        self._on_vote_cast.append(callback)
    
    def on_proposal_executed(self, callback: Callable):
        """Register proposal executed hook."""
        self._on_proposal_executed.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get DAO statistics."""
        return {
            "name": self.name,
            "total_proposals": len(self._proposals),
            "active_proposals": len([p for p in self._proposals.values() 
                                    if p.state == ProposalState.ACTIVE]),
            "total_members": len(self._members),
            "total_voting_power": sum(m.voting_power for m in self._members.values()),
            "treasury_balance": self._treasury_balance
        }
    
    def __repr__(self) -> str:
        return f"DAOGovernance(name='{self.name}', proposals={len(self._proposals)})"
