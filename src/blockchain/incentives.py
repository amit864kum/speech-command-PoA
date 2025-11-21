"""Incentive and staking mechanisms for blockchain FL."""

from typing import Dict, List, Optional
from collections import defaultdict
import time


class IncentiveManager:
    """Manages rewards and penalties for FL participants."""
    
    def __init__(
        self,
        base_reward: float = 10.0,
        quality_bonus: float = 5.0,
        slash_penalty: float = 50.0
    ):
        """Initialize incentive manager.
        
        Args:
            base_reward: Base reward for participation
            quality_bonus: Bonus for high-quality contributions
            slash_penalty: Penalty for malicious behavior
        """
        self.base_reward = base_reward
        self.quality_bonus = quality_bonus
        self.slash_penalty = slash_penalty
        
        # Track balances
        self.balances = defaultdict(float)
        
        # Track rewards history
        self.reward_history = []
        
        # Track penalties
        self.penalty_history = []
    
    def calculate_reward(
        self,
        participant: str,
        contribution_quality: float,
        num_samples: int,
        round_number: int
    ) -> float:
        """Calculate reward for a participant.
        
        Args:
            participant: Participant address
            contribution_quality: Quality score (0-1)
            num_samples: Number of training samples
            round_number: FL round number
            
        Returns:
            Reward amount
        """
        # Base reward
        reward = self.base_reward
        
        # Quality bonus (scaled by quality score)
        reward += self.quality_bonus * contribution_quality
        
        # Sample size bonus (logarithmic scaling)
        import math
        sample_bonus = math.log(1 + num_samples / 100)
        reward += sample_bonus
        
        # Record reward
        self.reward_history.append({
            "participant": participant,
            "reward": reward,
            "round": round_number,
            "quality": contribution_quality,
            "samples": num_samples,
            "timestamp": time.time()
        })
        
        # Update balance
        self.balances[participant] += reward
        
        return reward
    
    def apply_penalty(
        self,
        participant: str,
        penalty_type: str,
        evidence: str
    ) -> float:
        """Apply penalty to a participant.
        
        Args:
            participant: Participant address
            penalty_type: Type of violation
            evidence: Evidence CID or description
            
        Returns:
            Penalty amount
        """
        penalty = self.slash_penalty
        
        # Record penalty
        self.penalty_history.append({
            "participant": participant,
            "penalty": penalty,
            "type": penalty_type,
            "evidence": evidence,
            "timestamp": time.time()
        })
        
        # Deduct from balance
        self.balances[participant] -= penalty
        
        return penalty
    
    def get_balance(self, participant: str) -> float:
        """Get participant balance.
        
        Args:
            participant: Participant address
            
        Returns:
            Current balance
        """
        return self.balances[participant]
    
    def get_total_rewards(self, participant: str) -> float:
        """Get total rewards earned by participant.
        
        Args:
            participant: Participant address
            
        Returns:
            Total rewards
        """
        return sum(
            r["reward"] for r in self.reward_history 
            if r["participant"] == participant
        )
    
    def get_statistics(self) -> Dict:
        """Get incentive statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_rewards_distributed": sum(r["reward"] for r in self.reward_history),
            "total_penalties_applied": sum(p["penalty"] for p in self.penalty_history),
            "num_participants": len(self.balances),
            "avg_balance": sum(self.balances.values()) / len(self.balances) if self.balances else 0,
            "num_rewards": len(self.reward_history),
            "num_penalties": len(self.penalty_history)
        }


class StakingManager:
    """Manages staking for FL participants."""
    
    def __init__(
        self,
        min_stake: float = 100.0,
        max_stake: float = 10000.0,
        slash_rate: float = 0.5
    ):
        """Initialize staking manager.
        
        Args:
            min_stake: Minimum stake required
            max_stake: Maximum stake allowed
            slash_rate: Fraction of stake to slash for violations
        """
        self.min_stake = min_stake
        self.max_stake = max_stake
        self.slash_rate = slash_rate
        
        # Track stakes
        self.stakes = defaultdict(float)
        
        # Track locked stakes
        self.locked_stakes = defaultdict(float)
        
        # Staking history
        self.stake_history = []
    
    def stake(self, participant: str, amount: float) -> bool:
        """Stake tokens.
        
        Args:
            participant: Participant address
            amount: Amount to stake
            
        Returns:
            True if successful
        """
        if amount < self.min_stake:
            print(f"[Staking] Amount {amount} below minimum {self.min_stake}")
            return False
        
        if self.stakes[participant] + amount > self.max_stake:
            print(f"[Staking] Would exceed maximum stake {self.max_stake}")
            return False
        
        self.stakes[participant] += amount
        
        self.stake_history.append({
            "participant": participant,
            "action": "stake",
            "amount": amount,
            "timestamp": time.time()
        })
        
        print(f"[Staking] {participant} staked {amount}. Total: {self.stakes[participant]}")
        return True
    
    def unstake(self, participant: str, amount: float) -> bool:
        """Unstake tokens.
        
        Args:
            participant: Participant address
            amount: Amount to unstake
            
        Returns:
            True if successful
        """
        available = self.stakes[participant] - self.locked_stakes[participant]
        
        if amount > available:
            print(f"[Staking] Insufficient available stake. Available: {available}")
            return False
        
        self.stakes[participant] -= amount
        
        self.stake_history.append({
            "participant": participant,
            "action": "unstake",
            "amount": amount,
            "timestamp": time.time()
        })
        
        print(f"[Staking] {participant} unstaked {amount}. Remaining: {self.stakes[participant]}")
        return True
    
    def lock_stake(self, participant: str, amount: float) -> bool:
        """Lock stake for participation.
        
        Args:
            participant: Participant address
            amount: Amount to lock
            
        Returns:
            True if successful
        """
        available = self.stakes[participant] - self.locked_stakes[participant]
        
        if amount > available:
            return False
        
        self.locked_stakes[participant] += amount
        return True
    
    def unlock_stake(self, participant: str, amount: float):
        """Unlock stake after participation.
        
        Args:
            participant: Participant address
            amount: Amount to unlock
        """
        self.locked_stakes[participant] = max(0, self.locked_stakes[participant] - amount)
    
    def slash_stake(self, participant: str, reason: str) -> float:
        """Slash stake for malicious behavior.
        
        Args:
            participant: Participant address
            reason: Reason for slashing
            
        Returns:
            Amount slashed
        """
        slash_amount = self.stakes[participant] * self.slash_rate
        self.stakes[participant] -= slash_amount
        
        self.stake_history.append({
            "participant": participant,
            "action": "slash",
            "amount": slash_amount,
            "reason": reason,
            "timestamp": time.time()
        })
        
        print(f"[Staking] Slashed {slash_amount} from {participant}. Reason: {reason}")
        return slash_amount
    
    def get_stake(self, participant: str) -> float:
        """Get participant's stake.
        
        Args:
            participant: Participant address
            
        Returns:
            Current stake
        """
        return self.stakes[participant]
    
    def get_available_stake(self, participant: str) -> float:
        """Get participant's available (unlocked) stake.
        
        Args:
            participant: Participant address
            
        Returns:
            Available stake
        """
        return self.stakes[participant] - self.locked_stakes[participant]
    
    def is_eligible(self, participant: str) -> bool:
        """Check if participant has minimum stake.
        
        Args:
            participant: Participant address
            
        Returns:
            True if eligible
        """
        return self.stakes[participant] >= self.min_stake
    
    def get_statistics(self) -> Dict:
        """Get staking statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_staked": sum(self.stakes.values()),
            "total_locked": sum(self.locked_stakes.values()),
            "num_stakers": len([s for s in self.stakes.values() if s >= self.min_stake]),
            "avg_stake": sum(self.stakes.values()) / len(self.stakes) if self.stakes else 0,
            "total_slashed": sum(
                h["amount"] for h in self.stake_history 
                if h["action"] == "slash"
            )
        }