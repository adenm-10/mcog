# domains/nav/base.py
"""Base class for dynamical systems."""

from abc import ABC, abstractmethod
import jax.numpy as jnp

class DynamicalSystem(ABC):
    """Abstract base class for dynamical systems.
    
    A dynamical system defines:
    - State space and control space
    - Dynamics (how state evolves given control)
    - Cost function (what we're optimizing)
    - Visualization methods
    """
    
    def __init__(self, params, dt: float = 0.02):
        self.params = params
        self.dt = dt
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the system."""
        pass
    
    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of state space."""
        pass
    
    @property
    @abstractmethod
    def control_dim(self) -> int:
        """Dimension of control space."""
        pass
    
    @property
    @abstractmethod
    def default_initial_state(self) -> jnp.ndarray:
        """Default initial state."""
        pass
    
    @property
    @abstractmethod
    def u_max(self) -> float:
        """Maximum control magnitude."""
        pass
    
    @abstractmethod
    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Compute state derivatives: dx/dt = f(x, u)."""
        pass
    
    @abstractmethod
    def step(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Single integration step."""
        pass

