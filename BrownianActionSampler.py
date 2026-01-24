import numpy as np

class BrownianActionSampler:
    def __init__(
        self,
        action_space,
        dt: float = 0.1,
        volatility: np.ndarray = None,
    ):
        """
        Args:
            action_space: Gym action space (Box)
            dt: Time step for Brownian motion (smaller = smoother trajectories)
            volatility: Per-dimension volatility [steering, gas, brake]
                        Higher = more variation in that dimension
        """
        self.action_space = action_space
        self.low = action_space.low
        self.high = action_space.high
        self.dim = action_space.shape[0]
        self.dt = dt
        
        # Default volatility tuned for CarRacing
        # Steering is more volatile, gas moderate, brake minimal
        if volatility is None:
            self.volatility = np.array([0.3, 0.2, 0.1], dtype=np.float32)
        else:
            self.volatility = np.array(volatility, dtype=np.float32)
        
        self.reset()
    
    def reset(self, initial_action: np.ndarray = None):
        """
        Reset the sampler state.
        
        Args:
            initial_action: Starting action. If None, starts with slight forward bias.
        """
        if initial_action is not None:
            self.current_action = np.array(initial_action, dtype=np.float32)
        else:
            # Start with slight forward bias (some gas, no brake, centered steering)
            self.current_action = np.array([0.0, 0.3, 0.0], dtype=np.float32)
        
        return self.current_action.copy()
    
    def sample(self) -> np.ndarray:
        """
        Generate next action using Brownian motion.
        
        Returns:
            action: np.ndarray of shape (action_dim,), dtype float32
        """
        # Brownian increment: dW = sqrt(dt) * N(0,1)
        noise = np.random.randn(self.dim).astype(np.float32)
        increment = np.sqrt(self.dt) * noise * self.volatility
        
        # Update action
        self.current_action = self.current_action + increment
        
        # Clamp to valid range
        self.current_action = np.clip(self.current_action, self.low, self.high)
        
        return self.current_action.copy()
    
    def sample_with_forward_bias(self, gas_bias: float = 0.2) -> np.ndarray:
        """
        Sample with a bias towards forward motion.
        
        Useful for CarRacing where we want the car to generally move forward
        to explore the track, rather than sitting still.
        
        Args:
            gas_bias: Bias towards gas pedal (applied each step before clamping)
        """
        action = self.sample()
        
        # Add forward bias to gas
        action[1] = np.clip(action[1] + gas_bias * self.dt, self.low[1], self.high[1])
        
        # Reduce brake when gas is high (don't gas and brake simultaneously)
        if action[1] > 0.5:
            action[2] = action[2] * 0.5
        
        return action
