"""
Configuration parameters for the camera decision system.
These values can be adjusted to tune the behavior of the system.
"""

# Content Analysis Parameters
MIN_SLIDE_CONFIDENCE = 0.5  # Minimum confidence to consider content as slide-related
MIN_POSE_CONFIDENCE = 0.7   # Minimum confidence for pose detection

# Overlay Decision Parameters
OVERLAY_THRESHOLD = 0.5     # Minimum confidence to enable overlay
MIN_CORNER_SAFETY = 0.7     # Minimum safety score for corners to be considered
MIN_GESTURE_CONFIDENCE = 0.8  # Minimum confidence for gestures to be considered

# Confidence Weights
CONTENT_WEIGHT = 0.4        # Weight for content analysis in overall confidence
POSE_WEIGHT = 0.6           # Weight for pose analysis in overall confidence
CORNER_WEIGHT = 0.4         # Weight for corner safety in overlay confidence
GESTURE_WEIGHT = 0.6        # Weight for gestures in overlay confidence

# Primary Feed Parameters
PRIMARY_SLIDE_PENALTY = 0.7  # Multiplier for overlay confidence when slides are primary

# Confidence Boosts and Penalties
MODEL_AGREEMENT_BOOST = 1.5  # Boost when content and pose models agree
MODEL_DISAGREEMENT_PENALTY = 0.7  # Penalty when content and pose models disagree
NO_POSE_PENALTY = 0.6       # Penalty when no pose data is available 