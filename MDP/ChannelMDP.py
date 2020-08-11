"""
This MDP was designed as an example of the dangers of "reasonable" but inaccurate state abstraction.

The MDP consists of a starting state from which a large number of actions are available. Each of these actions
transitions (deterministically or stochastically) to one of many 'channel' states. Each of these channel states has
access to the same modest number of actions. In some high proportion of them, at least one of these actions will result
in catastrophic failure some small percentage of the time. These are the "fail-deadly" channels. The small remainder
are the "fail-safe" channels.

The motivation behind the design is that it can be reasonably assumed that a decent model-based method will recognize
many of the fail-deadly channels, but not all of them as the state-action space becomes too large to exhaustively
explore
"""

