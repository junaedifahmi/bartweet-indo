"""
It always nice to have separate model definitions and its weights.
A good model definition also nice if it has separate configuration file, so instead changing your model, you can always
change the configuration file only.
"""

from models.bartweet import bartweet_id

__all__ = [bartweet_id]
