"""
This script is used to transform the original preprocessed datasets into the format that required by each algorithms, where the node mapping is also included.
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime