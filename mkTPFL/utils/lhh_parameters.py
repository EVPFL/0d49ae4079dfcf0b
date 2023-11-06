"""A module to keep track of parameters for the LHH scheme."""

from LHH import *

lhh_msg_len = 256
lhh_scheme = LHHScheme(lhh_msg_len)

def LHH_init(msg_len=None):
    global lhh_msg_len, lhh_scheme
    if msg_len and lhh_msg_len!=msg_len:
        lhh_msg_len = msg_len
        lhh_scheme = LHHScheme(msg_len)
    return lhh_scheme



