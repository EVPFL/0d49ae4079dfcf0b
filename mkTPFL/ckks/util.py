from HEAAN import ZZ, Ciphertext

from mkTPFL.ckks.ckks_parameters import *

def reviseMod(z, mod_q, ZZ_size=1):
    assert isinstance(mod_q, int) and mod_q > 0
    half_q = mod_q >> 1
    if isinstance(z, int):
        z_rem = z % mod_q
        if z_rem > half_q: z_rem -= mod_q
        return z_rem
    if isinstance(z, ZZ):
        z_rems = []
        for i in range(ZZ_size):
            zi = int(z[i])
            zi_rem = zi % mod_q
            if zi_rem > half_q: zi_rem -= mod_q
            z.setitem(i, zi_rem)
            z_rems.append(zi_rem)
        return z_rems


def getZZMAC(zz, mod):
    assert isinstance(zz, ZZ)
    mac_sum = 0
    for i in range(N):
        mac_i = int( zz[i] )
        mac_sum += mac_i
    return mac_sum%mod, mod


def getPlaintxtMAC(plain, mod=None):
    assert isinstance(plain, Plaintext)
    if not mod:
        mod = int( ring.qpows[plain.logq] )
    mx_mac = getZZMAC(plain.mx, mod=mod)[0]
    return mx_mac, mod


def getCiphtxtMAC(ciph, mod=None):
    assert isinstance(ciph, Ciphertext)
    if not mod:
        mod = int( ring.qpows[ciph.logq] )
    return getZZMAC(ciph.ax, mod=mod)


def getCiphtxtBXMAC(ciph, mod=None):
    assert isinstance(ciph, Ciphertext)
    if not mod:
        mod = int( ring.qpows[ciph.logq] )
    bx_mac = getZZMAC(ciph.bx, mod=mod)[0]
    return bx_mac%mod, mod

