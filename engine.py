import numpy as np
CP=4180.0

def simulate(th,tc,mh,mc,ua):
    if not (th>tc and mh>0 and mc>0 and ua>0):
        raise ValueError('Hot inlet must be warmer than cold inlet; flow and UA must be positive.')
    ch,cc=mh*CP,mc*CP
    cmin,cmax=min(ch,cc),max(ch,cc)
    cr=cmin/cmax
    ntu=ua/cmin
    eff=ntu/(1+ntu) if abs(cr-1)<1e-10 else -np.expm1(-ntu*(1-cr))/(1-cr*np.exp(-ntu*(1-cr)))
    q=eff*cmin*(th-tc)
    return {'hot_out':th-q/ch,'cold_out':tc+q/cc,'heat_kw':q/1000,'effectiveness':eff,'ntu':ntu,'balance_error_w':abs(ch*(th-(th-q/ch))-cc*((tc+q/cc)-tc))}

def batch(X):
    return np.array([[simulate(*x)['hot_out'],simulate(*x)['cold_out']] for x in X])
