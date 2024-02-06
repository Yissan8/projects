


import numpy as np
import inspect
from pprint import pprint

from jupyter import * # local helper functions useful in Jupyter
import black_merton_scholes as bms

def antithetic_normal(n_days, n_paths):
    assert n_paths%2 == 0
    n2 = int(n_paths/2)
    z = np.random.normal(0,1, (n_days, n2))
    return np.hstack((z,-z))

# Python voodoo: dataclass allows a usage very similar to a Matlab struct
from dataclasses import dataclass
@dataclass
class measure:
    ex_r: np.array
    h:    np.array
    z:    np.array = None

class model:
    def __str__(self):
        string = self.__class__.__name__ + '(\n'
        #breakpoint()
        is_public = lambda name: not (name.startswith('__') and name.endswith('__'))
        is_field = lambda name: hasattr(self, name) and not inspect.ismethod(getattr(self, name))
        fields = self.__dict__
        for no,name in enumerate(fields):
            if is_public(name) and is_field(name):
                string += "    %s = %r" % (name, getattr(self, name))
            if no < len(fields)-1:
                string += ','
            string += '\n'
        string += ')'
        return string

    def __repr__(self):
        return str(self).replace('\n','').replace(' ','')
    
class ngarch(model):    
    def variance_targeting(self, var_target):
        omega = (1 - self.persistence())*var_target
        return omega
        
    def persistence(self):
        return self.alpha*(1 + self.gamma**2) + self.beta

    def uncond_var(self):
        return self.omega/(1 - self.persistence())
    
    def simulateP(self, S_t0, n_days, n_paths, h_tp1, z=None):
        '''Simulate excess returns and their variance under the P measure
        
        We consider that the simulation is starting at t0, and tp0 is a shorthand for "time
        t0+1" where p in tp1 stands for plus."

        This method simulates *excess* log-returns; the risk-free rate must be added outside
        this function to get the full log-return. This allows using different risk-free rates
        to price options at different horizons with the same core simulations.

        Args:
            S_t0:    Spot price at the beginning of the simulation
            n_days:  Length of the simulation
            n_paths: Number of paths in the simulation
            h_tp1:   Measurable at t0. Note that
            z:       The N(0,1) shocks for the simulation (optional) 

        Returns:
            ex_r:    Excess log-returns of the underlying (np.array: n_days x n_paths)
            h:       Corresponding variance (np.array: n_days+1 x n_paths)
            z:       The shocks used in the simulation
        '''
        ex_r = np.full((n_days, n_paths), np.nan)
        h = np.full((n_days+1, n_paths), np.nan)
        h[0,:] = h_tp1 # because indices start at 0 in Python
        if z is None:
            z = antithetic_normal(n_days, n_paths)

        # Going from 1 day to n_days ahead
        #  t0+1      is tn==0 because indices start at 0 in Python
        #  t0+n_days is consequently tn==n_days-1
        for tn in range(0,n_days):            
            # Simulate returns            
            sig = np.sqrt(h[tn,:])
            ex_r[tn,:] = self.lmbda*sig - 0.5*h[tn,:] + sig*z[tn,:]
            
            # Update the variance paths
            h[tn+1,:] = self.omega + self.alpha*h[tn,:]*(z[tn,:] - self.gamma)**2 + self.beta*h[tn,:]

        return ex_r, h, z


    def simulateQ(self, S_t0, n_days, n_paths, h_tp1, z=None):
        '''Simulate excess returns and their variance under the Q measure
        
        We consider that the simulation is starting at t0, and tp0 is a shorthand for "time
        t0+1" where p in tp1 stands for plus."

        This method simulates *excess* log-returns; the risk-free rate must be added outside
        this function to get the full log-return. This allows using different risk-free rates
        to price options at different horizons with the same core simulations.

        Args:
            S_t0:    Spot price at the beginning of the simulation
            n_days:  Length of the simulation
            n_paths: Number of paths in the simulation
            h_tp1:   Measurable at t0. Note that

        Returns:
            ex_r:    Excess log-returns of the underlying (np.array: n_days x n_paths)
            h:       Corresponding variance (np.array: n_days+1 x n_paths)
            z:       The shocks used in the simulation
        '''
        ex_r = np.full((n_days, n_paths), np.nan)
        h = np.full((n_days+1, n_paths), np.nan)
        h[0,:] = h_tp1 # because indices start at 0 in Python
        if z is None:
            z = antithetic_normal(n_days, n_paths)

        gamma_star = self.gamma + self.lmbda
        
        # Going from 1 day to n_days ahead
        #  t0+1      is tn==0 because indices start at 0 in Python
        #  t0+n_days is consequently tn==n_days-1
        for tn in range(0,n_days):
            # Simulate returns
            ex_r[tn,:] = -0.5*h[tn,:] + np.sqrt(h[tn,:])*z[tn,:]
            
            # Update the variance paths
            h[tn+1,:] = self.omega + self.alpha*h[tn,:]*(z[tn,:] - gamma_star)**2 + self.beta*h[tn,:]

        return ex_r, h, z
    
    #crÃ©ation d'une mÃ©thode afin de simuler des trajctoires de rendements excÃ©dentaires sous Blac-Scholes
    def BSsimulateQ(self, S_t0, n_days, n_paths, h_tp1, z=None):
        '''Simulate excess returns and their variance under the Q measure
        
        We consider that the simulation is starting at t0, and tp0 is a shorthand for "time
        t0+1" where p in tp1 stands for plus."
    
        This method simulates *excess* log-returns; the risk-free rate must be added outside
        this function to get the full log-return. This allows using different risk-free rates
        to price options at different horizons with the same core simulations.
    
        Args:
            S_t0:    Spot price at the beginning of the simulation
            n_days:  Length of the simulation
            n_paths: Number of paths in the simulation
            h_tp1:   Measurable at t0. Note that
    
        Returns:
            ex_r:    Excess log-returns of the underlying (np.array: n_days x n_paths)
            h:       Corresponding variance : constant, since we're in the Black_Scholes model'
            z:       The shocks used in the simulation
        '''
        ex_r = np.full((n_days, n_paths), np.nan)
        h = np.full((n_days+1, n_paths), np.nan)
        h[0,:] = h_tp1 # because indices start at 0 in Python
        if z is None:
            z = antithetic_normal(n_days, n_paths)
    
        gamma_star = self.gamma + self.lmbda
        
        # Going from 1 day to n_days ahead
        #  t0+1      is tn==0 because indices start at 0 in Python
        #  t0+n_days is consequently tn==n_days-1
        for tn in range(0,n_days):
            # Simulate returns
            ex_r[tn,:] = -0.5*h_tp1 + np.sqrt(h_tp1)*z[tn,:]
            
            
        return ex_r, h, z


#Fonction Black-Scholes 
def BS(S,K,T,vol,r,t):
    d1= (np.log(S/K)+T*(r + 0.5*vol**2))/(vol*np.sqrt(T))
    d2= d1 - vol*np.sqrt(T)
    if t =="c":
        p = (S*norm.cdf(d1) - K*norm.cdf(d2)*np.exp(-r*T))
    elif t == "p" :
        p = (K*norm.cdf(-d2)*np.exp(-r*T) - S*norm.cdf(-d1))
    return p

#Fonction  delta de Black-Scholes 
def delta(S,K,T,vol,r):
    d1= (np.log(S/K)+T*(r + 0.5*vol**2))/(vol*np.sqrt(T))
    return - norm.cdf(-d1)



#Fonction qui sert Ã  visualiser la moyenne des rendements excÃ©dentaires ainsi que l'ERP
def plot_excess_return_forecasts(horizon, P, Q, annualized=False):
    ann = [1.0]
    y_prefix = ''
    if annualized:
        ann = horizon
        y_prefix = 'Annualized '

    P.expected_ex_r = np.mean(np.exp(P.ex_r),axis=1)-1
    Q.expected_ex_r = np.mean(np.exp(Q.ex_r),axis=1)-1
    
    fig = plt.subplots(2,1, figsize=(14,10))

    ax = plt.subplot(2,1, 1)
    ax.plot(horizon, P.expected_ex_r/ann[0], label='P ex. returns forecasts (sim)')
    ax.plot(horizon, Q.expected_ex_r/ann[0], label='Q ex. returns forecasts (sim)')
    ax.legend(loc='upper right')   
    ax.set_ylabel(y_prefix+'Excess Returns')

    ax = plt.subplot(2,1, 2)
    # ERP is accumulated between t0 and horizon h (think about a buy and hold position)
    erp = np.cumsum(P.expected_ex_r - Q.expected_ex_r)/ann # Theoretically, the "- Q.expected_ex_r" part is useless here
    ax.plot(horizon, erp)    
    ax.set_ylabel(y_prefix+'Equity Risk Premium')
    ax.set_xlabel('Years to Maturity')
    
    
    
    
    
#Fonction qui sert Ã  visualiser l'Ã©volution de la moyenne de la variance sous P et sous Q ainsi que la VRP
def plot_variance_forecasts(horizon, P, Q, annualized=False):
    ann = [1.0]
    y_prefix = ''
    if annualized:
        ann = horizon
        ann1 = ann[:-1]
        y_prefix = 'Annualized '
    else :
        ann1 = ann


    
    P.var = np.var(P.ex_r,axis=1)
       
    Q.var = np.var(Q.ex_r,axis=1)
    
    
    P.expected_h = np.mean(P.h,axis=1)
    Q.expected_h = np.mean(Q.h,axis=1)
    
    fig = plt.subplots(4,1, figsize=(20,30))

    ax = plt.subplot(4,1, 1)
    ax.plot(horizon[:-1], P.var/ann[0], label='P Variance forecasts (sim)')
    ax.plot(horizon[:-1], Q.var/ann[0], label='Q Variance forecasts (sim)')
    ax.legend(loc='upper right')   
    ax.set_ylabel(y_prefix+'Variance')
    ax.set_title("Variances by taking the variance of simulations returns at each timestep")

    ax = plt.subplot(4,1, 2)
    
    erp = np.cumsum(P.var - Q.var)/ann1 # Theoretically, the "- Q.expected_ex_r" part is useless here
    ax.plot(horizon[:-1], erp)    
    ax.set_ylabel(y_prefix+'Variance Risk Premium')
    ax.set_xlabel('Years to Maturity')
    ax.set_title("VRP by taking the variance of simulations returns at each timestep")
    
    
    ax = plt.subplot(4,1, 3)
    ax.plot(horizon, P.expected_h/ann[0], label='P Variance forecasts (sim)')
    ax.plot(horizon, Q.expected_h/ann[0], label='Q Variance forecasts (sim)')
    ax.legend(loc='upper right')   
    ax.set_ylabel(y_prefix+'Variance')
    ax.set_title("Variances by taking the mean of the forecasted GARCH h at each timestep")

    ax = plt.subplot(4,1, 4)
    
    erp = np.cumsum(P.expected_h - Q.expected_h)/ann # Theoretically, the "- Q.expected_ex_r" part is useless here
    ax.plot(horizon, erp)    
    ax.set_ylabel(y_prefix+'Variance Risk Premium')
    ax.set_xlabel('Years to Maturity')
    ax.set_title("VRP by taking the mean of the forecasted GARCH h at each timestep")



#Fonction qui renvoie la valeur du put europÃ©en sous une dynamique NGARCH et sous BMS, pour les 3 maturitÃ©s demandÃ©es
def putval(S_t0,ndays,n_paths,vol,rf,K):
    
    #ng.omega = ng.variance_targeting( (vol**2)*dt )
    
    
    h   = vol **2 / 365 #variance journaliÃ¨re
    
    Q = measure( *ng.simulateQ(S_t0,ndays,n_paths,h) ) #simulation des rendements excÃ©dentaires sous Q
    S = S_t0*np.exp(np.cumsum(Q.ex_r + rf/365 ,axis = 0)   ) # Matrice des trajectoires de l'ASJ
    payoff_3m = K-S[16,]
    payoff_6m = K-S[92,]
    payoff_1y = K-S[364,]

    payoff_3m[payoff_3m<0] = 0
    payoff_6m[payoff_6m<0] = 0
    payoff_1y[payoff_1y<0] = 0
    
    put_3m = np.mean(payoff_3m)*np.exp(-rf*15/365)
    put_6m = np.mean(payoff_6m)*np.exp(-rf*91/365)
    put_1y = np.mean(payoff_1y)*np.exp(-rf)
    
    #Valeur des put sous BMS
    BS_3m = BS(S_t0,K,15/365,vol,rf,"p")
    BS_6m = BS(S_t0,K,91/365,vol,rf,"p")
    BS_1y = BS(S_t0,K,1,vol,rf,"p")
    
    return np.array([[put_3m, put_6m, put_1y], [BS_3m, BS_6m, BS_1y]]), S, Q.h,Q.ex_r, Q.z

#Fonction qui renvoie les statistiques descriptives du vecteur donnÃ©
def statdes(df):
    stats = df.describe()
    stats.loc['var'] = df.var().tolist()
    stats.loc['skew'] = df.skew().tolist()
    stats.loc['kurt'] = df.kurtosis().tolist()
    return stats

#Fonction qui renvoie la valeur du put europÃ©n ajustÃ©e selon une variable de controle
def putval_cont(S_t0,ndays,n_paths,vol,rf,K,c):
    
    #Fonction locale qui simule des trajectoires de rendement excÃ©dentaire sous BMS
    def BSsimulate(S_t0, n_days, n_paths, h_tp1, z=None):
        
        ex_r = np.full((n_days, n_paths), np.nan)
        h = np.full((n_days+1, n_paths), np.nan)
        h[0,:] = h_tp1 # because indices start at 0 in Python
        if z is None:
            z = antithetic_normal(n_days, n_paths)
        # Going from 1 day to n_days ahead
        #  t0+1      is tn==0 because indices start at 0 in Python
        #  t0+n_days is consequently tn==n_days-1
        for tn in range(0,n_days):
            # Simulate returns
            ex_r[tn,:] = -0.5*h_tp1 + np.sqrt(h_tp1)*z[tn,:]
            
        return ex_r
    
    #meme fonction que putval, mais qui renvoie uniquement la valeur du put selon la simulation NGARCH pour les trois maturitÃ©s
    def putval2(S_t0,ndays,n_paths,vol,rf,K):
    
        #ng.omega = ng.variance_targeting( (vol**2)*dt )
        h   = vol **2 / 365
        Q = measure( *ng.simulateQ(S_t0,ndays,n_paths,h) ) 
        S = S_t0*np.exp(np.cumsum(Q.ex_r + rf/365 ,axis = 0)   )
        payoff_3m = K-S[16,]
        payoff_6m = K-S[92,]
        payoff_1y = K-S[364,]

        payoff_3m[payoff_3m<0] = 0
        payoff_6m[payoff_6m<0] = 0
        payoff_1y[payoff_1y<0] = 0

        put_3m = np.mean(payoff_3m)*np.exp(-rf*15/365)
        put_6m = np.mean(payoff_6m)*np.exp(-rf*91/365)
        put_1y = np.mean(payoff_1y)*np.exp(-rf)


        return np.array([put_3m, put_6m, put_1y])
    
    #Fonction qui Ã©value le put europÃ©en par  simulation sous BMS
    def val(S_t0,ndays,n_paths,vol,rf,K):
    
        #ng.omega = ng.variance_targeting( (vol**2)*dt )
        h   = vol **2 / 365
        Q = BSsimulate(S_t0, n_days, n_paths, h) 
        S = S_t0*np.exp(np.cumsum(Q + rf/365 ,axis = 0)   )
        payoff_3m = K-S[16,]
        payoff_6m = K-S[92,]
        payoff_1y = K-S[364,]

        payoff_3m[payoff_3m<0] = 0
        payoff_6m[payoff_6m<0] = 0
        payoff_1y[payoff_1y<0] = 0

        put_3m = np.mean(payoff_3m)*np.exp(-rf*15/365)
        put_6m = np.mean(payoff_6m)*np.exp(-rf*91/365)
        put_1y = np.mean(payoff_1y)*np.exp(-rf)
        
        

        return np.array([put_3m, put_6m, put_1y])
    
    
    
    BS_3m = BS(S_t0,K,15/365,vol,rf,"p")
    BS_6m = BS(S_t0,K,91/365,vol,rf,"p")
    BS_1y = BS(S_t0,K,1,vol,rf,"p")
    BS_vec = np.array([BS_3m,BS_6m,BS_1y])
    
    
    adjusted_value = putval2(S_t0,ndays,n_paths,vol,rf,K) + c*( BS_vec - val(S_t0,ndays,n_paths,vol,rf,K))
    
    
    return  adjusted_value


    
