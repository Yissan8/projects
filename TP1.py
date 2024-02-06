




# On importe les librairies, on definit les fonctions et on initialise les paramÃ¨tres qu'on va utiliser dans le code : 

#Librairies : 
import numpy as np
from scipy.stats import norm
import opstrat as op
from scipy import optimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time as tm
import scipy.optimize as opt
import quadpy





# Fonctions : 
    # Fonction BS : calcul le prix d'une option call ou put avec t qui controle pour le type (call ou put)
def BS(S,K,T,vol,r,t):
    d1= (np.log(S/K)+T*(r + 0.5*vol**2))/(vol*np.sqrt(T))
    d2= d1 - vol*np.sqrt(T)
    if t =="c":
        p = (S*norm.cdf(d1) - K*norm.cdf(d2)*np.exp(-r*T))
    elif t == "p" :
        p = (K*norm.cdf(-d2)*np.exp(-r*T) - S*norm.cdf(-d1))
    return p
        
    # Fonction qui calcule la valatilitee implicite : 
def IV(S,K,T,r,marketoptionPrice):
    def bs_price(sigma):
        fx=BS(S,K,T,sigma,r,"p")-marketoptionPrice
        return fx
    return optimize.brentq(bs_price,0.0001,100,maxiter=1000)
###Utilisation de l'algorithme de root-finding de brent pour accÃ©lÃ©rer la convergence au vu de la forme de la fonction


    #Fonction de tarification du put en utilisant un arbre binomial : 
    # typ controle si c'est un put europeen ou americain (e ou a), aj controle avec ou sans ajustement de BS(yes or no) : 

def put_binomial(S,T,K,r,sigma,n,typ,aj):
    # Valeur de l'incrÃ©ment de temps
    delta=T/n
    # Calcul de u, d et q
    u=np.exp(sigma*delta**0.5)
    d=1/u
    q=(np.exp(r*delta)-d)/(u-d)
    qd = 1 - q
    
    if typ == 'e' and aj == 'no':
        # Initialisation de la somme
        somme=0
        # Initialisation de la combinaison et de l'incrÃ©ment servant Ã  son calcul
        c=1
        increment_c=0
        # Calcul de la somme par itÃ©ration
        for j in range(n+1):
            somme=somme+max(K-S*u**j*d**(n-j),0)*c*q**j*(1-q)**(n-j)
            # Mise Ã  jour de la combinaison
            c=c*(n-increment_c)/(increment_c+1)
            increment_c=increment_c+1
        # Calcul de la valeur du put
        put=np.exp(-r*T)*somme
    elif typ == 'e' and aj == 'yes' :
        somme=0
        # Initialisation de la combinaison et de l'incrÃ©ment servant Ã  son calcul
        c=1
        increment_c=0
        # Calcul de la somme par itÃ©ration
        for j in range(n):
            somme=somme+BS(S*u**j*d**(n-j),K,delta,sigma,r,'p')*c*q**j*(1-q)**(n-j)
            # Mise Ã  jour de la combinaison
            c=c*(n-increment_c)/(increment_c+1)
            increment_c=increment_c+1
        # Calcul de la valeur du put
        put=np.exp(-r*T)*somme
    elif typ == 'a' and aj == 'no' : 
        Terminal_Value = np.zeros(n+1)
        Middle_Value = np.zeros(n+1)
        for j in range(n+1):
            Terminal_Value[j]= max(K-S*(u**j)*(d**(n-j)),0)
        
        for time in reversed(range(n)): 
            np.resize(Middle_Value,time)
            for state in range(time+1):
                Middle_Value[state] = max(K - S*u**state*d**(time-state),qd*Terminal_Value[state] + q*Terminal_Value[state+1])
            np.resize(Terminal_Value,time)
            
            for state in range(time+1):
                Terminal_Value[state] = Middle_Value[state]
        put = Terminal_Value[0]
    elif typ == 'a' and aj == 'yes' :
        Terminal_Value = np.zeros(n)
        Middle_Value = np.zeros(n)
        for j in range(n):
            Terminal_Value[j]= BS(S*u**j*d**(n-1-j),K,delta,sigma,r,'p')
        
        for time in reversed(range(n-1)): 
            np.resize(Middle_Value,time)
            for state in range(time+1):
                Middle_Value[state] = max(K - S*u**state*d**(time-state),qd*Terminal_Value[state] + q*Terminal_Value[state+1])
            np.resize(Terminal_Value,time)
            
            for state in range(time+1):
                Terminal_Value[state] = Middle_Value[state]
        put = Terminal_Value[0]
    else:
        put = 're-enter appropriate parameters'
               
    return put

##Fonction que nous optimiserons a la Question 5 afin de trouver la frontiÃ¨re d'exercice. Sa rÃ©solution pour S nous donne le pix optimal d'exercice anticipÃ© pour une maturitÃ© rÃ©siduelle T donnÃ©e
def B(S,T,K,r,sigma,n,typ,aj):
    return put_binomial(S,T,K,r,sigma,n,typ,aj) + S -K

# Fonction qui calcule la moneyness :
def moneyness(S,K,T,vol,r):
    return 0.5- norm.cdf((np.log(S/K)+T*(r + 0.5*vol**2))/(vol*np.sqrt(T)))

# Fonctions qui calculent la frontiÃ¨re d'exercice avec la forme quasi-analytique de Carr-Jarrow-Myneni.

def m(r,sigma,tau):
    def mfunc(m):
        def erf(x):
            return 2*norm.cdf(x*np.sqrt(x))-1
        def beta(tau):
            return 0.75*sigma-0.5*r/sigma + 0.5*m/(2*np.sqrt(tau))
        
        return sigma*np.exp(-(r +0.5*sigma**2)*tau-sigma*np.sqrt(tau)*m)*norm.cdf(-m) -r*(erf(np.sqrt(r+0.5*beta(tau)**2)*tau))/(np.sqrt(2*r+beta(tau)**2))
    return optimize.brentq(mfunc,0.0001,100,maxiter=1000)    

def Bt(K,r,sigma,tau):
    return K*np.exp(-(r+0.5*sigma**2)*tau -sigma*np.sqrt(tau)*m(r,sigma,tau))





# ParamÃ¨tres du TP :
S_TP = 100
T_TP = 30/365
r_TP= 0.03
vol_hist = 0.35
K = np.array(range(90,115,5))
marketprices = [2.1809,2.5394,4.1029,6.7978,11.0215]


# Reponses aux questions : 

# Question 1 : Calcul de la volatilitÃ© implicite et de la moneyness 
# 
# Afin de calculer la moneyness nous utilisons la fonction dÃ©finie dans le TP : 0.5-N(d1)
# En ce qui concerne la volatilitÃ© implicite nous la bootstrap en retrouvant la racine de la fonction (sigma) : 
# c'est a dire f(sigma) = BS(S,K,T,sigma,r,"p") - prix de marchÃ© de l'option = 0 
# Afin d'optimiser le temps de calcul on utilise, au vu de la forme de la fonction, la mÃ©thode de Brent afin de retrouver le zero de la fonction. 




IV_v = np.empty([5])

##Boucle qui Bootstrap les Vol Implicites avec notre fonction IV pour chacun des K demandÃ©s 
for i in range(0,5): 
    IV_v[i] = IV(S_TP,K[i],T_TP, r_TP, marketprices[i])
Moneyness_vector = [moneyness(S_TP,i,T_TP,vol_hist,r_TP) for i in K]

# Plot du smile de la volatilitÃ© implicite en fonction de la moneyness : 
plt.plot(Moneyness_vector,IV_v)
for i in range(0,5):
    plt.scatter(Moneyness_vector[i],IV_v[i],label = 'K = ' + str(K[i]))
plt.legend()
plt.title('ACG Corp options Volatility smile')
plt.xlabel('Put Options Moneyness')
plt.ylabel('Implied Volatility in %')



# Question 2 : Calcul de la volatilitÃ© implicite en utilisant une interpolation linÃ©aire des puts ayant K = 92.50, 97.50, 102.50 et 107.50.

# Nous utilisons la fonction interp1d du package scipy qui permet de faire de l'interpolation linÃ©aire. Ce cas est nÃ©anmoins assez simple car les K se retrouvent au milieu de 2 points donc ce n'est que 0.5 * volatilitÃ© implicite1 + 0.5 * volatilitÃ© implicite2. 




interp = interp1d(K,IV_v) ###estimation de la fonction d'interpolation avec les K et VolatilitÃ©s implicites dÃ©ja connus
Vol_Imp = [interp(i) for i in [92.5,97.5,102.5,107.5]] ##Interpolation de la Vol Implicite pour les 4 puts inconnus

print('Put option Strike price','     Interpolated Implied Volatility')
[print(K[i], '                       :        ', Vol_Imp[i], '\n') for i in range(4)]



# Question 3 : GÃ©neration des graphiques de convergence des puts : K = 90, 95, 97.50 ,100, 105 ð‘’ð‘¡ 110. 


IV6 = interp(97.5)

strikes = [90,95,97.5,100,105,110]
put_prices = [[], [], [], [], [], []]
put_BSprices = [0,0,0,0,0,0]
IV_vector = [IV_v[0],IV_v[1],IV6,IV_v[2],IV_v[3],IV_v[4]]
erreur = []
vec_pas = range(10,501) ##Vecteur contenant les nombres de pas

##Boucle pour calculer/stocker les prix CRR et black-scholes des 6 puts europÃ©ens en fonction du nombre de pas 
j = 0
for i in strikes:
    put_prices[j] = [put_binomial(S_TP, T_TP, i, r_TP, IV_vector[j], n,'e','no') for n in vec_pas]
    put_BSprices[j] = BS(S_TP,i,T_TP,IV_vector[j],r_TP,"p")
    j = j + 1

erreur = [100*(put_prices[n] - put_BSprices[n])/put_BSprices[n] for n in range(6)]

fig,ax= plt.subplots(3,2,figsize=(14,14))

ax0,ax1,ax2,ax3,ax4,ax5=ax.flatten()

ax0.plot(vec_pas,erreur[0],'r')
ax0.plot(vec_pas,np.zeros(491),'-')
ax0.plot(vec_pas,0.1*np.ones(491),':')
ax0.plot(vec_pas,-0.1*np.ones(491),':')
ax0.set_ylabel("Erreur relative en %")
ax0.set_title("K=90")

fig.suptitle('Graphes de convergence', fontsize=16)

ax1.plot(vec_pas,erreur[1],'r')
ax1.plot(vec_pas,np.zeros(491),'-')
ax1.plot(vec_pas,0.1*np.ones(491),':')
ax1.plot(vec_pas,-0.1*np.ones(491),':')
ax1.set_title("K=95")

ax2.plot(vec_pas,erreur[2],'r')
ax2.plot(vec_pas,np.zeros(491),'-')
ax2.plot(vec_pas,0.1*np.ones(491),':')
ax2.plot(vec_pas,-0.1*np.ones(491),':')
ax2.set_ylabel("Erreur relative en %")
ax2.set_title("K=97.5")

ax3.plot(vec_pas,erreur[3],'r')
ax3.plot(vec_pas,np.zeros(491),'-')
ax3.plot(vec_pas,0.1*np.ones(491),':')
ax3.plot(vec_pas,-0.1*np.ones(491),':')
ax3.set_title("K=100")

ax4.plot(vec_pas,erreur[4],'r')
ax4.plot(vec_pas,np.zeros(491),'-')
ax4.plot(vec_pas,0.1*np.ones(491),':')
ax4.plot(vec_pas,-0.1*np.ones(491),':')
ax4.set_xlabel("Nombre de pas")
ax4.set_ylabel("Erreur relative en %")
ax4.set_title("K=105")

ax5.plot(vec_pas,erreur[5],'r')
ax5.plot(vec_pas,np.zeros(491),'-')
ax5.plot(vec_pas,0.1*np.ones(491),':')
ax5.plot(vec_pas,-0.1*np.ones(491),':')
ax5.set_xlabel("Nombre de pas")
ax5.set_title("K=110")
plt.show()


# Question 4 : 

#     Partie 1 : Calcul prix CRR put europeen 97,5 avec et sans ajustement : 




put_a = put_binomial(S_TP, T_TP, 97.5,r_TP, IV6, 500, 'e', 'no')
put_b = put_binomial(S_TP, T_TP, 97.5,r_TP, IV6, 500, 'e', 'yes')

print('prix CRR du put K = 97.5 sans ajustement BMS = ', put_a,'\nprix CRR du put K = 97.5 avec ajustement BMS =', put_b)



#     Partie 2 : GÃ©nerer les graphiques demandÃ©s : 




##Prix BS put europÃ©en
BS(S_TP,97.5,T_TP,IV6,r_TP,"p")





put975, put975_AJ, tems_calcul, temps_calcul_AJ = np.empty([491]),np.empty([491]),np.empty([491]),np.empty([491])

###Boucle pour calculer et stocker le temps de calcul des prix puts europÃ©ens par CRR ajustÃ© et non-ajustÃ©
for n in vec_pas : 
    start = tm.time()
    put975[n-10] = put_binomial(S_TP, T_TP, 97.5,r_TP, IV6, n, 'e', 'no')
    tems_calcul[n-10]= tm.time() - start
    start = tm.time()
    put975_AJ[n-10] = put_binomial(S_TP, T_TP, 97.5,r_TP, IV6, n, 'e', 'yes')
    temps_calcul_AJ[n-10] = tm.time() - start

##Calcul du prix du put americain avec CRR avec ajustement    
am_put975 = [put_binomial(S_TP, T_TP, 97.5,r_TP, IV6, n, 'a', 'yes') for n in vec_pas]

fig,ax= plt.subplots(3,1,figsize=(14,14))
ax0,ax1,ax2=ax.flatten()

ax0.plot(vec_pas,put975,vec_pas,put975_AJ,'r')
ax0.legend(["Sans ajustement","Avec ajustement"])
ax0.set_xlabel("Nombre de pas")
ax0.set_ylabel("CRR price")
ax0.set_title("Prix CRR put euro 97.5")

ax1.plot(vec_pas,tems_calcul,vec_pas,temps_calcul_AJ,'r')
ax1.legend(["Sans ajustement","Avec ajustement"])
ax1.set_xlabel("Nombre de pas")
ax1.set_ylabel("Temps en secondes")
ax1.set_title("Temps de calcul en fonction du nombre de pas")

ax2.plot(vec_pas,put975_AJ,vec_pas,am_put975,'r')
ax2.legend(["Europeen","Americain"])
ax2.set_xlabel("Nombre de pas")
ax2.set_ylabel("CRR price")
ax2.set_title("Prix CRR avec ajustement : put EU et US")
plt.tight_layout()
plt.show()


# Question 5 : FrontiÃ¨re d'exercice (Notez que notre fonction prend environ 3 heures a rouler sur Spyder), nous avons mis l'output en image. 

matres = T_TP - np.arange(0,T_TP,T_TP/500) # vecteur de maturitÃ©s rÃ©siduelles ; diminuer le nombre en changeant la valeur 500  pour tester le code dans un temps raisonnable
vec_K = [92.5,97.5,102.5,107.5]
of = [[], [], [], []] # Vecteur qui contiendra les la frontiere d'exercice pour 
of40 = [[], [], [], []]

##BOUCLE QUI RESOUD L'EQUATION MENTIONNEE PLUS HAUT POUR RETROUVER Bt a chacune des 500 maturitÃ©s rÃ©siduelles, avec un arbre binomial a 500 pas
j = 0
for i in vec_K:
    of[j] = [opt.fsolve(B, 100,args=(tau, i,0.03, 0.35, 500, 'a', 'yes')) for tau in matres] ##changer le nombre de pas de 500 a 50 pour tester lecode dans une durÃ©e raisonable 
    of40[j] = [opt.fsolve(B, 100,args=(tau, i,0.03, 0.4, 500, 'a', 'yes')) for tau in matres]##changer le nombre de pas de 500 a 50 pour tester lecode dans une durÃ©e raisonable 
    j = j+1


fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(14,14))
ax0,ax1,ax2,ax3=ax.flatten()

ax0.plot(matres,of[0],matres,of40[0],'r')
ax0.legend(["vol = 35%","vol = 40%"])
ax0.set_xlabel("maturitÃ© rÃ©siduelle")
ax0.set_ylabel("Prix optimal d'exercice")
ax0.set_title("K = 92.5")



ax1.plot(matres,of[1],matres,of40[1],'r')
ax1.legend(["vol = 35%","vol = 40%"])
ax1.set_xlabel("maturitÃ© rÃ©siduelle")
ax1.set_ylabel("Prix optimal d'exercice")
ax1.set_title("K = 97.5")

ax2.plot(matres,of[2],matres,of40[2],'r')
ax2.legend(["vol = 35%","vol = 40%"])
ax2.set_xlabel("maturitÃ© rÃ©siduelle")
ax2.set_ylabel("Prix optimal d'exercice")
ax2.set_title("K = 102.5")


ax3.plot(matres,of[3],matres,of40[3],'r')
ax3.legend(["vol = 35%","vol = 40%"])
ax3.set_xlabel("maturitÃ© rÃ©siduelle")
ax3.set_ylabel("Prix optimal d'exercice")
ax3.set_title("K = 107.5")
fig.tight_layout()
plt.show()


# Question 6 : FrontiÃ¨re d'exercice en utilisant la forme quasi-analytique de Carr-Jarrow-Myneni



matres = T_TP - np.arange(0,T_TP,T_TP/500)
vec_K = [92.5,97.5,102.5,107.5]
of = [[], [], [], []]
of40 = [[], [], [], []]


op_frontier = [[], [], [], []]
op_frontier40 = [[], [], [], []]

j =0
for i in vec_K:
    op_frontier[j]= [Bt(i,r_TP,vol_hist,tau) for tau in matres]
    op_frontier40[j]= [Bt(i,r_TP,0.4,tau) for tau in matres]
    j = j+1


fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(14,14))
ax0,ax1,ax2,ax3=ax.flatten()
fig.tight_layout()

ax0.plot(matres,op_frontier[0],matres,op_frontier40[0],'r')
ax0.legend(["vol = 35%","vol = 40%"])
ax0.set_xlabel("maturitÃ© rÃ©siduelle")
ax0.set_ylabel("Prix optimal d'exercice")
ax0.set_title("K = 92.5")



ax1.plot(matres,op_frontier[1],matres,op_frontier40[1],'r')
ax1.legend(["vol = 35%","vol = 40%"])
ax1.set_xlabel("maturitÃ© rÃ©siduelle")
ax1.set_ylabel("Prix optimal d'exercice")
ax1.set_title("K = 97.5")

ax2.plot(matres,op_frontier[2],matres,op_frontier40[2],'r')
ax2.legend(["vol = 35%","vol = 40%"])
ax2.set_xlabel("maturitÃ© rÃ©siduelle")
ax2.set_ylabel("Prix optimal d'exercice")
ax2.set_title("K = 102.5")


ax3.plot(matres,op_frontier[3],matres,op_frontier40[3],'r')
ax3.legend(["vol = 35%","vol = 40%"])
ax3.set_xlabel("maturitÃ© rÃ©siduelle")
ax3.set_ylabel("Prix optimal d'exercice")
ax3.set_title("K = 107.5")
plt.tight_layout()
plt.show()
