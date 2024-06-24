"""
This is a Monte Carlo simulation to calculate the distribution
of neutrons in nuclear waste pool.
This file simulates the movement of "all" the neutrons in the pool.
By changing the parameters (mu, lambda, pool size, depth etc) we can predict several
different regimes and scenarios.
Other files run this algorithm many times to analyze different kinds of distributions and
to visualize results.
Written as a final project for Modelska analiza (FMF Uni Lj) in 2022.
"""


from numba import njit, jit
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import time

# konstante:
mu_bor = 1
mu_voda = 1
mu = ((mu_bor + mu_voda)/2)/1   # vecji mu - daljsi R (*1,2,5)
lamb = 1/mu     # sipanje
mi_abs = 0.1      # absorpcija - manjse stevilo = manj umiranja
bazen_x, bazen_y, bazen_z = 30, 30, 25
elem_z = 20  # razlika!
r_element = 0.2
N_element_1d = 10
N_elementov = N_element_1d ** 2
vmesna_razd = (bazen_x - (N_element_1d * r_element)) / N_element_1d
vmesna_razd_z = 0.1


# konstrukcija mreze v bazenu:

#@njit
def konstrukcija():
    x = 0
    seznam_sredisc = []
    while x < bazen_x:
        y = 0
        while y < bazen_y:
            z = 0
            while z < elem_z:
                seznam_sredisc.append(np.array([x, y, z]))
                z += vmesna_razd_z
            y += vmesna_razd
        x += vmesna_razd
    print("Mreza skonstruirana")
    return np.array(seznam_sredisc)


#@njit
def poteza(koord):  # tu je samo sipanje
    x, y, z = koord
    # sipanje:
    r = -1 / lamb * np.log(np.random.random(1))  # r je porazdeljen exponentno!
    fi = np.random.random(1) * 2 * np.pi
    theta = np.random.random(1) * np.pi - np.pi / 2
    dx = (r * np.cos(fi) * np.cos(theta))[0]
    dy = (r * np.sin(fi))[0]
    dz = (r * np.cos(fi) * np.sin(theta))[0]
    return np.array([x+dx, y+dy, z+dz])


def ena_generacija(seznam, N):
    seznam_zunanji = np.array([[bazen_x, bazen_y, bazen_z]])  # [[30, 30, 25]] ??? narobe

    for n in range(N):  # tu delamo poteze (N)
        stevilo_zivih = len(seznam)
        # ABSORPCIJA = izumiranje
        dN = np.random.poisson(lam=(mi_abs*stevilo_zivih))  # nakljucno poissonsko izumiranje/absorpcija
        stevilo_zivih -= dN
        if stevilo_zivih <= 0:
            print("vsi umrli")
            break
        rng = np.random.default_rng()  # treba ustvarit nov generator naklj stevil
        lista_za_odstel = rng.choice(stevilo_zivih+dN, size=dN, replace=False)  #  izberemo dN nakljucnih pozicij in jih odstranimo s seznama
        seznam = np.delete(seznam, lista_za_odstel, 0)  # koncnen NOV seznam po absorpciji -> sledijo še premiki

        # PREMIKI
        seznam_nov = []
        for element in seznam:  # seznam z ze upostevano absorpcijo
            element = poteza(element)  # np array, dobimo [x+dx, y+dy, z+dz]
            if element[2] >= bazen_z:
                seznam_zunanji = np.append(seznam_zunanji, [element], axis=0)  # ce kaksen utece ven, "zamrzne" na novem seznamu
            else:
                seznam_nov.append(element)  # ostali grejo na drug seznam, brez ubeznikov
        seznam = seznam_nov
        # gremo v novo potezo
    seznam = np.append(seznam, seznam_zunanji, axis=0)  # zdruzimo seznam zivih z vsemi nad gladino
    return seznam


seznam = (konstrukcija())
seznam_orig = seznam    # za plot kasneje

#  zanka z N generacijami:
stevilo_generacij = 1
stevilo_potez = 60
seznam0 = seznam  # zacetni seznam je isti za vse generacije
seznam = np.array([[0, 0, 0]])  # tu bo koncen seznam
for i in range(stevilo_generacij):  #ponovimo za N generacij
    print("generacija",i)
    seznam1 = ena_generacija(seznam0, stevilo_potez)
    seznam = np.append(seznam, seznam1, axis=0)  # SEDAJ JE PRVA GENERACIJA PODVOJENA?
print(time.process_time())


# PLOT ZADNJIH LOKACIJ ENE GENERACIJE

xo, yo, zo = np.transpose(seznam_orig)[0], np.transpose(seznam_orig)[1], np.transpose(seznam_orig)[2]
xi, yi, zi = np.transpose(seznam)[0], np.transpose(seznam)[1], np.transpose(seznam)[2]

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])
ax.scatter(xi, yi, zi, c="C1", s=12)
ax.scatter(xo, yo, zo, c="C2", s=2, alpha=0.2)

faktor = 1.1  # koliko se gladina raztegne
X = np.arange(-faktor*bazen_x+bazen_x, faktor*bazen_x, 0.25)  # gladina
Y = np.arange(-faktor*bazen_y+bazen_y, faktor*bazen_y, 0.25)
Z = bazen_z
Z, Y = np.meshgrid(Z, Y)
ax.plot_surface(X, Y, Z, alpha=0.5)

plt.suptitle(f"Bazen z izrabljenimi gorivnimi elementi")
#plt.title("st. generacij: %s, st. potez: %s, $\mu$: %s, $\lambda$: %s"%(stevilo_generacij, stevilo_potez, mi_abs, lamb))
plt.xlabel("$x$")
plt.ylabel("$y$")
ax.set_zlabel("$z$")
plt.minorticks_on()
plt.grid(linestyle=":")
plt.tight_layout()

plt.show()

