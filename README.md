# SchrödArt 🌌

**Résolution numérique de l'équation de Schrödinger & génération d'art quantique**

Application interactive Streamlit qui transforme les solutions de l'équation de Schrödinger en visualisations scientifiques et en images génératives artistiques.

> Projet informatique individuel — ENSC Bordeaux INP 
> Auteure : **SAAB Alyaa**

---

## Aperçu

SchrödArt implémente deux méthodes numériques indépendantes pour résoudre l'équation de Schrödinger, les confronte, puis exploite les spectres propres obtenus pour générer cinq styles d'images artistiques (rosace, nébuleuse, cristal, mandala, galaxie).

$$i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m}\nabla^2\psi + V(x,t)\psi$$

---

## Fonctionnalités

L'application est organisée en quatre onglets accessibles depuis la barre latérale :

| Onglet | Description |
|--------|-------------|
| **Régime stationnaire 2D** | Diagonalisation du hamiltonien 2D — visualisation des modes propres et du potentiel gaussien |
| **Régime dépendant du temps 1D** | Évolution temporelle par décomposition modale — densité de probabilité et surface 3D |
| **Art quantique** | Génération d'images à partir des valeurs propres (5 styles artistiques) |
| **Split-Step Fourier** | Simulation directe de ψ par la méthode de Strang — 3 scénarios physiques |

---

## Méthodes numériques

### Méthode 1 — Décomposition modale (`simulation.py`)

Le hamiltonien 2D est construit avec un produit de Kronecker sur la matrice de différences finies :

$$H = -\frac{1}{2}(D_x \otimes I + I \otimes D_y) + V$$

La diagonalisation creuse (ARPACK via `scipy.sparse.linalg.eigsh`) extrait les $k$ états propres de plus basse énergie. L'évolution temporelle est ensuite calculée analytiquement :

$$\psi(x, t) = \sum_j c_j \psi_j(x)\, e^{-iE_j t}$$

La norme est conservée exactement puisque $|e^{-iE_j t}| = 1$.

### Méthode 2 — Split-Step Fourier (`Fourier.py`)

Intégration directe de l'équation pas à pas, par le schéma de Strang (ordre 2) :

$$\psi(x, t+\Delta t) \approx e^{-iV\Delta t/2} \cdot \mathcal{F}^{-1}\!\left[e^{-ik^2\Delta t}\,\mathcal{F}[\cdot]\right] \cdot e^{-iV\Delta t/2}\,\psi(x,t)$$

Trois scénarios sont disponibles : **effet tunnel** (barrière), **oscillateur harmonique**, **double puits**.

---

## Art quantique

Chaque spectre propre $\{E_j\}$ est une signature unique du système simulé. Cinq générateurs l'exploitent comme paramètre de fonctions mathématiques complexes :

- **Rosace** — ondes stationnaires en coordonnées polaires
- **Nébuleuse** — anneaux gaussiens et filaments d'ondes planes
- **Cristal** — symétries $n$-aires ($n = 3$ à $10$) atténuées exponentiellement
- **Mandala fractal** — symétries paires entrelacées (ordres 4, 6, 8, 10, 12, 14)
- **Galaxie spirale** — bras logarithmiques + halo central + étoiles ponctuelles

Le rendu de base code la phase $\phi = \arg(\psi)$ sur la teinte HSV et l'amplitude $|\psi|^2$ sur la valeur :

$$\text{Image}(x,y) = \text{HSV}\!\left(\frac{\phi + \pi}{2\pi},\; 1,\; \frac{|\psi|^2}{\max|\psi|^2}\right)$$

---

## Installation

```bash
git clone https://github.com/<votre-username>/schrodart.git
cd schrodart
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Dépendances

| Bibliothèque | Rôle |
|---|---|
| `streamlit >= 1.30` | Interface web interactive |
| `numpy >= 1.24` | Calcul vectoriel et FFT |
| `scipy >= 1.10` | Algèbre linéaire creuse, diagonalisation |
| `matplotlib >= 3.7` | Tracés scientifiques |
| `pillow >= 9.0` | Génération et export GIF |

---

## Structure du projet

```
schrodart/
├── streamlit_app.py   # Interface principale Streamlit
├── simulation.py      # Méthode modale (stationnaire 2D + temporel 1D)
├── Fourier.py         # Méthode Split-Step Fourier
├── visualisation.py   # Rendus artistiques (5 styles)
└── requirements.txt   # Dépendances Python
```

---

## Démo en ligne

🔗 [Accéder à l'application sur Streamlit Cloud](#) *[(https://gnm4pxwnrpb6cy3syst6sn.streamlit.app)]*

---

## Références

- Griffiths, D.J. *Introduction to Quantum Mechanics*, 3e éd., Cambridge University Press, 2018.
- Tannor, D.J. *Introduction to Quantum Mechanics: A Time-Dependent Perspective*, University Science Books, 2007.
- Strang, G. *On the construction and comparison of difference schemes*, SIAM J. Numer. Anal., 1968.
- [Documentation SciPy — `eigsh`](https://docs.scipy.org/doc/scipy/)
- [Documentation Streamlit](https://docs.streamlit.io)

---
