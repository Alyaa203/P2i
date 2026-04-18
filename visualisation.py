import numpy as np
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter


def rendu_artistique(psi):
    # Extraction des grandeurs physiques
    rho = np.abs(psi) ** 2      # densité de probabilité
    phi = np.angle(psi)         # phase complexe

    # Normalisation de la densité (pour affichage stable)
    rho_max = rho.max()
    if rho_max > 0:
        rho = rho / rho_max

    # Encodage visuel : phase → teinte, densité → luminosité
    h = (phi + np.pi) / (2 * np.pi)  # phase ∈ [-π, π] → [0,1]
    s = np.ones_like(h)              # saturation maximale
    v = rho                          # intensité = densité

    hsv = np.stack((h, s, v), axis=-1)
    rvb = colors.hsv_to_rgb(hsv)

    return rho, phi, rvb


def normaliser_champ(z):
    # Normalisation affine → [0,1]
    z = z - np.min(z)
    m = np.max(z)
    if m > 0:
        z = z / m
    return z


def _normaliser_valeurs_propres(valeurs_propres):
    # Mise à l’échelle des énergies pour piloter les motifs
    vp = np.array(valeurs_propres, dtype=float)
    vp = vp - vp.min()
    if vp.max() > 0:
        vp = vp / vp.max()
    return vp


def generer_rosace(valeurs_propres, taille=700):
    # Passage en coordonnées polaires
    y, x = np.mgrid[-1:1:taille*1j, -1:1:taille*1j]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    img = np.zeros_like(r)
    vp = _normaliser_valeurs_propres(valeurs_propres)

    # Superposition de modes oscillatoires radiaux + angulaires
    for i, e in enumerate(vp):
        freq = 3 + i
        img += np.sin((8 + 25 * e) * np.pi * r + freq * theta) ** 2
        img += 0.5 * np.cos((5 + 20 * e) * theta - 10 * r)

    img = normaliser_champ(img)

    # Mapping couleur simple (RGB non linéaire)
    rvb = np.zeros((taille, taille, 3))
    rvb[..., 0] = img
    rvb[..., 1] = gaussian_filter(1 - img, sigma=2)
    rvb[..., 2] = np.sqrt(img)

    return np.clip(rvb, 0, 1)


def generer_cristal(valeurs_propres, taille=700):
    # Domaine élargi pour effet cristallin
    y, x = np.mgrid[-1.5:1.5:taille*1j, -1.5:1.5:taille*1j]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    vp = _normaliser_valeurs_propres(valeurs_propres)
    img = np.zeros_like(r)
    mask = r < 1.4  # découpe circulaire

    # Motifs amortis radialement → effet structure solide
    for i, e in enumerate(vp):
        n = 3 + (i % 8)
        freq = 4 + 12 * e
        img += np.cos(n * theta + freq * r) * np.exp(-r * (0.8 + e))
        img += 0.4 * np.sin((i + 2) * np.pi * r**2 + (i % 5) * theta)

    img *= mask
    img = gaussian_filter(img, sigma=taille / 400)
    img = normaliser_champ(img)

    # Palette non linéaire (accent sur contrastes)
    rvb = np.zeros((taille, taille, 3))
    rvb[..., 0] = np.clip(0.15 * img + 0.7 * img**3, 0, 1)
    rvb[..., 1] = np.clip(0.6 * img**1.5 + 0.3 * gaussian_filter(img, sigma=taille / 200), 0, 1)
    rvb[..., 2] = np.clip(0.85 * np.sqrt(img) + 0.2 * (1 - img)**2, 0, 1)

    # Surbrillance des maxima → effet "facettes"
    sparkle = np.clip((img - 0.82) * 6, 0, 1)
    rvb += sparkle[..., None]

    return np.clip(rvb, 0, 1)


def generer_mandala(valeurs_propres, taille=700):
    # Géométrie pure : symétries angulaires discrètes
    y, x = np.mgrid[-1:1:taille*1j, -1:1:taille*1j]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    vp = _normaliser_valeurs_propres(valeurs_propres)
    img = np.zeros_like(r)

    for i, e in enumerate(vp):
        n = 4 + (i % 6) * 2   # symétries paires
        k = 6 + int(20 * e)
        phase = i * np.pi / (len(vp) + 1)

        petale = np.cos(n * theta + phase) ** 2
        anneau = np.exp(-((r - 0.12 * (i % 6 + 1)) ** 2) / 0.004)
        radial = np.sin(k * r + phase) ** 2

        img += petale * (anneau + 0.3 * radial)

    img = gaussian_filter(img, sigma=taille / 500)
    img = normaliser_champ(img)

    # Palette sombre → or → clair
    rvb = np.zeros((taille, taille, 3))
    rvb[..., 0] = img ** 0.6
    rvb[..., 1] = 0.75 * img ** 0.8
    rvb[..., 2] = 0.15 * img ** 1.5

    # Accentuation des crêtes
    crete = np.clip((img - 0.75) * 4, 0, 1)
    rvb += crete[..., None] * np.array([0.4, 0.35, 0.05])

    return np.clip(rvb, 0, 1)


def generer_galaxie(valeurs_propres, taille=700):
    # Structure spirale (logarithmique approximée)
    y, x = np.mgrid[-1.3:1.3:taille*1j, -1.3:1.3:taille*1j]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    vp = _normaliser_valeurs_propres(valeurs_propres)
    spirale = np.zeros_like(r)

    for i, e in enumerate(vp):
        n_bras = 2 + (i % 3)
        pitch = 1.8 + 2.5 * e

        bras = np.cos(n_bras * theta - pitch * np.log1p(r * 4) + i * 0.4) ** 2
        enveloppe = np.exp(-r ** 2 / (0.18 + 0.25 * e))

        spirale += bras * enveloppe

    spirale = gaussian_filter(spirale, sigma=taille / 300)
    spirale = normaliser_champ(spirale)

    # Noyau + bruit structuré
    noyau = np.exp(-8 * r ** 2)
    poussiere = normaliser_champ(
        np.sin(18 * theta + 22 * r) * np.exp(-1.5 * r) +
        np.cos(11 * theta - 15 * r) * np.exp(-2 * r)
    )

    rvb = np.zeros((taille, taille, 3))
    rvb[..., 0] = 0.85 * spirale + 0.6 * noyau
    rvb[..., 1] = 0.5 * spirale ** 1.2 + 0.3 * poussiere + 0.4 * noyau
    rvb[..., 2] = 0.3 + 0.6 * gaussian_filter(spirale, sigma=taille / 180) + 0.5 * noyau

    # Étoiles ponctuelles (processus aléatoire rare)
    rng = np.random.default_rng(seed=42)
    masque_etoile = rng.random((taille, taille)) > 0.997
    rvb[masque_etoile] = 1.0

    return np.clip(rvb, 0, 1)


def generer_nebuleuse(valeurs_propres, taille=700):
    # Motifs diffus + turbulence
    y, x = np.mgrid[-1.2:1.2:taille*1j, -1.2:1.2:taille*1j]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    img = np.zeros_like(r)
    vp = _normaliser_valeurs_propres(valeurs_propres)

    for i, e in enumerate(vp):
        freq = 2 + (i % 7)
        phase = (i + 1) * 0.35 + 3.0 * e

        img += np.exp(-((r - 0.18 * (i % 5 + 1)) ** 2) / (0.012 + 0.03 * e))
        img += 0.8 * np.cos(10 * r - freq * theta + phase)
        img += 0.6 * np.sin((6 + 10 * e) * x + (4 + freq) * y + phase)

    img = gaussian_filter(img, sigma=taille / 250)
    img = normaliser_champ(img)

    halo = np.exp(-2.8 * r**2)
    filament = normaliser_champ(
        np.sin(12 * theta + 18 * r) +
        np.cos(7 * theta - 14 * r)
    )

    rvb = np.zeros((taille, taille, 3))
    rvb[..., 0] = 0.95 * img + 0.35 * halo
    rvb[..., 1] = 0.45 * filament + 0.55 * np.sqrt(img)
    rvb[..., 2] = 0.25 + 0.95 * gaussian_filter(1 - img, sigma=taille / 300)

    return np.clip(rvb, 0, 1)


