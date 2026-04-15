import numpy as np
import matplotlib.pyplot as plt

_trapz = np.trapz if hasattr(np, 'trapz') else np.trapezoid


def construire_grille(x_min, x_max, N):
    x  = np.linspace(x_min, x_max, N, endpoint=False)
    dx = x[1] - x[0]
    L  = x_max - x_min
    k  = np.fft.fftfreq(N, d=1.0 / N) * (2 * np.pi / L)
    return x, dx, k


def paquet_onde_gaussien(x, x0, sigma, k0):
    psi = np.exp(-((x - x0)**2) / (4 * sigma**2)) * np.exp(1j * k0 * x)
    norme = np.sqrt(_trapz(np.abs(psi)**2, x))
    return psi / norme


def potentiel_barriere(x, x_centre, largeur, hauteur):
    V = np.zeros_like(x)
    masque = np.abs(x - x_centre) < largeur / 2
    V[masque] = hauteur
    return V


def potentiel_harmonique(x, omega=1.0):
    return 0.5 * omega**2 * x**2


def potentiel_double_puits(x, a=2.0, b=0.5):
    return -a * x**2 + b * x**4


def pas_split_step(psi, V, k, dt):
    psi   = np.exp(-1j * V * dt / 2) * psi
    psi_k = np.fft.fft(psi)
    psi_k = np.exp(-1j * k**2 * dt) * psi_k
    psi   = np.fft.ifft(psi_k)
    psi   = np.exp(-1j * V * dt / 2) * psi
    return psi


def resoudre(psi0, V, k, dt, N_steps, save_every=1):
    psi          = psi0.copy()
    psi_histoire = [psi.copy()]
    t_histoire   = [0.0]
    for n in range(N_steps):
        psi = pas_split_step(psi, V, k, dt)
        if (n + 1) % save_every == 0:
            psi_histoire.append(psi.copy())
            t_histoire.append((n + 1) * dt)
    return np.array(psi_histoire), np.array(t_histoire)


def norme(psi, dx):
    return _trapz(np.abs(psi)**2, dx=dx).real


def energie(psi, V, k, dx):
    psi_k = np.fft.fft(psi) / len(psi)
    E_cin = np.sum(k**2 * np.abs(psi_k)**2) * dx * len(psi)
    E_pot = _trapz(V * np.abs(psi)**2, dx=dx).real
    return (E_cin + E_pot).real


def simuler_et_afficher(scenario="barriere"):
    N       = 1024
    dt      = 0.002
    T_max   = 4.0
    N_steps = int(T_max / dt)

    if scenario == "barriere":
        x, dx, k = construire_grille(-12, 12, N)
        V    = potentiel_barriere(x, x_centre=2.0, largeur=0.8, hauteur=8.0)
        psi0 = paquet_onde_gaussien(x, x0=-4.0, sigma=1.0, k0=3.0)
        titre = "Effet tunnel — Barrière de potentiel"
    elif scenario == "harmonique":
        x, dx, k = construire_grille(-8, 8, N)
        V    = potentiel_harmonique(x, omega=2.0)
        psi0 = paquet_onde_gaussien(x, x0=3.0, sigma=0.7, k0=0.0)
        titre = "Oscillateur harmonique quantique"
    elif scenario == "double_puits":
        x, dx, k = construire_grille(-4, 4, N)
        V    = potentiel_double_puits(x, a=4.0, b=1.0)
        psi0 = paquet_onde_gaussien(x, x0=-1.4, sigma=0.4, k0=0.0)
        titre = "Double puits — Tunneling quantique"
    else:
        raise ValueError(f"Scénario inconnu : {scenario}")

    save_every = 5
    psi_hist, t_hist = resoudre(psi0, V, k, dt, N_steps, save_every=save_every)

    normes   = [norme(p, dx) for p in psi_hist]
    energies = [energie(p, V, k, dx) for p in psi_hist]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(titre, fontsize=14, fontweight='bold')
    indices = [0, len(t_hist)//2, -1]
    labels  = ["t = 0 (initial)", f"t = {t_hist[len(t_hist)//2]:.2f}", f"t = {t_hist[-1]:.2f} (final)"]
    V_affiche = V / (np.max(np.abs(V)) + 1e-10) * 0.5
    for ax, idx, label in zip(axes, indices, labels):
        rho = np.abs(psi_hist[idx])**2
        ax.fill_between(x, rho, alpha=0.5, color='royalblue', label=r'$|\psi|^2$')
        ax.plot(x, rho, color='royalblue', linewidth=1.5)
        ax.plot(x, V_affiche, color='tomato', linewidth=2, linestyle='--', label='V(x) [normalisé]')
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Position x")
        ax.set_ylabel(r"$|\psi(x,t)|^2$")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.1, max(np.max(np.abs(psi_hist[0])**2), 0.5) * 1.3)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig2.suptitle("Quantités conservées (diagnostics numériques)", fontsize=13)
    ax1.plot(t_hist, normes, color='steelblue', linewidth=2)
    ax1.set_title("Norme L² = ∫|ψ|²dx")
    ax1.set_xlabel("Temps")
    ax1.set_ylabel("Norme")
    ax1.set_ylim(min(normes) * 0.9999, max(normes) * 1.0001)
    ax1.grid(True, alpha=0.3)
    ax2.plot(t_hist, energies, color='darkorange', linewidth=2)
    ax2.set_title("Énergie totale <H>")
    ax2.set_xlabel("Temps")
    ax2.set_ylabel("Énergie")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    return psi_hist, t_hist, x, V, fig, fig2, normes, energies


if __name__ == "__main__":
    for sc in ["barriere", "harmonique", "double_puits"]:
        simuler_et_afficher(sc)