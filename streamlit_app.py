import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
from simulation import (
    solve_stationary_2d,
    get_mode_2d,
    solve_time_basis,
    density_t,
    density_surface,
)
from visualisation import rendu_artistique, generer_rosace, generer_nebuleuse, generer_cristal, generer_mandala, generer_galaxie
from Fourier import simuler_et_afficher

st.set_page_config(page_title="Visualisation de Schrödinger", layout="wide")

st.title("Visualisation de l'équation de Schrödinger")
page = st.sidebar.radio(
    "Choix de la partie",
    ["Régime stationnaire 2D", "Régime dèpendant du temps 1D", "Art quantique", "Split-Step Fourier"]
)

NX = 301
NT = 80
T_MAX = 0.02

if "donnees_art" not in st.session_state:
    st.session_state["donnees_art"] = {}

if page == "Régime stationnaire 2D":
    st.header("Régime stationnaire 2D")

    st.sidebar.subheader("Paramètres physiques")
    N = st.sidebar.slider("Résolution de la grille", 80, 250, 150)
    k = st.sidebar.slider("Nombre de modes calculés", 3, 20, 10)
    mode = st.sidebar.slider("Mode à afficher", 0, k - 1, 0)

    x0 = st.sidebar.slider("Position x du potentiel", 0.0, 1.0, 0.3)
    y0 = st.sidebar.slider("Position y du potentiel", 0.0, 1.0, 0.3)
    sigma = st.sidebar.slider("Largeur du potentiel", 0.02, 0.30, 0.10)
    amplitude = st.sidebar.slider("Amplitude du potentiel", 0.1, 5.0, 1.0)

    X, Y, V, valeurs_propres, vecteurs_propres = solve_stationary_2d(
        N=N, k=k, x0=x0, y0=y0, sigma=sigma, amplitude=amplitude,
    )

    psi = get_mode_2d(vecteurs_propres, N, mode)
    rho, phi, rvb = rendu_artistique(psi)

    st.session_state["donnees_art"]["stationnaire_2d"] = {
        "X": X, "Y": Y, "V": V,
        "valeurs_propres": valeurs_propres,
        "vecteurs_propres": vecteurs_propres,
        "psi": psi, "rho": rho, "phi": phi, "N": N, "k": k,
    }

    st.subheader("Visualisation scientifique")
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        im1 = ax1.imshow(V, origin="lower", extent=(0, 1, 0, 1), cmap="viridis", aspect="equal")
        ax1.set_title("Potentiel V(x,y)")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        im2 = ax2.imshow(rho, origin="lower", extent=(0, 1, 0, 1), cmap="magma", aspect="equal")
        ax2.contour(X, Y, rho, levels=15, linewidths=0.6, colors="white")
        ax2.set_title(f"Densité du mode {mode}")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    st.subheader("Valeurs propres calculées")
    st.write(valeurs_propres)

elif page == "Régime dépendant du temps 1D":
    st.header("Régime dépendant du temps 1D")

    st.sidebar.subheader("Paramètres physiques")
    n_modes = st.sidebar.slider("Nombre de modes utilisés", 10, 100, 70, step=10)
    mu = st.sidebar.slider("Centre du potentiel", 0.1, 0.9, 0.5)
    sigma = st.sidebar.slider("Largeur du potentiel", 0.01, 0.15, 0.05)
    amplitude = st.sidebar.slider("Amplitude du puits", -20000.0, -100.0, -10000.0, step=100.0)
    t = st.sidebar.slider("Temps", 0.0, 0.05, 0.01)

    x, psi0, Vx, E_js, psi_js, cs = solve_time_basis(
        Nx=NX, mu=mu, sigma=sigma, amplitude=amplitude, n_modes=n_modes,
    )
    rho_t = density_t(x, E_js, psi_js, cs, t)

    st.session_state["donnees_art"]["temporel_1d"] = {
        "x": x, "psi0": psi0, "Vx": Vx,
        "E_js": E_js, "psi_js": psi_js, "cs": cs,
        "t": t, "rho_t": rho_t,
    }

    col1, col2 = st.columns(2)

    with col1:
        fig5, ax5 = plt.subplots(figsize=(5, 5))
        ax5.plot(x, psi0**2, label=r"$|\psi_0|^2$")
        ax5.plot(x, rho_t, label=rf"$|\psi(x,t)|^2$ à $t={t:.4f}$")
        ax5.set_xlabel("Position")
        ax5.set_ylabel("Densité de probabilité")
        ax5.set_title("Évolution temporelle de la densité")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        fig5.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        plt.close(fig5)

    with col2:
        fig6, ax6 = plt.subplots(figsize=(5, 5))
        ax6.plot(x, Vx)
        ax6.set_title("Potentiel V(x)")
        ax6.set_xlabel("Position")
        ax6.set_ylabel("Énergie potentielle")
        ax6.grid(True, alpha=0.3)
        fig6.tight_layout()
        st.pyplot(fig6, use_container_width=True)
        plt.close(fig6)

    st.subheader("Surface 3D de la densité")
    vals_temps = np.linspace(0, T_MAX, NT)
    rho_surface = density_surface(x, E_js, psi_js, cs, vals_temps)
    Tgrid, Xgrid = np.meshgrid(vals_temps, x)

    col3, col4 = st.columns(2)
    with col3:
        fig7 = plt.figure(figsize=(5, 5))
        ax7 = fig7.add_subplot(111, projection="3d")
        ax7.plot_surface(Xgrid, Tgrid, rho_surface.T, cmap="viridis")
        ax7.set_xlabel("Position")
        ax7.set_ylabel("Temps")
        ax7.set_zlabel(r"$|\psi(x,t)|^2$")
        ax7.set_title("Surface 3D de la densité de probabilité")
        fig7.tight_layout()
        st.pyplot(fig7, use_container_width=True)
        plt.close(fig7)

elif page == "Art quantique":
    st.header("Art quantique")

    donnees_art = st.session_state["donnees_art"]
    if not donnees_art:
        st.warning("Aucune donnée disponible. Exécute d'abord une simulation 2D ou 1D.")
        st.stop()

    labels_sources = {
        "stationnaire_2d": "Régime stationnaire 2D",
        "temporel_1d": "Régime temporel 1D",
    }

    # Contrôles dans la sidebar
    st.sidebar.subheader("Style")
    source = st.sidebar.selectbox(
        "Source des données",
        list(donnees_art.keys()),
        format_func=lambda k: labels_sources.get(k, k),
    )
    style_art = st.sidebar.radio(
        "Style artistique",
        ["Rosace", "Nébuleuse quantique", "Cristal quantique", "Mandala fractal", "Galaxie spirale"],
    )

    st.sidebar.subheader("Export")
    format_export = st.sidebar.radio("Format", ["PNG", "JPEG", "TIFF"])
 

    TAILLE_ART = 700

    if source == "stationnaire_2d":
        valeurs_propres = donnees_art[source]["valeurs_propres"]
    else:
        valeurs_propres = donnees_art[source]["E_js"]

    styles = {
        "Rosace":              (generer_rosace,    "Rosace"),
        "Nébuleuse quantique": (generer_nebuleuse, "Nébuleuse quantique"),
        "Cristal quantique":   (generer_cristal,   "Cristal quantique"),
        "Mandala fractal":     (generer_mandala,   "Mandala fractal"),
        "Galaxie spirale":     (generer_galaxie,   "Galaxie spirale"),
    }
    fn, titre = styles[style_art]
    image_art = fn(valeurs_propres, taille=TAILLE_ART)

    # Image centrée, taille limitée
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        figA, axA = plt.subplots(figsize=(5, 5))
        axA.imshow(image_art, origin="lower", aspect="equal")
        axA.axis("off")
        figA.subplots_adjust(left=0, right=1, top=1, bottom=0)
        st.pyplot(figA, use_container_width=True)
        plt.close(figA)

    # Bouton export
    img_uint8 = (np.clip(image_art, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    save_kwargs = {"quality": 95} if format_export == "JPEG" else {}
    pil_img.save(buf, format=format_export, **save_kwargs)
    buf.seek(0)

    nom_fichier = f"art_quantique_{style_art.lower().replace(' ', '_')}.{format_export.lower()}"
    st.download_button(
        label=f"Télécharger — {titre} · {format_export}",
        data=buf,
        file_name=nom_fichier,
        mime=f"image/{format_export.lower()}",
        use_container_width=True,
    )

elif page == "Split-Step Fourier":
    st.header("Split-Step Fourier — Schrödinger dépendante du temps")
    

    st.sidebar.subheader("Paramètres")
    scenario = st.sidebar.selectbox(
        "Scénario",
        ["barriere", "harmonique", "double_puits"],
        format_func=lambda s: {
            "barriere": "Effet tunnel — Barrière",
            "harmonique": "Oscillateur harmonique",
            "double_puits": "Double puits",
        }[s],
    )

    st.sidebar.subheader("Animation")
    n_frames   = st.sidebar.slider("Nombre de frames", 30, 150, 80)
    vitesse_ms = st.sidebar.slider("Vitesse (ms/frame)", 20, 200, 60)

    with st.spinner("Simulation en cours…"):
        psi_hist, t_hist, x, V, fig_densite, fig_diag, normes, energies = simuler_et_afficher(scenario)

    st.subheader("Densité de probabilité à trois instants")
    st.pyplot(fig_densite, use_container_width=True)
    plt.close(fig_densite)

    st.subheader("Quantités conservée")
    st.pyplot(fig_diag, use_container_width=True)
    plt.close(fig_diag)

    st.subheader("Résumé ")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Variation de norme", f"{max(normes) - min(normes):.2e}", help="Idéalement → 0")
    with col2:
        st.metric("Variation d'énergie", f"{max(energies) - min(energies):.4F}", help="Idéalement → 0")


    st.subheader(" Animation de l'évolution temporelle")

    total_frames  = len(psi_hist)
    indices_frames = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    psi_anim = psi_hist[indices_frames]
    t_anim   = t_hist[indices_frames]

    V_norm  = V / (np.max(np.abs(V)) + 1e-10) * 0.5
    rho_max = np.max(np.abs(psi_hist[0])**2) * 1.35

    titres_scenario = {
        "barriere":     "Effet tunnel — Barrière de potentiel",
        "harmonique":   "Oscillateur harmonique quantique",
        "double_puits": "Double puits — Tunneling quantique",
    }

    with st.spinner("Génération de l'animation GIF…"):
        fig_anim, ax_anim = plt.subplots(figsize=(9, 4.5))
        fig_anim.patch.set_facecolor("#0d1117")
        ax_anim.set_facecolor("#0d1117")

        # Tracé initial
        rho0     = np.abs(psi_anim[0])**2
        line_rho, = ax_anim.plot(x, rho0, color='#58a6ff', linewidth=2, label=r'$|\psi|^2$')
        line_V,   = ax_anim.plot(x, V_norm, color='#f97583', linewidth=2, linestyle='--', label='V(x) [norm.]')
        time_txt  = ax_anim.text(
            0.02, 0.93, f"t = {t_anim[0]:.3f}",
            transform=ax_anim.transAxes, color='white', fontsize=12, fontfamily='monospace'
        )

        ax_anim.set_xlim(x[0], x[-1])
        ax_anim.set_ylim(-0.05, rho_max)
        ax_anim.set_xlabel("Position x", color='#8b949e')
        ax_anim.set_ylabel(r"$|\psi(x,t)|^2$", color='#8b949e')
        ax_anim.tick_params(colors='#8b949e')
        for spine in ax_anim.spines.values():
            spine.set_edgecolor('#30363d')
        ax_anim.legend(loc='upper right', fontsize=9, facecolor='#161b22', labelcolor='white')
        ax_anim.set_title(titres_scenario[scenario], color='white', fontsize=13, pad=10)
        fig_anim.tight_layout()

        frames_images = []
        progress_bar  = st.progress(0, text="Rendu des frames…")

        for i in range(n_frames):
            rho_i = np.abs(psi_anim[i])**2

            # Supprimer les fills précédents
            for coll in list(ax_anim.collections):
                coll.remove()
            ax_anim.fill_between(x, rho_i, alpha=0.40, color='#58a6ff')

            line_rho.set_ydata(rho_i)
            time_txt.set_text(f"t = {t_anim[i]:.3f}")

            buf_frame = io.BytesIO()
            fig_anim.savefig(buf_frame, format='png', dpi=100, facecolor=fig_anim.get_facecolor())
            buf_frame.seek(0)
            frames_images.append(Image.open(buf_frame).copy())

            progress_bar.progress((i + 1) / n_frames, text=f"Frame {i+1}/{n_frames}")

        plt.close(fig_anim)
        progress_bar.empty()

        # Encoder en GIF
        buf_gif = io.BytesIO()
        frames_images[0].save(
            buf_gif,
            format='GIF',
            save_all=True,
            append_images=frames_images[1:],
            duration=vitesse_ms,
            loop=0,
            optimize=False,
        )
        buf_gif.seek(0)
        gif_bytes = buf_gif.read()

    # Affichage du GIF inline
    b64 = base64.b64encode(gif_bytes).decode()
    html_gif = f"""
    <div style="display:flex; flex-direction:column; align-items:center;
                background:#0d1117; border-radius:12px; padding:16px;
                border:1px solid #30363d; margin-bottom:12px;">
        <img src="data:image/gif;base64,{b64}"
             style="max-width:100%; border-radius:8px;"
             alt="Animation Split-Step Fourier"/>
    </div>
    """
    st.markdown(html_gif, unsafe_allow_html=True)

    