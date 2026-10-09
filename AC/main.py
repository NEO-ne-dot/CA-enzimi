import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re

# Page Configuration
st.set_page_config(
    page_title="CA & Karbon Yakalama Simülatörü",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Karbonik Anhidraz (CA) & Karbon Yakalama (CCS) Simülatörü")
st.markdown("""
Bu simülatörde sulu ortamdaki kimyasal formüllerin yanı sıra **Karbon Yakalama Teknolojilerinde ($CO_2$ Capture)** 
enzim tutuklama (immobilizasyon) ve hibrit destek malzemesi olarak kullanılan **SBA-15, MOF, ZIF, Zeolit, Amine-Silika ve Titanya** 
gibi tüm nanogözenekli matrislerin etkilerini simüle edebilirsiniz.
""")

# --- ELEMENTAL DATA & ATOMIC WEIGHTS ---
ATOMIC_WEIGHTS = {
    'H': 1.008, 'B': 10.81, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.06,
    'Cl': 35.453, 'K': 39.098, 'Ca': 40.078, 'Ti': 47.867, 'Fe': 55.845, 'Cu': 63.546,
    'Zn': 65.380, 'Br': 79.904, 'Zr': 91.224
}

# Advanced Carbon Capture Matrices Database
CCS_MATRICES = {
    "SBA-15": {
        "name": "SBA-15 (Mesogözenekli Silika)",
        "mw": 60.08,
        "delta_Tm": 16.0,
        "retention": 88.0,
        "elements": {'Si': 1, 'O': 2},
        "description": "Yüksek yüzey alanına sahip altıgen mesogözenekli silika. Enzimi termal denatürasyona karşı korur ve CO2 tutum dayanımını artırır."
    },
    "MCM-41": {
        "name": "MCM-41 (Nanogözenekli Silika)",
        "mw": 60.08,
        "delta_Tm": 13.0,
        "retention": 82.0,
        "elements": {'Si': 1, 'O': 2},
        "description": "Düzenli küresel gözenekli silika matris. Fiziksel tutuklama ile termal kararlılık sağlar."
    },
    "MOF-808": {
        "name": "MOF-808 (Zirkonyum Bazlı MOF)",
        "mw": 1350.0,
        "delta_Tm": 22.0,
        "retention": 92.0,
        "elements": {'Zr': 6, 'O': 8, 'C': 12, 'H': 6},
        "description": "Zirkonyum küme yapılı Metal-Organik Kafes (MOF). Enzimin mikroçevre kararlılığını ve asidik ortamlara direncini yükseltir."
    },
    "ZIF-8": {
        "name": "ZIF-8 (Çinko İmidazolat Kafes)",
        "mw": 229.6,
        "delta_Tm": 25.0,
        "retention": 95.0,
        "elements": {'Zn': 1, 'C': 8, 'H': 10, 'N': 4},
        "description": "Biyokompatibilitesi yüksek Zeolitik İmidazolat Kafes. CA enzimini biyomimetik zırhlama ile yüksek ısıya karşı korur."
    },
    "APTES-SBA15": {
        "name": "Amine-Functionalized Silika (APTES-SBA-15)",
        "mw": 179.28,
        "delta_Tm": 19.0,
        "retention": 90.0,
        "elements": {'Si': 1, 'C': 3, 'H': 9, 'N': 1, 'O': 3},
        "description": "Amin grupları (–NH₂) yüklenmiş silika. CO2 afinitesini artırarak enzimatik hidratasyon verimine katkı sağlar."
    },
    "PEI-SILICA": {
        "name": "PEI (Polietilenimin) Modifiye Matris",
        "mw": 43.07,
        "delta_Tm": 15.0,
        "retention": 85.0,
        "elements": {'C': 2, 'H': 5, 'N': 1},
        "description": "Poliamin kaplı destek malzemesi. CO2 yakalama reaktörlerinde rezonans tamponlama sağlar."
    },
    "TIO2-NANO": {
        "name": "TiO2 (Titanya Nanopartikül Matris)",
        "mw": 79.87,
        "delta_Tm": 11.0,
        "retention": 80.0,
        "elements": {'Ti': 1, 'O': 2},
        "description": "Fotokatalitik/Biyokatalitik hibrit sistemlerde kullanılan nanoyapılı tutuklama desteği."
    },
    "ZEOLITE-13X": {
        "name": "Zeolit 13X (Alüminosilikat Matris)",
        "mw": 161.9,
        "delta_Tm": 14.0,
        "retention": 84.0,
        "elements": {'Na': 1, 'Al': 1, 'Si': 1, 'O': 4},
        "description": "Yüksek CO2 adsorpsiyon kapasitesine sahip alüminosilikat kafes."
    }
}

ION_EFFECTS = {
    'B': {'charge': 3, 'inhibition_potency': 0.70, 'destabilization': -0.3, 'type': 'Bor / Borat Kompleksi (Aktif Bölge İnhibitörü & Tampon)'},
    'Zn': {'charge': 2, 'inhibition_potency': 0.85, 'destabilization': 0.7, 'type': 'Ağır Metal İnhibitörü'},
    'Cu': {'charge': 2, 'inhibition_potency': 0.95, 'destabilization': 0.9, 'type': 'Güçlü İnhibitör / Denatüre Edici'},
    'Fe': {'charge': 2, 'inhibition_potency': 0.50, 'destabilization': 0.4, 'type': 'Orta İnhibitör'},
    'Ca': {'charge': 2, 'inhibition_potency': 0.10, 'destabilization': -0.3, 'type': 'Hafif Stabilizatör'},
    'Mg': {'charge': 2, 'inhibition_potency': 0.05, 'destabilization': -0.4, 'type': 'İyonik Stabilizatör'},
    'Na': {'charge': 1, 'inhibition_potency': 0.02, 'destabilization': -0.2, 'type': 'Nötr / Hafif Tuz Etkisi'},
    'K':  {'charge': 1, 'inhibition_potency': 0.01, 'destabilization': -0.2, 'type': 'Nötr / Hafif Tuz Etkisi'}
}


def parse_formula(formula):
    tokens = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    parsed = {}
    total_mw = 0.0

    for element, count in tokens:
        if not element:
            continue
        cnt = int(count) if count else 1
        parsed[element] = parsed.get(element, 0) + cnt
        mw = ATOMIC_WEIGHTS.get(element, 20.0)
        total_mw += mw * cnt

    return parsed, total_mw


# --- SIDEBAR ---
st.sidebar.header("🧪 1. Kimyasal / CCS Matris Seçimi")

preset_choice = st.sidebar.selectbox(
    "Örnek Madde veya Karbon Yakalama Matrisi Seçin",
    [
        "SBA-15 (Mesogözenekli Silika)",
        "MCM-41 (Nanogözenekli Silika)",
        "ZIF-8 (Çinko İmidazolat Kafes MOF)",
        "MOF-808 (Zirkonyum Bazlı MOF)",
        "APTES-SBA15 (Amin-Modifiye Silika)",
        "Zeolite 13X (Alüminosilikat)",
        "TiO2 Nanopartikül",
        "H3BO3 (Borik Asit)",
        "Na2B4O7 (Boraks)",
        "ZnSO4",
        "CuSO4",
        "CaCO3",
        "Özel Formül Yaz"
    ]
)

if "SBA-15" in preset_choice:
    default_chem = "SBA-15"
elif "MCM-41" in preset_choice:
    default_chem = "MCM-41"
elif "ZIF-8" in preset_choice:
    default_chem = "ZIF-8"
elif "MOF-808" in preset_choice:
    default_chem = "MOF-808"
elif "APTES" in preset_choice:
    default_chem = "APTES-SBA15"
elif "Zeolite" in preset_choice:
    default_chem = "ZEOLITE-13X"
elif "TiO2" in preset_choice:
    default_chem = "TIO2-NANO"
elif preset_choice != "Özel Formül Yaz":
    default_chem = preset_choice.split()[0]
else:
    default_chem = "SBA-15"

chemical_input = st.sidebar.text_input(
    "Girdi (Formül veya Matris Adı)", value=default_chem)
chemical_input = chemical_input.strip()

st.sidebar.subheader("Deneysel Koşullar")
max_conc = st.sidebar.slider("Maksimum Miktar / Derişim (mM veya mg/mL)",
                             min_value=10.0, max_value=500.0, value=100.0, step=10.0)
work_temp = st.sidebar.slider(
    "Çalışma Sıcaklığı (°C)", min_value=20.0, max_value=95.0, value=37.0, step=1.0)
base_Tm = st.sidebar.slider("Serbest Enzim Baz Tm (°C)",
                            min_value=45.0, max_value=65.0, value=58.0, step=1.0)

# --- CALCULATIONS ---
chem_conc = np.linspace(0, max_conc, 200)

matched_matrix_key = None
clean_input = chemical_input.upper().replace(
    "-", "").replace("_", "").replace(" ", "")

for key in CCS_MATRICES:
    clean_key = key.upper().replace("-", "").replace("_", "").replace(" ", "")
    if clean_key in clean_input or clean_input in clean_key:
        matched_matrix_key = key
        break

if matched_matrix_key:
    mat_data = CCS_MATRICES[matched_matrix_key]
    matrix_name = mat_data["name"]
    mol_weight = mat_data["mw"]
    delta_Tm_max = mat_data["delta_Tm"]
    retention_base = mat_data["retention"]
    parsed_elements = mat_data["elements"]

    delta_Tm = delta_Tm_max * (chem_conc / (chem_conc + 15.0))
    current_Tm = base_Tm + delta_Tm
    activity_yield_raw = np.full_like(chem_conc, retention_base)

    ion_notes = [
        f"**{matrix_name}:** {mat_data['description']}",
        f"**Termal Koruma:** Erime sıcaklığını ($T_m$) +{delta_Tm_max:.1f} °C kadar artırarak yüksek sıcaklıklarda aktiviteyi korur.",
        f"**İmmobilizasyon Verimi:** Gözenek içi difüzyon etkisine bağlı olarak temel oda sıcaklığı aktivite korunumu ~%{retention_base:.0f} seviyesindedir."
    ]
    is_ccs_matrix = True

else:
    is_ccs_matrix = False
    parsed_elements, mol_weight = parse_formula(chemical_input)
    inhibition_factor = 0.05
    stability_shift_factor = 0.0
    ion_notes = []

    for elem, count in parsed_elements.items():
        if elem in ION_EFFECTS:
            info = ION_EFFECTS[elem]
            inhibition_factor += info.get('inhibition_potency', 0.0) * count
            stability_shift_factor -= info.get('destabilization', 0.0) * count
            ion_notes.append(f"**{elem} ({count} adet):** {info['type']}")

    if "B4O7" in chemical_input:
        stability_shift_factor += 0.8
        ion_notes.append(
            "**[B4O7] Tetraborat:** Güçlü Tampon ve Termal Kararlılık Destekleyici")
    elif "BO3" in chemical_input or "B" in parsed_elements:
        stability_shift_factor += 0.5
        ion_notes.append(
            "**[BO3 / Borat]:** Enzim Aktif Bölgesiyle Etkileşen Borat Yapısı")

    if "SO4" in chemical_input:
        stability_shift_factor += 0.8
        ion_notes.append(
            "**[SO4] Sülfat:** Kosmotropik Anyon (Proteini Korur)")

    Ki_est = max(0.5, 35.0 / (inhibition_factor * 8 + 0.1))
    activity_yield_raw = 100.0 / (1.0 + (chem_conc / Ki_est))

    delta_Tm_max = stability_shift_factor * 7.0
    delta_Tm = delta_Tm_max * (chem_conc / (chem_conc + 20.0))
    current_Tm = base_Tm + delta_Tm

# Thermal Folding Calculation
slope = 0.4
folded_fraction = 100.0 / (1.0 + np.exp(slope * (work_temp - current_Tm)))
final_activity = activity_yield_raw * (folded_fraction / 100.0)

df = pd.DataFrame({
    'Miktar / Derişim': chem_conc,
    'Aktivite Verimi (%)': final_activity,
    'Erime Sıcaklığı Tm (°C)': current_Tm,
    'Katlanmış Enzim Oranı (%)': folded_fraction
})

# --- DASHBOARD LAYOUT ---
system_type_str = "CCS İmmobilizasyon Matrisi" if is_ccs_matrix else "Sulu Çözelti Kimyasalı"
st.success(
    f"✅ **{chemical_input}** Çözümlendi | Mol Kütlesi (Tekrarlayan Birim): ~{mol_weight:.2f} g/mol | Tür: {system_type_str}")

col1, col2 = st.columns(2)

with col1:
    fig_act = go.Figure()
    fig_act.add_trace(go.Scatter(
        x=df['Miktar / Derişim'],
        y=df['Aktivite Verimi (%)'],
        mode='lines',
        name='Verim (%)',
        line=dict(color='#2980B9', width=3)
    ))
    fig_act.update_layout(
        title=f"<b>{chemical_input} Varlığında Net Aktivite Verimi</b>",
        xaxis_title=f"{chemical_input} Miktarı / Derişimi",
        yaxis_title="Relatif Aktivite Verimi (%)",
        yaxis=dict(range=[0, 110]),
        template="plotly_white"
    )
    st.plotly_chart(fig_act, use_container_width=True)

with col2:
    line_color_str = '#27AE60' if delta_Tm_max >= 0 else '#E74C3C'
    fig_stab = go.Figure()
    fig_stab.add_trace(go.Scatter(
        x=df['Miktar / Derişim'],
        y=df['Erime Sıcaklığı Tm (°C)'],
        mode='lines',
        name='Tm Sıcaklığı (°C)',
        line=dict(color=line_color_str, width=3)
    ))
    fig_stab.add_hline(
        y=work_temp,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Çalışma Sıcaklığı ({work_temp}°C)"
    )
    fig_stab.update_layout(
        title=f"<b>{chemical_input} Etkisiyle Termal Kararlılık (Tm)</b>",
        xaxis_title=f"{chemical_input} Miktarı / Derişimi",
        yaxis_title="Erime Sıcaklığı Tm (°C)",
        template="plotly_white"
    )
    st.plotly_chart(fig_stab, use_container_width=True)

# --- REPORT ---
st.markdown("---")
st.subheader(f"📊 {chemical_input} Otomatik Analiz Raporu")

mid_idx = len(chem_conc) // 2

st.write(f"**Sistem Türü:** {system_type_str}")
st.write(f"**Hesaplanan Birim Kütlesi:** ~{mol_weight:.2f} g/mol")

st.markdown("**Mekanizma ve Özellikler:**")
if ion_notes:
    for note in ion_notes:
        st.markdown(f"- {note}")
else:
    st.markdown("- Standart iyonik etki")

st.markdown(f"**{chem_conc[mid_idx]:.1f} Miktarındaki Simülasyon Sonuçları:**")
st.markdown(f"- **Net Aktivite Verimi:** %{final_activity[mid_idx]:.1f}")
st.markdown(
    f"- **Erime Sıcaklığı ($T_m$):** {current_Tm[mid_idx]:.1f} °C (Değişim: {current_Tm[mid_idx]-base_Tm:+.1f} °C)")

if is_ccs_matrix:
    st.info("💡 **CCS Matris Notu:** SBA-15, ZIF-8 ve MOF yapılarında tutuklanmış (immobilize) Karbonik Anhidraz enzimi, serbest enzimin aksine 60–80°C sıcaklıklardaki endüstriyel CO2 baca gazı yıkayıcılarında (scrubber) denatüre olmadan uzun ömürlü olarak kullanılabilir.")

st.download_button(
    label="📥 Verileri CSV Olarak İndir",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name=f"{chemical_input}_CA_simulasyon.csv",
    mime="text/csv"
)
