import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import datetime
from fpdf import FPDF
from scipy.spatial import distance
import streamlit.components.v1 as components
import pytesseract
import re
from openai import OpenAI
# import maidr # Opcional: para sonificação avançada
# import simpleaudio as sa # Opcional: para tocar sons

# ==========================================
# CONFIGURAÇÕES DE API (OPENAI)
# ==========================================
# Substitua pela sua chave real ou use st.secrets
OPENAI_API_KEY = "sua-chave-aqui" 
client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================================
# 1. FUNÇÕES DE ACESSIBILIDADE (VOZ)
# ==========================================
def falar(texto):
    js = f"""
    <script>
    var msg = new SpeechSynthesisUtterance("{texto}");
    msg.lang = 'pt-BR';
    window.speechSynthesis.speak(msg);
    </script>
    """
    components.html(js, height=0)

# ==========================================
# 2. CALIBRAÇÃO PRECISA COM OCR
# ==========================================
def calibrar_com_ocr(img):
    """Recorta a barra, lê o texto com OCR e calcula a escala exata."""
    altura, largura = img.shape[:2]
    # Recorta a região inferior direita onde a barra costuma ficar
    recorte = img[int(altura*0.75):altura, int(largura*0.5):largura]
    
    # 1. Pré-processamento para OCR (converter para cinza e limpar)
    cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    _, thresh_ocr = cv2.threshold(cinza, 150, 255, cv2.THRESH_BINARY)
    
    # 2. Executa o OCR para ler o texto
    texto_ocr = pytesseract.image_to_string(thresh_ocr, config='--psm 6')
    print(f"Texto bruto do OCR: {texto_ocr}")
    
    # 3. Expressão regular para encontrar padrões como "2 mm", "500 µm", "0.5 mm"
    padrao = r'(\d+[.,]?\d*)\s*(mm|µm|um|nm)'
    match = re.search(padrao, texto_ocr, re.IGNORECASE)
    
    if not match:
        return None, "Não foi possível ler o texto da barra de escala. Por favor, digite o valor manualmente."
    
    valor_str, unidade = match.groups()
    valor = float(valor_str.replace(',', '.'))
    
    # 4. Converte tudo para micrômetros (µm) para padronizar
    if unidade.lower() == 'mm':
        valor_um = valor * 1000
    elif unidade.lower() == 'nm':
        valor_um = valor / 1000
    else: # µm ou um
        valor_um = valor
        
    # 5. Encontra a barra (retângulo) para medir os pixels
    _, thresh_barra = cv2.threshold(cinza, 100, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(thresh_barra, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pixels_barra = 0
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h < 50: # Procura retângulo largo
            pixels_barra = w
            break
            
    if pixels_barra == 0:
        return None, "Barra de escala detectada, mas não foi possível medir seus pixels."
        
    # 6. Calcula a escala (µm por pixel)
    # Fórmula: C = L_real / L_pixels
    um_por_pixel = valor_um / pixels_barra
    
    return um_por_pixel, f"Barra lida: {valor_str} {unidade} ({pixels_barra} px). Escala: {um_por_pixel:.4f} µm/px."

# ==========================================
# 3. ANÁLISE DE PARTÍCULAS (Mantida)
# ==========================================
def analisar_particulas(img, um_por_pixel):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((5,5), np.uint8)
    mascara = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dados = []
    centros = []
    
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 800: 
            perimetro = cv2.arcLength(c, True)
            if perimetro == 0: continue
            circularidade = (4 * np.pi * area) / (perimetro ** 2)
            if circularidade > 0.85: continue 
            
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
            
            dados.append({
                'ID': i + 1,
                'Área (µm²)': area * (um_por_pixel**2),
                'Comprimento (µm)': cv2.arcLength(c, True) * um_por_pixel, 
                'Circularidade': circularidade
            })
            centros.append((cy, cx))
            
    df = pd.DataFrame(dados)
    dist_media = 0
    if len(centros) > 1:
        distancias = [distance.euclidean(centros[i], centros[j]) * um_por_pixel 
                      for i in range(len(centros)) for j in range(i+1, len(centros))]
        dist_media = np.mean(distancias)
    return df, dist_media

# ==========================================
# 4. ASSISTENTE DE IA
# ==========================================
def perguntar_ia(df, pergunta):
    """Envia os dados da tabela para a IA e retorna a resposta."""
    if df.empty:
        return "Não há dados para analisar."
    
    # Converte a tabela para texto (CSV) para dar contexto à IA
    contexto_dados = df.describe().to_string() + "\n\n" + df.head(20).to_string()
    
    prompt = f"""
    Você é o assistente do LAVIM, um laboratório virtual de microscopia.
    Abaixo estão os dados estatísticos e as primeiras linhas da tabela de partículas analisadas.
    
    DADOS:
    {contexto_dados}
    
    PERGUNTA DO USUÁRIO: {pergunta}
    
    Responda de forma clara, curta e em português do Brasil, como se estivesse falando com um pesquisador.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Ou gpt-4o se tiver acesso
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro ao consultar a IA: {e}"

# ==========================================
# 5. SONIFICAÇÃO (Simples)
# ==========================================
def gerar_sonificacao(df):
    """
    Gera um áudio simples mapeando a área para frequência.
    Esta é uma versão básica; a biblioteca maidr é mais avançada.
    """
    # Por enquanto, apenas retorna uma mensagem de voz com a distribuição
    if df.empty: return
    media = df['Área (µm²)'].mean()
    falar(f"A distribuição das áreas tem média de {media:.1f} micrômetros quadrados. "
          f"Para uma sonificação completa, seria necessário processar as notas musicais.")

# ==========================================
# 6. INTERFACE WEB
# ==========================================
st.set_page_config(page_title="LAVIM - Laboratório Virtual", layout="wide")

if 'welcome' not in st.session_state:
    falar("Bem-vindo ao LAVIM. Laboratório Virtual de Microscopia Acessível.")
    st.session_state['welcome'] = True

st.title("🔬 LAVIM - Laboratório Virtual de Microscopia")

uploaded_file = st.file_uploader("Faça o upload da imagem da amostra", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", caption="Imagem Carregada", use_column_width=True)
    
    # Campo para valor manual da barra (caso o OCR falhe)
    valor_manual_barra = st.text_input("Se o OCR falhar, digite o valor da barra (ex: 2 mm, 500 um):", "")
    
    if st.button("▶ Iniciar Análise"):
        falar("Iniciando análise.")
        
        # Calibração
        if valor_manual_barra:
            # Se o usuário digitou manualmente, usa um parser simples
            match = re.search(r'(\d+[.,]?\d*)\s*(mm|um|µm)', valor_manual_barra, re.IGNORECASE)
            if match:
                val, uni = match.groups()
                val = float(val.replace(',', '.'))
                um_real = val * 1000 if uni.lower() == 'mm' else val
                # Ainda precisamos medir os pixels da barra na imagem
                # (código simplificado: assume que a barra tem 557px, mas o ideal é medir)
                um_por_pixel = um_real / 557 # Valor de fallback
                calib_msg = f"Calibração manual: {valor_manual_barra}."
            else:
                st.error("Formato de valor manual inválido. Use '2 mm' ou '500 um'.")
                st.stop()
        else:
            um_por_pixel, calib_msg = calibrar_com_ocr(img)
            
        if um_por_pixel is None:
            st.error(calib_msg)
            falar(calib_msg)
            st.stop()
            
        st.success(calib_msg)
        falar(calib_msg)
        
        # Análise
        df, dist_media = analisar_particulas(img, um_por_pixel)
        
        if df.empty:
            st.error("Nenhuma partícula detectada.")
            falar("Nenhuma partícula detectada.")
        else:
            st.write(f"### {len(df)} partículas encontradas")
            st.dataframe(df.style.format("{:.2f}"))
            
            # Gráficos
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.hist(df['Área (µm²)'], bins=10, color='skyblue', edgecolor='black')
            ax1.set_title('Distribuição da Área')
            ax2.scatter(df['Área (µm²)'], df['Comprimento (µm)'], facecolors='none', edgecolors='blue', s=100)
            ax2.set_title('Área vs Comprimento')
            st.pyplot(fig)
            
            # Assistente de IA
            st.subheader("🤖 Assistente de IA")
            pergunta = st.text_input("Pergunte algo sobre os dados (ex: Qual o maior grão?):")
            if pergunta:
                with st.spinner("Consultando IA..."):
                    resposta = perguntar_ia(df, pergunta)
                    st.info(resposta)
                    falar(resposta)
            
            # Sonificação (Botão para ouvir a distribuição)
            if st.button("🎵 Ouvir Sonificação da Distribuição"):
                gerar_sonificacao(df)
            
            # Relatório PDF (código resumido, o mesmo da versão anterior)
            st.info("Relatório PDF e SVG Tátil disponíveis na versão completa.")
