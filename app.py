import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import geopandas as gpd
import seaborn as sns
import folium
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import streamlit as st

from shapely.geometry import Point
from matplotlib import pyplot as plt
from folium import GeoJson
from folium.features import GeoJsonPopup, GeoJsonTooltip
from unidecode import unidecode
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from plotly.subplots import make_subplots

# ================ PARÂMETROS ================

# Paths
# BASE_PATH = '/mnt/d/PESSOAL/240319-RS-MATR/notebooks' # DEV
BASE_PATH = '/mount/src/matr/' # PRD
DATA_PATH = f'{BASE_PATH}/data'

# Configurações de Mapa
DISPLAY_MAP = False
INITIAL_COORDS = [-51.1794, -29.1678] # Caxias do Sul
BASEMAPS = [
    'Esri.WorldStreetMap',        # 0 
    'Esri.WorldTopoMap',          # 1
    'Esri.WorldImagery',          # 2
    'OpenTopoMap',                # 3
    'Stadia.AlidadeSmooth',       # 4
    'Stadia.AlidadeSmoothDark',   # 5
    'Stadia.AlidadeSatellite'     # 6
]

# Colunas dos Dataframes
COLS_MONITORAMENTO = [
    'bairro',
    'data',
    'temperatura',
    'umidade',
    'luminosidade',
    'ruido',
    'eco2',
    'etvoc',
    'F_PERIODO',
    'F_HORA',
    'F_MINUTO',
    'F_DIA',
    'F_MES',
    'F_ANO',
    'F_DIA_SEMANA'
]
COLS_SEGURANCA = [
    'Municipio',
    'Bairro',
    'Data Fato',
    'Dia Semana Fato',
    'Hora Fato',
    'Tipo Local',
    'Desc Fato',
    'Tipo Fato',
    'Flagrante',
    'Endereco',
    'Nro Endereco',
    'data',
    'F_PERIODO',
    'F_HORA',
    'F_MINUTO',
    'F_DIA',
    'F_MES',
    'F_ANO',
    'F_DIA_SEMANA',
    'F_CLASSIFICACAO'
]
COLS_SATISFACAO = [
    'BAIRRO', 
    'Qtd respostas', 
    'Satisfação com o bairro',
    'Satisfação com a Saúde', 
    'Prática de atividade física',
    'Satisfação financeira', 
    'Satisfação com atividade comercial',
    'Satisfação com qualidade do ar', 
    'Satisfação com ruído',
    'Satisfação com espaços de lazer', 
    'Satistação com coleta de lixo',
    'Satisfação com distância da parada de ônibus',
    'Satisfação com qualidade das paradas de ônibus',
    'Satisfação com acesso aos locais importantes da cidade',
    'Sentimento de segurança', 
    'Sentimento de confiança nas pessoas',
    'Satisfação com tratamento de esgoto'
]

# Dias da Semana
DAY_NAME_MAP = {
    'Monday': 'SEG',
    'Tuesday': 'TER',
    'Wednesday': 'QUA',
    'Thursday': 'QUI',
    'Friday': 'SEX',
    'Saturday': 'SAB',
    'Sunday': 'DOM'
}

st.set_page_config(layout='wide')

# ================ CLASSES DE NEGÓCIO ================
class DataLoader:
    @staticmethod
    @st.cache_data
    def loadCSV(folderPath, fileName, separator=','):
        """
        Carrega dados de um arquivo CSV em um DataFrame pandas.
        
        Parâmetros:
        - folderPath (str): Caminho para a pasta onde o arquivo CSV está localizado.
        - fileName (str): Nome do arquivo CSV.
        
        Retorna:
        - DataFrame: Dados carregados do CSV.
        """
        filePath = os.path.join(folderPath, f'{fileName}.csv')

        try:
            if os.path.exists(filePath):
                df = pd.read_csv(filePath, sep=separator)
                return df
            else:
                raise FileNotFoundError(f"Arquivo {fileName} não encontrado na pasta {folderPath}.")
        except Exception as e:
            print(f"Erro ao carregar o arquivo CSV: {e}")
            return None

    @staticmethod
    @st.cache_data
    def loadXLSX(folderPath, fileName, sheetIndex=0):
        """
        Carrega dados de um arquivo XLSX em um DataFrame pandas.
        
        Parâmetros:
        - folderPath (str): Caminho para a pasta onde o arquivo XLSX está localizado.
        - fileName (str): Nome do arquivo XLSX.
        - sheetIndex (int, opcional): Índice da planilha a ser carregada. Padrão (0).
        
        Retorna:
        - DataFrame: Dados carregados do XLSX.
        """
        filePath = os.path.join(folderPath, f'{fileName}.xlsx')
        
        try:
            if os.path.exists(filePath):
                df = pd.read_excel(filePath)
                return df
            else:
                raise FileNotFoundError(f"Arquivo {fileName} não encontrado na pasta {folderPath}.")
        except Exception as e:
            print(f"Erro ao carregar o arquivo XLSX: {e}")
            return None

    @staticmethod
    @st.cache_data
    def loadSHP(folderPath, shpName):
        """
        Carrega o shapefile dos limites dos bairros em um GeoDataFrame.
        
        Parâmetros:
        - folderPath (str): Caminho para a pasta onde o arquivo XLSX está localizado.
        - shpName (str): Nome do arquivo shapefile.
        
        Retorna:
        - GeoDataFrame: Dados geoespaciais dos bairros.
        """
        filePath = os.path.join(folderPath, f'{shpName}.shp')
        
        try:
            if os.path.exists(filePath):
                gdf = gpd.read_file(filePath)
                return gdf
            else:
                raise FileNotFoundError(f"Arquivo {filePath} não encontrado.")
        except Exception as e:
            print(f"Erro ao carregar o shapefile: {e}")
            return None

class MapUtils:
    @staticmethod
    def createMap(
        initialCoords=[-46.633308,-23.55052], 
        zoomStart=12, 
        basemap='OpenStreetMap.Mapnik',
        controlScale=True, 
        zoomControl=True, 
        scrollWheelZoom=True, 
        dragging=True):
        """
        Cria um mapa folium com os dados de um GeoDataFrame.
        
        Parâmetros:
        - initialCoords (list): Coordenadas iniciais [latitude, longitude] para centrar o mapa.
        - zoomStart (int): Nível inicial de zoom do mapa.
        - controlScale (boolean): Controla o nível de escala no mapa.
        - zoomControl (boolean): Controles de zoom no mapa.
        - scrollWheelZoom (boolean): Controla de rolagem no mouse.
        - dragging (boolean): Controla de movimentação no mapa.
        
        Retorna:
        - folium.Map: Mapa folium com os dados do GeoDataFrame.
        """
        # Criar um mapa folium centrado nas coordenadas iniciais
        fmap = folium.Map(
            location=initialCoords[::-1], 
            zoom_start=zoomStart, 
            tiles=basemap,
            control_scale=controlScale, 
            zoom_control=zoomControl, 
            scrollWheelZoom=scrollWheelZoom, 
            dragging=dragging)
        return fmap

    @staticmethod
    def addLayer(
        geoDF, 
        styleConfig=None, 
        popupField=None, 
        tooltipField=None):
        """
        Adiciona uma camada de GeoDataFrame ao mapa folium com a simbologia especificada.
        
        Parâmetros:
        - geoDF (GeoDataFrame): GeoDataFrame com os dados geoespaciais.
        - styleConfig (dict): Configuração de estilo para a camada.
        - popupField (str): Nome da coluna para exibir em popups (opcional).
        - tooltipField (str): Nome da coluna para exibir em tooltips (opcional).
        
        Retorna:
        - folium.Map: Objeto de mapa folium com a nova camada adicionada.
        """
        # Configuração padrão de estilo
        defaultStyle = {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 2,
            'fillOpacity': 0.6
        }
        
        # Atualizar configuração de estilo com a fornecida pelo usuário
        if styleConfig:
            defaultStyle.update(styleConfig)
        
        # Converter GeoDataFrame para GeoJSON
        geojson_data = geoDF.to_json()
        
        # Adicionar camada GeoJSON ao mapa
        geojson_layer = GeoJson(
            geojson_data,
            style_function=lambda feature: defaultStyle
        )
        
        # Adicionar popup se especificado
        if popupField:
            popup = GeoJsonPopup(fields=[popupField])
            geojson_layer.add_child(popup)
        
        # Adicionar tooltip se especificado
        if tooltipField:
            tooltip = GeoJsonTooltip(fields=[tooltipField])
            geojson_layer.add_child(tooltip)
        
        return geojson_layer

    @staticmethod
    def removeLayer(fmap, layerName):
        """
        Remove uma camada do mapa folium com base no nome da camada.
        
        Parâmetros:
        - fmap (folium.Map): Objeto de mapa folium.
        - layerName (str): Nome da camada a ser removida.
        
        Retorna:
        - folium.Map: Objeto de mapa folium com a camada removida.
        """
        layers_to_remove = [layer for layer in fmap._children if layer == layer_name]
        for layer in layers_to_remove:
            del fmap._children[layer]
        return fmap
    
    @staticmethod
    def setZoomLevel(fmap, zoomLevel):
        """
        Ajusta o nível de zoom do mapa folium.
        
        Parâmetros:
        - fmap (folium.Map): Objeto de mapa folium.
        - zoom_level (int): Nível de zoom desejado.
        
        Retorna:
        - folium.Map: Objeto de mapa folium com o nível de zoom ajustado.
        """
        fmap.options['zoom'] = zoomLevel
        return fmap

    @staticmethod
    def createSpatialJoin(referenceDF, targetDF):
        """
        Atribui bairros aos registros do DataFrame baseado em latitudes e longitudes.
        
        Parâmetros:
        - referenceDF (DataFrame): DataFrame com as colunas 'LATITUDE' e 'LONGITUDE'.
        - targetDF (GeoDataFrame): GeoDataFrame dos limites dos bairros.
        
        Retorna:
        - DataFrame: DataFrame original com uma nova coluna 'BAIRRO' indicando o bairro de cada registro.
        """
        # Realizar a junção espacial
        joinDF = gpd.sjoin(targetDF, referenceDF, how="left", op='intersects')
        joinDF.drop(columns=['index_right'], inplace=True)
        joinDF.reset_index(drop=True, inplace=True)
        return joinDF

class ChartUtils:
    def createGauge(title, value=50, min=0, max=100, 
                    chartColor="orange", shadownColor="yellow", theme='light'):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            gauge={
                'axis': {'range': [min, max]},
                'bar': {'color': chartColor},
                'steps': [
                    {'range': [min, value], 'color': shadownColor},
                    {'range': [value, max], 'color': "lightgray"}
                ]
            }
        ))
        
        if (theme=='dark'):
            fig.update_layout(
                height=300,
                paper_bgcolor="black",
                plot_bgcolor="black",
                template="plotly_dark",
                # title_font_family='Arial',
                title_text=title,
                title_font_size=20,
                title_font_weight='bold',
                title_xanchor='center',
                title_yanchor='top',
                title_x=0.5,
                title_y=0.9,
            )
        
        return fig

    def getGaugeIndicatorColors(currentValue, cutoff25, cutoff75):
        # Determinando as Cores dos Gráficos
        colorGreen  = {"title":"Normal",  "color": "#4FBA74", "shadown": "#3FA261"}
        colorOrange = {"title":"Atenção", "color": "#FCAB10", "shadown": "#F29E02"}
        colorRed    = {"title":"Alerta",  "color": "#F6131E", "shadown": "#D90812"}

        chartColor   = colorOrange["color"]
        chartShadown = colorOrange["shadown"]
        if currentValue < cutoff25:
            chartColor   = colorGreen["color"]
            chartShadown = colorGreen["shadown"]
        elif currentValue > cutoff75:
            chartColor   = colorRed["color"]
            chartShadown = colorRed["shadown"]
            
        return chartColor, chartShadown
    
    def createRadar(title,
                    dataframe, 
                    fieldClasses, 
                    colors=px.colors.sequential.Plasma_r, 
                    theme='light'):
        if (dataframe.empty == False):
            plotDF = pd.melt(dataframe, id_vars=fieldClasses, var_name='theta', value_name='r')
        else:
            plotDF = pd.DataFrame({
                f'{fieldClasses}': ['','','','','',''],
                'theta': ['Temperatura', 'Umidade', 'Luminosidade', 'Ruído', 'CO₂', 'ETVOC'],
                'r': [0, 0, 0, 0, 0, 0]
            })
            
        fig = px.line_polar(
            plotDF,
            r='r',
            theta='theta',
            title=title,
            color=fieldClasses,
            line_close=True,
            color_discrete_sequence=colors
        )
        
        if (theme=='dark'):
            fig.update_layout(
                height=800,
                paper_bgcolor="black",
                plot_bgcolor="black",
                template="plotly_dark",
                # title_font_family='Arial',
                title_font_size=20,
                title_font_weight='bold',
                title_xanchor='center',
                title_yanchor='top',
                title_x=0.5,
                title_y=0.95,
                # showlegend=False,
                legend_title='LEGENDA',
                legend_orientation='h',
            )
        
        return fig

def checkDayPeriod(hora):
    if 5 <= hora < 12: return 'Manhã'
    elif 12 <= hora < 18: return 'Tarde'
    else: return 'Noite'

def classifyCrime(row):
    if row['Tipo Fato'] == 'Tentado' and 'HOMICIDIO' in row['Desc Fato']:
        return 'Tentativa de Homicídio'
    elif row['Tipo Fato'] == 'Tentado' and 'ROUBO' in row['Desc Fato']:
        return 'Tentativa de Roubo'
    elif row['Tipo Fato'] == 'Consumado' and 'HOMICIDIO' in row['Desc Fato']:
        return 'Homicídio'
    elif row['Tipo Fato'] == 'Consumado' and 'ROUBO' in row['Desc Fato']:
        return 'Roubo'
    else:
        return 'Outros'

# ================ MAIN ================

DF_BAIRROS = DataLoader.loadSHP(DATA_PATH, 'RS_CAXIASDOSUL_BAIRROS')
DF_BAIRROS.drop(columns=['numerolei', 'link_doc_b', 'observacoe',
                         'OBJECTID', 'bairro', 'FREQUENCY', 
                         'MIN_temper', 'MAX_temper', 'MEAN_tempe', 
                         'MIN_umidad', 'MAX_umidad', 'MEAN_umida', 
                         'MIN_lumino', 'MAX_lumino', 'MEAN_lumin',
                         'MIN_ruido', 'MAX_ruido', 'MEAN_ruido', 
                         'MIN_eco2', 'MAX_eco2', 'MEAN_eco2', 
                         'MIN_etvoc', 'MAX_etvoc', 'MEAN_etvoc', 
                         'Shape_Leng', 'Shape_Area'], 
                axis='columns', 
                inplace=True)

# Reprojetando camada de bairros
DF_BAIRROS = DF_BAIRROS.to_crs(crs="EPSG:4326")

# Carregar dados de Monitoramento
DF_AMV_01 = DataLoader.loadCSV(DATA_PATH, 'AMV_01', '|')
DF_AMV_02 = DataLoader.loadCSV(DATA_PATH, 'AMV_02', '|')

# Unificando dataframes de monitoramento
DF_AMV = pd.concat([DF_AMV_01, DF_AMV_02])

# Removendo campos desnecessários
DF_AMV.drop('device', axis='columns', inplace=True)

# Removendo Latitude e Longitude zero
DF_AMV = DF_AMV[(DF_AMV['latitude'] != 0) & (DF_AMV['longitude'] != 0)]

# Geoespacializando pontos de monitoramento
geometry = [Point(xy) for xy in zip(DF_AMV['longitude'], DF_AMV['latitude'])]
DF_AMV = gpd.GeoDataFrame(DF_AMV, geometry=geometry, crs="EPSG:4326")

# Carregar dados de Segurança Pública
DF_SEGURANCA = DataLoader.loadXLSX(DATA_PATH, 'SEGURANCA_PUBLICA')

# Carregar dados de Segurança Pública
DF_SATISFACAO = DataLoader.loadXLSX(DATA_PATH, 'SATISFACAO')

DF_AMV_BAIRRO = MapUtils.createSpatialJoin(referenceDF=DF_BAIRROS,targetDF=DF_AMV)
DF_AMV_BAIRRO.rename(columns={'nome':'bairro'}, inplace=True)

# Padronizando valores da coluna de Bairro
DF_AMV_BAIRRO['bairro'] = DF_AMV_BAIRRO['bairro'].apply(lambda x: unidecode(str(x)).upper())

# Removendo registros de bairro nulos
DF_AMV_BAIRRO = DF_AMV_BAIRRO.dropna(subset=['bairro'])

# Renomeando coluna de BAIRRO utilizada para busca
DF_AMV_BAIRRO.rename(columns={'bairro': 'BAIRRO'}, inplace=True)

# Determinar formato do campo data
DF_AMV_BAIRRO['data'] = pd.to_datetime(DF_AMV_BAIRRO['data'])
DF_AMV_BAIRRO['day_name'] = DF_AMV_BAIRRO['data'].dt.day_name()

# Criar campos de período, data, hora e dia da semana
DF_AMV_BAIRRO['F_PERIODO'] = DF_AMV_BAIRRO['data'].dt.hour.apply(checkDayPeriod)
DF_AMV_BAIRRO['F_HORA'] = DF_AMV_BAIRRO['data'].dt.strftime('%H').astype(int)
DF_AMV_BAIRRO['F_MINUTO'] = DF_AMV_BAIRRO['data'].dt.strftime('%M').astype(int)
DF_AMV_BAIRRO['F_DIA'] = DF_AMV_BAIRRO['data'].dt.strftime('%d').astype(int)
DF_AMV_BAIRRO['F_MES'] = DF_AMV_BAIRRO['data'].dt.strftime('%m').astype(int)
DF_AMV_BAIRRO['F_ANO'] = DF_AMV_BAIRRO['data'].dt.strftime('%Y').astype(int)
DF_AMV_BAIRRO['F_DIA_SEMANA'] = DF_AMV_BAIRRO['day_name'].map(DAY_NAME_MAP)

# Apagar campos de processamento temporários
DF_AMV_BAIRRO.drop(columns=['day_name'], inplace=True)

# Padronizando valores das colunas Bairro e Município
DF_SEGURANCA['Bairro'] = DF_SEGURANCA['Bairro'].apply(lambda x: unidecode(str(x)).upper())
DF_SEGURANCA['Municipio'] = DF_SEGURANCA['Municipio'].apply(lambda x: unidecode(str(x)).upper())

# Determinar formato do campo data
DF_SEGURANCA['datafato'] = DF_SEGURANCA['Data Fato'].dt.strftime('%Y-%m-%d')
DF_SEGURANCA['horafato'] = DF_SEGURANCA['Hora Fato'].astype(str)

DF_SEGURANCA['data'] = pd.to_datetime(DF_SEGURANCA['datafato'] + ' ' + DF_SEGURANCA['horafato'])
DF_SEGURANCA['day_name'] = DF_SEGURANCA['data'].dt.day_name()

# Criar campos de período, data, e hora
DF_SEGURANCA['F_PERIODO'] = DF_AMV_BAIRRO['data'].dt.hour.apply(checkDayPeriod)
DF_SEGURANCA['F_HORA'] = DF_AMV_BAIRRO['data'].dt.strftime('%H').astype(int)
DF_SEGURANCA['F_MINUTO'] = DF_SEGURANCA['data'].dt.strftime('%M').astype(int)
DF_SEGURANCA['F_DIA'] = DF_SEGURANCA['data'].dt.strftime('%d').astype(int)
DF_SEGURANCA['F_MES'] = DF_SEGURANCA['data'].dt.strftime('%m').astype(int)
DF_SEGURANCA['F_ANO'] = DF_SEGURANCA['data'].dt.strftime('%Y').astype(int)
DF_SEGURANCA['F_DIA_SEMANA'] = DF_SEGURANCA['day_name'].map(DAY_NAME_MAP)

# Criar campo de classificação do crime
DF_SEGURANCA['F_CLASSIFICACAO'] = DF_SEGURANCA.apply(classifyCrime, axis=1)

# Apagar campos de processamento temporários
DF_SEGURANCA.drop(columns=['day_name','datafato','horafato'], inplace=True)

# Padronizando valores das colunas Bairro
DF_SATISFACAO['BAIRRO'] = DF_SATISFACAO['BAIRRO'].apply(lambda x: unidecode(str(x)).upper())

DF_SATISFACAO_STATS = DF_SATISFACAO.groupby('BAIRRO').agg({
  'Qtd respostas': 'sum',
  'Satisfação com o bairro': 'mean',
  'Satisfação com a Saúde': 'mean',
  'Prática de atividade física': 'mean',
  'Satisfação financeira': 'mean',
  'Satisfação com atividade comercial': 'mean',
  'Satisfação com qualidade do ar': 'mean',
  'Satisfação com ruído': 'mean',
  'Satisfação com espaços de lazer': 'mean',
  'Satistação com coleta de lixo': 'mean',
  'Satisfação com distância da parada de ônibus': 'mean',
  'Satisfação com qualidade das paradas de ônibus': 'mean',
  'Satisfação com acesso aos locais importantes da cidade': 'mean',
  'Sentimento de segurança': 'mean',
  'Sentimento de confiança nas pessoas': 'mean',
  'Satisfação com tratamento de esgoto': 'mean'
})

# Configurando Filtros
FILTROS = {
  'BAIRRO': DF_AMV_BAIRRO['BAIRRO'].unique(),
  'PERÍODO': DF_AMV_BAIRRO['F_PERIODO'].unique(),
  'HORA': DF_AMV_BAIRRO['F_HORA'].unique(),
  'MINUTO': DF_AMV_BAIRRO['F_MINUTO'].unique(),
  'DIA': DF_AMV_BAIRRO['F_DIA'].unique(),
  'MÊS': DF_AMV_BAIRRO['F_MES'].unique(),
  'ANO': DF_AMV_BAIRRO['F_ANO'].unique(),
  'DIA DA SEMANA': DF_AMV_BAIRRO['F_DIA_SEMANA'].unique(),
}

with st.container():
        st.markdown(
            """
            <style>
            .header-bar {
                background-color: #222222;
                padding: 10px;
                color: white;
                text-align: center;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 15px;
            }
            </style>
            <div class="header-bar">
                Monitoramento Ambiental em Tempo Real
            </div>
            """,
            unsafe_allow_html=True
        )

# FILTROS
FILTRO_BAIRRO = st.sidebar.multiselect('Bairro(s)', options=FILTROS['BAIRRO'], placeholder="Escolha uma opção")
FILTRO_PERIODO = st.sidebar.multiselect('Período(s)', options=FILTROS['PERÍODO'], placeholder="Escolha uma opção")
FILTRO_DIA_SEMANA = st.sidebar.multiselect('Dia(s) da Semana', options=FILTROS['DIA DA SEMANA'], placeholder="Escolha uma opção")

# DATA E HORA (DE)
with st.sidebar.container():
    st.write("### Data e Hora (DE)")
    FILTRO_DIA_DE = st.sidebar.slider(key='DIA_DE', label='Dia', min_value=1, max_value=31, value=1)
    FILTRO_MES_DE = st.sidebar.slider(key='MES_DE', label='Mês', min_value=1, max_value=12, value=1)
    FILTRO_ANO_DE = st.sidebar.slider(key='ANO_DE', label='Ano', min_value=2023, max_value=2024, value=2023)
    FILTRO_HORA_DE = st.sidebar.slider(key='HORA_DE', label='Hora', min_value=0, max_value=23, value=0)
    FILTRO_MINUTO_DE = st.sidebar.slider(key='MIN_DE', label='Minuto', min_value=0, max_value=59, value=0)

# DATA E HORA (ATÉ)
with st.sidebar.container():
    st.write("### Data e Hora (ATÉ)")
    FILTRO_DIA_ATE = st.sidebar.slider(key='DIA_ATE', label='Dia', min_value=1, max_value=31, value=1)
    FILTRO_MES_ATE = st.sidebar.slider(key='MES_ATE', label='Mês', min_value=1, max_value=12, value=1)
    FILTRO_ANO_ATE = st.sidebar.slider(key='ANO_ATE', label='Ano', min_value=2023, max_value=2024, value=2023)
    FILTRO_HORA_ATE = st.sidebar.slider(key='HORA_ATE', label='Hora', min_value=0, max_value=23, value=0)
    FILTRO_MINUTO_ATE = st.sidebar.slider(key='MIN_ATE', label='Minuto', min_value=0, max_value=59, value=0)

# APLICANDO FILTRO
DF_AMV_FILTERED = DF_AMV_BAIRRO[DF_AMV_BAIRRO['BAIRRO'].isin(FILTRO_BAIRRO)]

# TEMPERATURA
TEMPERATURE_MIN = DF_AMV_FILTERED['temperatura'].min()
TEMPERATURE_MAX = DF_AMV_FILTERED['temperatura'].max()
TEMPERATURE_MEAN = DF_AMV_FILTERED['temperatura'].mean()
TEMPERATURE_CUTOFF_25 = TEMPERATURE_MIN + 0.25 * (TEMPERATURE_MAX - TEMPERATURE_MIN)
TEMPERATURE_CUTOFF_75 = TEMPERATURE_MIN + 0.75 * (TEMPERATURE_MAX - TEMPERATURE_MIN)

# UMIDADE
UMIDADE_MIN = DF_AMV_FILTERED['umidade'].min()
UMIDADE_MAX = DF_AMV_FILTERED['umidade'].max()
UMIDADE_MEAN = DF_AMV_FILTERED['umidade'].mean()
UMIDADE_CUTOFF_25 = UMIDADE_MIN + 0.25 * (UMIDADE_MAX - UMIDADE_MIN)
UMIDADE_CUTOFF_75 = UMIDADE_MIN + 0.75 * (UMIDADE_MAX - UMIDADE_MIN)

# LUMINOSIDADE
LUMINOSIDADE_MIN = DF_AMV_FILTERED['luminosidade'].min()
LUMINOSIDADE_MAX = DF_AMV_FILTERED['luminosidade'].max()
LUMINOSIDADE_MEAN = DF_AMV_FILTERED['luminosidade'].mean()
LUMINOSIDADE_CUTOFF_25 = LUMINOSIDADE_MIN + 0.25 * (LUMINOSIDADE_MAX - LUMINOSIDADE_MIN)
LUMINOSIDADE_CUTOFF_75 = LUMINOSIDADE_MIN + 0.75 * (LUMINOSIDADE_MAX - LUMINOSIDADE_MIN)

# RUÍDO
RUIDO_MIN = DF_AMV_FILTERED['ruido'].min()
RUIDO_MAX = DF_AMV_FILTERED['ruido'].max()
RUIDO_MEAN = DF_AMV_FILTERED['ruido'].mean()
RUIDO_CUTOFF_25 = RUIDO_MIN + 0.25 * (RUIDO_MAX - RUIDO_MIN)
RUIDO_CUTOFF_75 = RUIDO_MIN + 0.75 * (RUIDO_MAX - RUIDO_MIN)

# CO2
CO2_MIN = DF_AMV_FILTERED['eco2'].min()
CO2_MAX = DF_AMV_FILTERED['eco2'].max()
CO2_MEAN = DF_AMV_FILTERED['eco2'].mean()
CO2_CUTOFF_25 = CO2_MIN + 0.25 * (CO2_MAX - CO2_MIN)
CO2_CUTOFF_75 = CO2_MIN + 0.75 * (CO2_MAX - CO2_MIN)

# TVOC
TVOC_MIN = DF_AMV_FILTERED['etvoc'].min()
TVOC_MAX = DF_AMV_FILTERED['etvoc'].max()
TVOC_MEAN = DF_AMV_FILTERED['etvoc'].mean()
TVOC_CUTOFF_25 = TVOC_MIN + 0.25 * (TVOC_MAX - TVOC_MIN)
TVOC_CUTOFF_75 = TVOC_MIN + 0.75 * (TVOC_MAX - TVOC_MIN)

indicatorCharts = []

# GRÁFICO DE TEMPERATURA
chartTemperatureColor, chartTemperatureShadown = ChartUtils.getGaugeIndicatorColors(
  TEMPERATURE_MEAN, 
  TEMPERATURE_CUTOFF_25, 
  TEMPERATURE_CUTOFF_75
)
chartTemperature = ChartUtils.createGauge(
  title="Temperatura (°C)",
  value=TEMPERATURE_MEAN,
  min=TEMPERATURE_MIN,
  max=TEMPERATURE_MAX,
  chartColor=f"{chartTemperatureColor}",
  shadownColor=f"{chartTemperatureShadown}",
  theme='dark'
)
indicatorCharts.append(chartTemperature)

# GRÁFICO DE UMIDADE
chartUmidadeColor, chartUmidadeShadown = ChartUtils.getGaugeIndicatorColors(
  UMIDADE_MEAN, 
  UMIDADE_CUTOFF_25, 
  UMIDADE_CUTOFF_75
)
chartUmidade = ChartUtils.createGauge(
  title="Umidade",
  value=UMIDADE_MEAN,
  min=UMIDADE_MIN,
  max=UMIDADE_MAX,
  chartColor=f"{chartUmidadeColor}",
  shadownColor=f"{chartUmidadeShadown}",
  theme='dark'
)
indicatorCharts.append(chartUmidade)

# GRÁFICO DE LUMINOSIDADE
chartLuminosidadeColor, chartLuminosidadeShadown = ChartUtils.getGaugeIndicatorColors(
  LUMINOSIDADE_MEAN, 
  LUMINOSIDADE_CUTOFF_25, 
  LUMINOSIDADE_CUTOFF_75
)
chartLuminosidade = ChartUtils.createGauge(
  title="Luminosidade",
  value=LUMINOSIDADE_MEAN,
  min=LUMINOSIDADE_MIN,
  max=LUMINOSIDADE_MAX,
  chartColor=f"{chartLuminosidadeColor}",
  shadownColor=f"{chartLuminosidadeShadown}",
  theme='dark'
)
indicatorCharts.append(chartLuminosidade)

# GRÁFICO DE RUÍDO
chartRuidoColor, chartRuidoShadown = ChartUtils.getGaugeIndicatorColors(
  RUIDO_MEAN, 
  RUIDO_CUTOFF_25, 
  RUIDO_CUTOFF_75
)
chartRuido = ChartUtils.createGauge(
  title="Ruído",
  value=RUIDO_MEAN,
  min=RUIDO_MIN,
  max=RUIDO_MAX,
  chartColor=f"{chartRuidoColor}",
  shadownColor=f"{chartRuidoShadown}",
  theme='dark'
)
indicatorCharts.append(chartRuido)

# GRÁFICO DE CO2
chartCO2Color, chartCO2Shadown = ChartUtils.getGaugeIndicatorColors(
  CO2_MEAN, 
  CO2_CUTOFF_25, 
  CO2_CUTOFF_75
)
chartCO2 = ChartUtils.createGauge(
  title="CO₂",
  value=CO2_MEAN,
  min=CO2_MIN,
  max=CO2_MAX,
  chartColor=f"{chartCO2Color}",
  shadownColor=f"{chartCO2Shadown}",
  theme='dark'
)
indicatorCharts.append(chartCO2)

# GRÁFICO DE TVOC
chartTVOCColor, chartTVOCShadown = ChartUtils.getGaugeIndicatorColors(
  TVOC_MEAN, 
  TVOC_CUTOFF_25, 
  TVOC_CUTOFF_75
)
chartTVOC = ChartUtils.createGauge(
  title="ETVOC",
  value=TVOC_MEAN,
  min=TVOC_MIN,
  max=TVOC_MAX,
  chartColor=f"{chartTVOCColor}",
  shadownColor=f"{chartTVOCShadown}",
  theme='dark'
)
indicatorCharts.append(chartTVOC)

# Criar subplots
# fig = make_subplots(
#     rows=2, cols=3,
#     specs=[
#         [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
#         [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
#     ]
# )

# Adicionar gráficos aos subplots
# for i, chart in enumerate(indicatorCharts):
#     row = i // 3 + 1
#     col = i % 3 + 1
#     fig.add_trace(chart.data[0], row=row, col=col)

# Atualizar layout
# fig.update_layout(
#     font={'color': "white"},
#     paper_bgcolor="black",
#     plot_bgcolor="black",
#     height=900, 
#     showlegend=False)

# Mostrar gráfico
# st.plotly_chart(fig)

chartCols = st.columns(3)
chartCols[0].plotly_chart(chartTemperature, use_container_width=True)
chartCols[1].plotly_chart(chartUmidade, use_container_width=True)
chartCols[2].plotly_chart(chartLuminosidade, use_container_width=True)

chartCols = st.columns(3)
chartCols[0].plotly_chart(chartRuido, use_container_width=True)
chartCols[1].plotly_chart(chartCO2, use_container_width=True)
chartCols[2].plotly_chart(chartTVOC, use_container_width=True)

DF_TABLE = DF_AMV_FILTERED[[
    'id',
    'data',
    'temperatura', 'umidade', 'luminosidade',
    'ruido', 'eco2', 'etvoc',
    'BAIRRO', 
    'F_PERIODO', 'F_HORA', 'F_MINUTO', 
    'F_DIA', 'F_MES', 'F_ANO', 'F_DIA_SEMANA'
]]

st.dataframe(
    data=DF_TABLE, 
    use_container_width=True, 
    hide_index=True,
    selection_mode="single-row")

COLS_MONITORAMENTO_GROUP = ['BAIRRO']

# Interpolando valores das colunas de indicadores
COLS_MONITORAMENTO_RADAR = [
    'TEMPERATURA', 
    'UMIDADE', 
    'LUMINOSIDADE', 
    'RUIDO', 
    'CO2', 
    'ETVOC'
]

COLS_MONITORAMENTO_N = [
    'N_TEMPERATURA', 
    'N_UMIDADE', 
    'N_LUMINOSIDADE', 
    'N_RUIDO', 
    'N_CO2', 
    'N_ETVOC'
]

DF_AMV_RADAR = DF_AMV_FILTERED[[
    'BAIRRO',
    'temperatura',
    'umidade',
    'luminosidade',
    'ruido',
    'eco2',
    'etvoc'
]]

DF_AMV_RADAR.rename(
  columns={
      'temperatura': 'TEMPERATURA',
      'umidade': 'UMIDADE',
      'luminosidade': 'LUMINOSIDADE',
      'ruido': 'RUIDO',
      'eco2': 'CO2',
      'etvoc': 'ETVOC'},
  inplace=True
)

DF_AMV_RADAR[f'{COLS_MONITORAMENTO_N[0]}'] = DF_AMV_RADAR[f'{COLS_MONITORAMENTO_RADAR[0]}']
DF_AMV_RADAR[f'{COLS_MONITORAMENTO_N[1]}'] = DF_AMV_RADAR[f'{COLS_MONITORAMENTO_RADAR[1]}']
DF_AMV_RADAR[f'{COLS_MONITORAMENTO_N[2]}'] = DF_AMV_RADAR[f'{COLS_MONITORAMENTO_RADAR[2]}']
DF_AMV_RADAR[f'{COLS_MONITORAMENTO_N[3]}'] = DF_AMV_RADAR[f'{COLS_MONITORAMENTO_RADAR[3]}']
DF_AMV_RADAR[f'{COLS_MONITORAMENTO_N[4]}'] = DF_AMV_RADAR[f'{COLS_MONITORAMENTO_RADAR[4]}']
DF_AMV_RADAR[f'{COLS_MONITORAMENTO_N[5]}'] = DF_AMV_RADAR[f'{COLS_MONITORAMENTO_RADAR[5]}']

if(DF_AMV_RADAR.empty == False):
    scaler = MinMaxScaler()
    DF_AMV_RADAR[COLS_MONITORAMENTO_N] = scaler.fit_transform(DF_AMV_RADAR[COLS_MONITORAMENTO_N])

DF_AMV_RADAR_PLOT = DF_AMV_RADAR.groupby(COLS_MONITORAMENTO_GROUP).mean()

DF_AMV_RADAR_PLOT.drop(
    columns=['TEMPERATURA','UMIDADE', 'LUMINOSIDADE', 'RUIDO', 'CO2', 'ETVOC'], 
    axis='columns', 
    inplace=True
)

DF_AMV_RADAR_PLOT.rename(
columns={
    'N_TEMPERATURA': 'TEMPERATURA', 
    'N_UMIDADE': 'UMIDADE', 
    'N_LUMINOSIDADE': 'LUMINOSIDADE', 
    'N_RUIDO': 'RUIDO', 
    'N_CO2': 'CO2', 
    'N_ETVOC': 'ETVOC'},
inplace=True
)

DF_AMV_RADAR_PLOT.reset_index(inplace=True)

chartMonitoramentoBairro = ChartUtils.createRadar(
    title='INDICADORES POR BAIRRO',
    dataframe=DF_AMV_RADAR_PLOT,
    fieldClasses='BAIRRO',
    colors=px.colors.sequential.Jet_r,
    theme='dark',
)

chartCols = st.columns(1)
chartCols[0].plotly_chart(chartMonitoramentoBairro, use_container_width=True)

# Tabela Correspondente ao Gráfico de RADAR
st.dataframe(
    data=DF_AMV_RADAR_PLOT, 
    use_container_width=True, 
    hide_index=True,
    selection_mode="single-row")