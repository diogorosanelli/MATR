import warnings

import folium.map
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
import matplotlib.colors as mcolors

from shapely.geometry import Point
from matplotlib import pyplot as plt
from folium import GeoJson
from folium.features import GeoJsonPopup, GeoJsonTooltip, DivIcon
from folium.plugins import HeatMap, HeatMapWithTime, MarkerCluster, FeatureGroupSubGroup
from unidecode import unidecode
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from streamlit_folium import st_folium, folium_static
from branca.colormap import linear
from branca import colormap
from branca import colormap as cm

# ================ PARÂMETROS ================

ENV = 'PRD' # DEV / PRD

# Paths
BASE_PATH = '/mnt/d/PESSOAL/240319-RS-MATR/source' if (ENV == 'DEV') else '/mount/src/matr/'
DATA_PATH = f'{BASE_PATH}/data'

NRO_CLASSES = 10

# Configurações de Mapa
USE_MAP = False
INITIAL_COORDS = [-51.1794, -29.1678] # Caxias do Sul
BASEMAPS = [
    'Esri.WorldStreetMap',        # 0 
    'Esri.WorldTopoMap',          # 1
    'Esri.WorldImagery',          # 2
    'Esri.WorldGrayCanvas',       # 3
    'OpenTopoMap',                # 4
    'OpenStreetMap',              # 5
    'CartoDB.Positron',           # 6
    'CartoDB.DarkMatter',         # 7
    'CartoDB.Voyager',            # 8
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

st.set_page_config(layout='wide')
make_map_responsive = """
     <style>
        [title~="st.iframe"] { width: 100%}
     </style>
    """
st.markdown(make_map_responsive, unsafe_allow_html=True)

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
        attr = (
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
            'contributors, &copy; <a href="https://cartodb.com/attributions">CartoDB</a>'
        )
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
        layerName=None,
        styleConfig=None, 
        popupField=None, 
        tooltipField=None):
        """
        Adiciona uma camada de GeoDataFrame ao mapa folium com a simbologia especificada.
        
        Parâmetros:
        - geoDF (GeoDataFrame): GeoDataFrame com os dados geoespaciais.
        - layerName (str): Nome da camada.
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
        
        geojson_layer.layer_name = layerName
        
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
        layers_to_remove = [layer for layer in fmap._children if layer == layerName]
        for layer in layers_to_remove:
            del fmap._children[layer]
        return fmap
    
    @staticmethod
    def hasLayer(fmap, layerName):
        """
        Remove uma camada do mapa folium com base no nome da camada.
        
        Parâmetros:
        - fmap (folium.Map): Objeto de mapa folium.
        - layerName (str): Nome da camada a ser removida.
        
        Retorna:
        - folium.Map: Objeto de mapa folium com a camada removida.
        """
        foundedLayers = [layer for layer in fmap._children if layer.find(layerName) >= 0]
        return True if len(foundedLayers) > 0 else False
    
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
    def createSpatialJoin(referenceDF, targetDF, spatialRelation='intersects'):
        """
        Atribui bairros aos registros do DataFrame baseado em latitudes e longitudes.
        
        Parâmetros:
        - referenceDF (DataFrame): DataFrame com as colunas 'LATITUDE' e 'LONGITUDE'.
        - targetDF (GeoDataFrame): GeoDataFrame dos limites dos bairros.
        
        Retorna:
        - DataFrame: DataFrame original com uma nova coluna 'BAIRRO' indicando o bairro de cada registro.
        """
        # Realizar a junção espacial
        joinDF = gpd.sjoin(targetDF, referenceDF, how="left", predicate=spatialRelation)
        joinDF.drop(columns=['index_right'], inplace=True)
        joinDF.reset_index(drop=True, inplace=True)
        return joinDF

class ChartUtils:
    @staticmethod
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
        else:
            fig.update_layout(
                height=300,
                # paper_bgcolor="#FFFFFF",
                # plot_bgcolor="#FFFFFF",
                template="plotly_white",
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

    @staticmethod
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
    
    @staticmethod
    def createRadar(title,
                    dataframe, 
                    fieldClasses, 
                    colors=px.colors.sequential.Turbo, 
                    theme='light'):
        if (dataframe.empty == False):
            plotDF = pd.melt(dataframe, id_vars=fieldClasses, var_name='theta', value_name='r')
        else:
            plotDF = pd.DataFrame({
                f'{fieldClasses}': ['','','','','',''],
                'theta': ['Temperatura', 'Umidade', 'Luminosidade', 'Ruído', 'CO₂', 'ETVOC'],
                'r': [0, 0, 0, 0, 0, 0],
                'label': [0, 0, 0, 0, 0, 0]
            })
        
        plotDF.rename(
            columns={
                f'{fieldClasses}': f'{fieldClasses.title()}',
                'theta': 'Categoria',
                'r': 'Valor'
            },
            inplace=True
        )
        
        # fig = go.Figure()
        # fig.add_trace(
        #     go.Scatterpolar(
        #         r = dataframe['r'],
        #         theta = dataframe['theta'],
        #         mode = 'lines'
        #     )
        # )
        
        fig = px.line_polar(
            plotDF,
            r='Valor',
            theta='Categoria',
            title=title,
            color=f'{fieldClasses.title()}',
            line_close=True,
            color_discrete_sequence=colors,
            markers=True
        )
        
        fig.update_traces(line={'width': 3},fill='toself')
        
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
        else:
            fig.update_layout(
                height=800,
                paper_bgcolor="white",
                plot_bgcolor="white",
                template="plotly_white",
                # title_font_family='Arial',
                title_font_size=20,
                title_font_weight='bold',
                title_xanchor='center',
                title_yanchor='top',
                title_x=0.5,
                title_y=0.95,
                # showlegend=False,
                # legend_title='LEGENDA',
                legend_orientation='h',
                
                polar_angularaxis_color='#000',
                polar_angularaxis_gridcolor='#FFF',
                polar_angularaxis_gridwidth=3,
                polar_angularaxis_griddash='solid',
                polar_angularaxis_linecolor='#AAA',
                polar_angularaxis_linewidth=1,
                polar_angularaxis_tickcolor='#AAA',
                
                polar_radialaxis_color='#000',
                polar_radialaxis_gridcolor='#FFF',
                polar_radialaxis_gridwidth=3,
                polar_radialaxis_griddash='dot',
                polar_radialaxis_linecolor='#FFF',
                
                polar_bgcolor='#F0F0F0',
            )
        
        return fig

class Utils:
  DAY_NAME_MAP = {
    'Monday': 'SEG',
    'Tuesday': 'TER',
    'Wednesday': 'QUA',
    'Thursday': 'QUI',
    'Friday': 'SEX',
    'Saturday': 'SAB',
    'Sunday': 'DOM'
  }
  
  @staticmethod
  def checkDayPeriod(hora):
    if 5 <= hora < 12: return 'Manhã'
    elif 12 <= hora < 18: return 'Tarde'
    else: return 'Noite'

  @staticmethod
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

# Carregar dados de Bairros
DF_BAIRROS_PLG = DataLoader.loadSHP(DATA_PATH, 'RS_CAXIASDOSUL_BAIRROS')
DF_BAIRROS_PLG.drop(
    columns=['numerolei', 'link_doc_b', 'observacoe',
             'OBJECTID', 'bairro', 'FREQUENCY', 
             'MIN_temper', 'MAX_temper', 'MEAN_tempe', 
             'MIN_umidad', 'MAX_umidad', 'MEAN_umida', 
             'MIN_lumino', 'MAX_lumino', 'MEAN_lumin',
             'MIN_ruido', 'MAX_ruido', 'MEAN_ruido', 
             'MIN_eco2', 'MAX_eco2', 'MEAN_eco2', 
             'MIN_etvoc', 'MAX_etvoc', 'MEAN_etvoc', 
             'Shape_Leng', 'Shape_Area'], 
    axis='columns', 
    inplace=True
)
# DF_BAIRROS_PLG.rename(columns={'nome': 'BAIRRO'}, inplace=True)

DF_BAIRROS_PTN = DataLoader.loadSHP(DATA_PATH, 'RS_CAXIASDOSUL_PTN_Bairros')
# DF_BAIRROS_PTN.rename(columns={'nome': 'BAIRRO'}, inplace=True)

# Reprojetando camada de bairros
DF_BAIRROS_PLG = DF_BAIRROS_PLG.to_crs(crs="EPSG:4326")

# Carregar dados de Setores Censitários
DF_SETORES_GEO = DataLoader.loadSHP(DATA_PATH, 'RS_Malha_Preliminar_2022')
DF_SETORES_GEO = DF_SETORES_GEO[DF_SETORES_GEO['NM_MUN'] == 'Caxias do Sul']

# Reprojetando camada de setores censitários
DF_SETORES_GEO = DF_SETORES_GEO.to_crs(crs="EPSG:4326")

# Carregando tabelas de apoio do CENSO
DF_CENSO_RENDA = DataLoader.loadCSV(f'{DATA_PATH}/CENSO_2010', 'DomicilioRenda_RS', ';')
DF_CENSO_RENDA = DF_CENSO_RENDA[['Cod_setor','V002']]
DF_CENSO_RENDA.rename(columns={'V002':'v0008'}, inplace=True)
DF_CENSO_RENDA = DF_CENSO_RENDA[DF_CENSO_RENDA['v0008'] != 'X']
DF_CENSO_RENDA['v0008'] = DF_CENSO_RENDA[['v0008']].astype(int)

DF_CENSO_PES01 = DataLoader.loadCSV(f'{DATA_PATH}/CENSO_2010', 'Pessoa01_RS', ';')
DF_CENSO_PES01 = DF_CENSO_PES01[['Cod_setor','V001']]
DF_CENSO_PES01.rename(columns={'V001':'v0009'}, inplace=True)
DF_CENSO_PES01['v0009'] = DF_CENSO_PES01[['v0009']].astype(int)

# DF_CENSO_PES02 = DataLoader.loadCSV(f'{DATA_PATH}/CENSO_2010', 'Pessoa02_RS', ';')

DF_SETORES_RENDA = pd.concat([DF_SETORES_GEO, DF_CENSO_RENDA], axis=1, join='inner')
DF_SETORES_RENDA.drop(columns=['Cod_setor'], inplace=True)

DF_SETORES = pd.concat([DF_SETORES_RENDA, DF_CENSO_PES01], axis=1, join='inner')
DF_SETORES.drop(columns=['Cod_setor'], inplace=True)

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
# DF_SEGURANCA = DataLoader.loadXLSX(DATA_PATH, 'SEGURANCA_PUBLICA')

DF_SEGURANCA = DataLoader.loadSHP(DATA_PATH, 'RS_CAXIASDOSUL_PTN_SEG_PUB')
DF_SEGURANCA.rename(
    columns={
        'SP_Data_Fa': 'Data Fato',
        'SP_Dia_Sem': 'Data Fato',
        'SP_Dia_Sem': 'Dia Semana Fato', 
        'SP_Hora_Fa': 'Hora Fato', 
        'SP_Desc_Fa': 'Desc Fato', 
        'SP_Tipo_Fa': 'Tipo Fato',
        'SP_Flagran': 'Flagrante',
        'SP_Enderec': 'Endereco', 
        'SP_Nro_End': 'Nro Endereco', 
        'SP_Tipo_Lo': 'Tipo Local', 
        'SP_Bairro': 'Bairro',
        'SP_Municip': 'Municipio'
    },
    inplace=True
)
DF_SEGURANCA.drop(
    columns=['Status', 'Score', 'SP_DAY', 'SP_MONTH', 'SP_YEAR', 'SP_HOUR', 'SP_MIN', 
             'SP_PERIOD', 'SP_WEEKDAY', 'SP_CLASS'], inplace=True)

DF_SEGURANCA['Data Fato'] = pd.to_numeric(DF_SEGURANCA['Data Fato'], errors='coerce')
DF_SEGURANCA = DF_SEGURANCA.dropna(subset=['Data Fato'])
DF_SEGURANCA['Data Fato'] = pd.to_datetime(DF_SEGURANCA['Data Fato'], origin='1899-12-30', unit='D')
DF_SEGURANCA = DF_SEGURANCA.to_crs(crs="EPSG:4326")

DF_SEGURANCA = gpd.GeoDataFrame(DF_SEGURANCA, geometry='geometry')
DF_SEGURANCA['LON'] = DF_SEGURANCA.geometry.x
DF_SEGURANCA['LAT'] = DF_SEGURANCA.geometry.y

# Carregar dados de Satisfação da População
DF_SATISFACAO = DataLoader.loadXLSX(DATA_PATH, 'SATISFACAO')

# Carregar dados de Agregado Setor 2022
DF_CENSO_2022 = DataLoader.loadCSV(DATA_PATH, 'AGREGADO_SETOR_RS',';')
DF_CENSO_2022 = DF_CENSO_2022[DF_CENSO_2022['NM_MUN'] == 'Caxias do Sul']

# MONITORAMENTO AMBIENTAL ← BAIRROS
DF_AMV_BAIRRO = MapUtils.createSpatialJoin(
  referenceDF=DF_BAIRROS_PLG[['geometry','nome']],
  targetDF=DF_AMV)
DF_AMV_BAIRRO.rename(columns={'nome':'bairro'}, inplace=True)

# SETOR CENSITÁRIO ← BAIRROS
DF_SETORES_BAIRROS = MapUtils.createSpatialJoin(
  referenceDF=DF_BAIRROS_PLG[['geometry','nome']],
  targetDF=DF_SETORES)
DF_SETORES_BAIRROS.rename(columns={'nome':'bairro'}, inplace=True)

# Padronizando valores da coluna de Bairro
DF_AMV_BAIRRO['bairro'] = DF_AMV_BAIRRO['bairro'].apply(lambda x: unidecode(str(x)).upper())

# Removendo registros de bairro nulos
DF_AMV_BAIRRO = DF_AMV_BAIRRO.dropna(subset=['bairro'])

# Renomeando coluna de BAIRRO utilizada para busca
DF_AMV_BAIRRO.rename(columns={'bairro': 'BAIRRO'}, inplace=True)

# Eliinnado valores inválidos
DF_AMV_BAIRRO = DF_AMV_BAIRRO[DF_AMV_BAIRRO['BAIRRO'] != 'NAN']

# Determinar formato do campo data
DF_AMV_BAIRRO['data'] = pd.to_datetime(DF_AMV_BAIRRO['data'])
DF_AMV_BAIRRO['day_name'] = DF_AMV_BAIRRO['data'].dt.day_name()

# Criar campos de período, data, hora e dia da semana
DF_AMV_BAIRRO['F_PERIODO'] = DF_AMV_BAIRRO['data'].dt.hour.apply(Utils.checkDayPeriod)
DF_AMV_BAIRRO['F_HORA'] = DF_AMV_BAIRRO['data'].dt.strftime('%H').astype(int)
DF_AMV_BAIRRO['F_MINUTO'] = DF_AMV_BAIRRO['data'].dt.strftime('%M').astype(int)
DF_AMV_BAIRRO['F_DIA'] = DF_AMV_BAIRRO['data'].dt.strftime('%d').astype(int)
DF_AMV_BAIRRO['F_MES'] = DF_AMV_BAIRRO['data'].dt.strftime('%m').astype(int)
DF_AMV_BAIRRO['F_ANO'] = DF_AMV_BAIRRO['data'].dt.strftime('%Y').astype(int)
DF_AMV_BAIRRO['F_DIA_SEMANA'] = DF_AMV_BAIRRO['day_name'].map(Utils.DAY_NAME_MAP)

# Apagar campos de processamento temporários
DF_AMV_BAIRRO.drop(columns=['day_name'], inplace=True)

# Padronizando valores das colunas Bairro e Município
DF_SEGURANCA['Bairro'] = DF_SEGURANCA['Bairro'].apply(lambda x: unidecode(str(x)).upper())
DF_SEGURANCA['Municipio'] = DF_SEGURANCA['Municipio'].apply(lambda x: unidecode(str(x)).upper())

# Determinar formato do campo data
# DF_SEGURANCA['datafato'] = pd.to_datetime(DF_SEGURANCA['Data Fato'], origin='1899-12-30', unit='D')
DF_SEGURANCA['datafato'] = DF_SEGURANCA['Data Fato'].dt.strftime('%Y-%m-%d')
DF_SEGURANCA['horafato'] = DF_SEGURANCA['Hora Fato'].astype(str)

DF_SEGURANCA['data'] = pd.to_datetime(DF_SEGURANCA['datafato'] + ' ' + DF_SEGURANCA['horafato'])
DF_SEGURANCA['day_name'] = DF_SEGURANCA['data'].dt.day_name()

# Criar campos de período, data, e hora
DF_SEGURANCA['F_PERIODO'] = DF_AMV_BAIRRO['data'].dt.hour.apply(Utils.checkDayPeriod)
DF_SEGURANCA['F_HORA'] = DF_AMV_BAIRRO['data'].dt.strftime('%H').astype(int)
DF_SEGURANCA['F_MINUTO'] = DF_SEGURANCA['data'].dt.strftime('%M').astype(int)
DF_SEGURANCA['F_DIA'] = DF_SEGURANCA['data'].dt.strftime('%d').astype(int)
DF_SEGURANCA['F_MES'] = DF_SEGURANCA['data'].dt.strftime('%m').astype(int)
DF_SEGURANCA['F_ANO'] = DF_SEGURANCA['data'].dt.strftime('%Y').astype(int)
DF_SEGURANCA['F_DIA_SEMANA'] = DF_SEGURANCA['day_name'].map(Utils.DAY_NAME_MAP)

# Criar campo de classificação do crime
DF_SEGURANCA['F_CLASSIFICACAO'] = DF_SEGURANCA.apply(Utils.classifyCrime, axis=1)

# Apagar campos de processamento temporários
DF_SEGURANCA.drop(columns=['day_name','datafato','horafato'], inplace=True)

# Renomeando colunas de ligação
DF_SEGURANCA.rename(
  columns={
    "Municipio": "MUNICIPIO",
    "Bairro": "BAIRRO",
  }, 
  inplace=True
)

DF_SEGURANCA.reset_index(drop=True, inplace=True)

# Padronizando valores das colunas Bairro
DF_SATISFACAO['BAIRRO'] = DF_SATISFACAO['BAIRRO'].apply(lambda x: unidecode(str(x)).upper())

# Renomeando colunas de análise
DF_SATISFACAO.rename(
  columns={
    'Qtd respostas': 'QTD_RESP',
    'Satisfação com o bairro': 'Sat_Bairro',
    'Satisfação com a Saúde': 'Sat_Saúde',
    'Prática de atividade física': 'Pratica_Atividade',
    'Satisfação financeira': 'Sat_Financeira',
    'Satisfação com atividade comercial': 'Sat_atv_comercial',
    'Satisfação com qualidade do ar': 'Sat_Qual_Ar',
    'Satisfação com ruído': 'Sat_Ruído',
    'Satisfação com espaços de lazer': 'Sat_Lazer',
    'Satistação com coleta de lixo': 'Sat_col_Lixo',
    'Satisfação com distância da parada de ônibus': 'Sat_dist_bus_stop',
    'Satisfação com qualidade das paradas de ônibus': 'Sat_qual_bus_stop',
    'Satisfação com acesso aos locais importantes da cidade': 'Sat_Acesso',
    'Sentimento de segurança': 'Sent_Segurança',
    'Sentimento de confiança nas pessoas': 'Sent_Conf_Pessoas',
    'Satisfação com tratamento de esgoto': 'Sat_Trat_Esgoto'
  },
  inplace=True
)

# Setores Censitários
DF_SETORES_BAIRROS.rename(
    columns={
        'bairro': 'BAIRRO',
        'v0001': 'Tot_Pessoas',
        'v0002': 'Tot_Domicílios',
        'v0003': 'Tot_Domicílios_pvt',
        'v0004': 'Tot_Domicílios_col',
        'v0005': 'Med_pess_dom_pvt_ocup',
        'v0006': 'Perc_Dom_pvt_ocup',
        'v0007': 'Tot_Dom_pvt_ocup',
        'v0008': 'Renda',
        'v0009': 'Alfabetizados',
    }, 
    inplace=True
)

# ==================== UNIFICANDO INFORMAÇÕES ====================

# ==================== DASHBOARD ====================
DF_AMV_FILTERED = DF_AMV_BAIRRO.copy()
DF_SEG_FILTERED = DF_SEGURANCA.copy()

FILTROS = {
  'BAIRRO': list(sorted(DF_AMV_FILTERED['BAIRRO'].unique())),
  'PERÍODO': list(DF_AMV_FILTERED['F_PERIODO'].unique()),
  'DIA DA SEMANA': list(DF_AMV_FILTERED['F_DIA_SEMANA'].unique()),
  'DIA': list(DF_AMV_FILTERED['F_DIA'].unique()),
  'MES': list(DF_AMV_FILTERED['F_MES'].unique()),
  'ANO': list(DF_AMV_FILTERED['F_ANO'].unique()),
  'HORA': list(DF_AMV_FILTERED['F_HORA'].unique()),
  'MINUTO': list(DF_AMV_FILTERED['F_MINUTO'].unique()),
}

st.markdown(
    """
    <style>
    .header-bar {
        background-color: #CCCCCC;
        padding: 10px;
        color: #000000;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    </style>
    <div class="header-bar">MONITORAMENTO AMBIENTAL EM TEMPO REAL</div>
    """,
    unsafe_allow_html=True
)

st.caption(body="<div style='text-align:center;font-weight:bold;font-size:14pt;color:#000;padding:10px;'>FILTROS<div>", unsafe_allow_html=True)

chartsFilterCols = st.columns(2)
with chartsFilterCols[0]:
    FILTRO_BAIRRO = st.multiselect(label='Bairro(s)', options=FILTROS['BAIRRO'], placeholder="Escolha uma opção",)

with chartsFilterCols[1]:
    daysFilterCols = st.columns(2)
    with daysFilterCols[0]:
        FILTRO_PERIODO = st.multiselect('Período(s)', options=FILTROS['PERÍODO'], placeholder="Escolha uma opção")    
    with daysFilterCols[1]:
        FILTRO_DIA_SEMANA = st.multiselect('Dia(s) da Semana', options=FILTROS['DIA DA SEMANA'], placeholder="Escolha uma opção")

datesFilterCols = st.columns(2)
with datesFilterCols[0]:
    st.caption(body="<div style='text-align:center;font-weight:bold;font-size:12pt;color:#000;padding:5px;'>DATA E HORA INICIAL<div>", unsafe_allow_html=True)
    
    datesFromCols = st.columns(3)
    with datesFromCols[0]:
        FILTRO_DIA_DE = st.number_input(
            key='DIA_DE', 
            label='Dia', 
            min_value=1, max_value=31, 
            value=np.array(FILTROS['DIA']).min())
    with datesFromCols[1]:
        FILTRO_MES_DE = st.number_input(
            key='MES_DE', 
            label='Mês', 
            min_value=1, max_value=12, 
            value=np.array(FILTROS['MES']).min())
    with datesFromCols[2]:
        FILTRO_ANO_DE = st.number_input(
            key='ANO_DE', 
            label='Ano', 
            min_value=2023, max_value=2024, 
            value=np.array(FILTROS['ANO']).min())
    
    timeFromCols = st.columns(2)
    with timeFromCols[0]:
        FILTRO_HORA_DE = st.number_input(
            key='HORA_DE', 
            label='Hora', 
            min_value=0, max_value=23, 
            value=np.array(FILTROS['HORA']).min())
    with timeFromCols[1]:
        FILTRO_MINUTO_DE = st.number_input(
            key='MIN_DE', 
            label='Minuto', 
            min_value=0, max_value=59, 
            value=np.array(FILTROS['MINUTO']).min())

with datesFilterCols[1]:
    st.caption(body="<div style='text-align:center;font-weight:bold;font-size:12pt;color:#000;padding:5px;'>DATA E HORA FINAL<div>", unsafe_allow_html=True)
    
    datesFromCols = st.columns(3)
    with datesFromCols[0]:
        FILTRO_DIA_ATE = st.number_input(
            key='DIA_ATE', 
            label='Dia', 
            min_value=1, max_value=31, 
            value=np.array(FILTROS['DIA']).max())
    with datesFromCols[1]:
        FILTRO_MES_ATE = st.number_input(
            key='MES_ATE', 
            label='Mês', 
            min_value=1, max_value=12, 
            value=np.array(FILTROS['MES']).max())
    with datesFromCols[2]:
        FILTRO_ANO_ATE = st.number_input(
            key='ANO_ATE', 
            label='Ano', 
            min_value=2023, max_value=2024, 
            value=np.array(FILTROS['ANO']).max())
    
    timeFromCols = st.columns(2)
    with timeFromCols[0]:
        FILTRO_HORA_ATE = st.number_input(
            key='HORA_ATE', 
            label='Hora', 
            min_value=0, max_value=23, 
            value=np.array(FILTROS['HORA']).max())
    with timeFromCols[1]:
        FILTRO_MINUTO_ATE = st.number_input(
            key='MIN_ATE', 
            label='Minuto', 
            min_value=0, max_value=59, 
            value=np.array(FILTROS['MINUTO']).max())

# APLICANDO FILTRO
if FILTRO_BAIRRO != []:
    DF_AMV_FILTERED = DF_AMV_FILTERED[DF_AMV_FILTERED['BAIRRO'].isin(FILTRO_BAIRRO)]
    DF_SEG_FILTERED = DF_SEG_FILTERED[DF_SEG_FILTERED['BAIRRO'].isin(FILTRO_BAIRRO)]
if FILTRO_PERIODO != []:
    DF_AMV_FILTERED = DF_AMV_FILTERED[DF_AMV_FILTERED['F_PERIODO'].isin(FILTRO_PERIODO)]
    DF_SEG_FILTERED = DF_SEG_FILTERED[DF_SEG_FILTERED['F_PERIODO'].isin(FILTRO_PERIODO)]
if FILTRO_DIA_SEMANA != []:
    DF_AMV_FILTERED = DF_AMV_FILTERED[DF_AMV_FILTERED['F_DIA_SEMANA'].isin(FILTRO_DIA_SEMANA)]
    DF_SEG_FILTERED = DF_SEG_FILTERED[DF_SEG_FILTERED['F_DIA_SEMANA'].isin(FILTRO_DIA_SEMANA)]

DF_AMV_FILTERED = DF_AMV_FILTERED[
    # DATA / HORA DE
    (DF_AMV_FILTERED['F_DIA'] >= FILTRO_DIA_DE) &
    (DF_AMV_FILTERED['F_MES'] >= FILTRO_MES_DE) &
    (DF_AMV_FILTERED['F_ANO'] >= FILTRO_ANO_DE) &
    (DF_AMV_FILTERED['F_HORA'] >= FILTRO_HORA_DE) &
    (DF_AMV_FILTERED['F_MINUTO'] >= FILTRO_MINUTO_DE) &
    # DATA / HORA ATÉ
    (DF_AMV_FILTERED['F_DIA'] <= FILTRO_DIA_ATE) &
    (DF_AMV_FILTERED['F_MES'] <= FILTRO_MES_ATE) &
    (DF_AMV_FILTERED['F_ANO'] <= FILTRO_ANO_ATE) &
    (DF_AMV_FILTERED['F_HORA'] <= FILTRO_HORA_ATE) &
    (DF_AMV_FILTERED['F_MINUTO'] <= FILTRO_MINUTO_ATE)
]

# UNIFICANDO DADOS
DF_SEGURANCA_GRP = DF_SEG_FILTERED.groupby(['BAIRRO']).size().reset_index(name='NRO_CRIMES')

DF_SATISFACAO_GRP = DF_SATISFACAO.groupby(['BAIRRO']).agg({
  'QTD_RESP': 'sum',
  'Sat_Bairro': 'sum',
  'Sat_Saúde': 'sum',
  'Pratica_Atividade': 'sum',
  'Sat_Financeira': 'sum',
  'Sat_atv_comercial': 'sum',
  'Sat_Qual_Ar': 'sum',
  'Sat_Ruído': 'sum',
  'Sat_Lazer': 'sum',
  'Sat_col_Lixo': 'sum',
  'Sat_dist_bus_stop': 'sum',
  'Sat_qual_bus_stop': 'sum',
  'Sat_Acesso': 'sum',
  'Sent_Segurança': 'sum',
  'Sent_Conf_Pessoas': 'sum',
  'Sat_Trat_Esgoto': 'sum',
}).reset_index()

DF_SETORES_GRP = DF_SETORES_BAIRROS.groupby(['BAIRRO']).agg({
  'Tot_Pessoas': 'sum',
  'Tot_Domicílios': 'sum',
  'Tot_Domicílios_pvt': 'sum',
  'Tot_Domicílios_col': 'sum',
  'Med_pess_dom_pvt_ocup': 'sum',
  'Perc_Dom_pvt_ocup': 'sum',
  'Tot_Dom_pvt_ocup': 'sum',
  'Renda': 'sum',
  'Alfabetizados': 'sum',
}).reset_index()

DF_DATA = DF_AMV_FILTERED.copy()
DF_DATA = DF_DATA.merge(DF_SEGURANCA_GRP, how='left', left_on='BAIRRO', right_on='BAIRRO')
DF_DATA = DF_DATA.merge(DF_SATISFACAO_GRP, how='left', left_on='BAIRRO', right_on='BAIRRO')
DF_DATA = DF_DATA.merge(DF_SETORES_GRP, how='left', left_on='BAIRRO', right_on='BAIRRO')

# TEMPERATURA
TEMPERATURE_MIN = DF_DATA['temperatura'].min() if (FILTRO_BAIRRO != []) else 0
TEMPERATURE_MAX = DF_DATA['temperatura'].max() if (FILTRO_BAIRRO != []) else 1
TEMPERATURE_MEAN = DF_DATA['temperatura'].mean() if (FILTRO_BAIRRO != []) else 0
TEMPERATURE_CUTOFF_25 = (TEMPERATURE_MIN + 0.25 * (TEMPERATURE_MAX - TEMPERATURE_MIN)) if (FILTRO_BAIRRO != []) else 0.25
TEMPERATURE_CUTOFF_75 = (TEMPERATURE_MIN + 0.75 * (TEMPERATURE_MAX - TEMPERATURE_MIN)) if (FILTRO_BAIRRO != []) else 0.75

# UMIDADE
UMIDADE_MIN = DF_DATA['umidade'].min() if (FILTRO_BAIRRO != []) else 0
UMIDADE_MAX = DF_DATA['umidade'].max() if (FILTRO_BAIRRO != []) else 1
UMIDADE_MEAN = DF_DATA['umidade'].mean() if (FILTRO_BAIRRO != []) else 0
UMIDADE_CUTOFF_25 = (UMIDADE_MIN + 0.25 * (UMIDADE_MAX - UMIDADE_MIN)) if (FILTRO_BAIRRO != []) else 0.25
UMIDADE_CUTOFF_75 = (UMIDADE_MIN + 0.75 * (UMIDADE_MAX - UMIDADE_MIN)) if (FILTRO_BAIRRO != []) else 0.75

# LUMINOSIDADE
LUMINOSIDADE_MIN = DF_DATA['luminosidade'].min() if (FILTRO_BAIRRO != []) else 0
LUMINOSIDADE_MAX = DF_DATA['luminosidade'].max() if (FILTRO_BAIRRO != []) else 1
LUMINOSIDADE_MEAN = DF_DATA['luminosidade'].mean() if (FILTRO_BAIRRO != []) else 0
LUMINOSIDADE_CUTOFF_25 = (LUMINOSIDADE_MIN + 0.25 * (LUMINOSIDADE_MAX - LUMINOSIDADE_MIN)) if (FILTRO_BAIRRO != []) else 0.25
LUMINOSIDADE_CUTOFF_75 = (LUMINOSIDADE_MIN + 0.75 * (LUMINOSIDADE_MAX - LUMINOSIDADE_MIN)) if (FILTRO_BAIRRO != []) else 0.75

# RUÍDO
RUIDO_MIN = DF_DATA['ruido'].min() if (FILTRO_BAIRRO != []) else 0
RUIDO_MAX = DF_DATA['ruido'].max() if (FILTRO_BAIRRO != []) else 1
RUIDO_MEAN = DF_DATA['ruido'].mean() if (FILTRO_BAIRRO != []) else 0
RUIDO_CUTOFF_25 = (RUIDO_MIN + 0.25 * (RUIDO_MAX - RUIDO_MIN)) if (FILTRO_BAIRRO != []) else 0.25
RUIDO_CUTOFF_75 = (RUIDO_MIN + 0.75 * (RUIDO_MAX - RUIDO_MIN)) if (FILTRO_BAIRRO != []) else 0.75

# CO2
CO2_MIN = DF_DATA['eco2'].min() if (FILTRO_BAIRRO != []) else 0
CO2_MAX = DF_DATA['eco2'].max() if (FILTRO_BAIRRO != []) else 1
CO2_MEAN = DF_DATA['eco2'].mean() if (FILTRO_BAIRRO != []) else 0
CO2_CUTOFF_25 = (CO2_MIN + 0.25 * (CO2_MAX - CO2_MIN)) if (FILTRO_BAIRRO != []) else 0.25
CO2_CUTOFF_75 = (CO2_MIN + 0.75 * (CO2_MAX - CO2_MIN)) if (FILTRO_BAIRRO != []) else 0.75

# TVOC
TVOC_MIN = DF_DATA['etvoc'].min() if (FILTRO_BAIRRO != []) else 0
TVOC_MAX = DF_DATA['etvoc'].max() if (FILTRO_BAIRRO != []) else 1
TVOC_MEAN = DF_DATA['etvoc'].mean() if (FILTRO_BAIRRO != []) else 0
TVOC_CUTOFF_25 = (TVOC_MIN + 0.25 * (TVOC_MAX - TVOC_MIN)) if (FILTRO_BAIRRO != []) else 0.25
TVOC_CUTOFF_75 = (TVOC_MIN + 0.75 * (TVOC_MAX - TVOC_MIN)) if (FILTRO_BAIRRO != []) else 0.75

st.markdown(
    """
    <style>
    .title-chart-indicator {
        background-color: #CCCCCC;
        padding: 5px;
        color: #000000;
        text-align: center;
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    </style>
    <div class="title-chart-indicator">GRÁFICO DE INDICADORES</div>
    """,
    unsafe_allow_html=True
)

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
    # theme='dark'
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
    # theme='dark'
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
    # theme='dark'
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
    # theme='dark'
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
    # theme='dark'
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
    # theme='dark'
)
indicatorCharts.append(chartTVOC)

chartCols = st.columns(3)
chartCols[0].plotly_chart(chartTemperature, use_container_width=True)
chartCols[1].plotly_chart(chartUmidade, use_container_width=True)
chartCols[2].plotly_chart(chartLuminosidade, use_container_width=True)

chartCols = st.columns(3)
chartCols[0].plotly_chart(chartRuido, use_container_width=True)
chartCols[1].plotly_chart(chartCO2, use_container_width=True)
chartCols[2].plotly_chart(chartTVOC, use_container_width=True)

DF_TABLE = DF_DATA[[
    'BAIRRO', 
    'data',
    'temperatura', 'umidade', 'luminosidade', 'ruido', 'eco2', 'etvoc',
    # 'F_PERIODO', 'F_HORA', 'F_MINUTO', 'F_DIA', 'F_MES', 'F_ANO', 'F_DIA_SEMANA'
]]

DF_TABLE.rename(
    columns={
        'data': 'DATA',
        'temperatura': 'TEMPERATURA', 
        'umidade': 'UMIDADE', 
        'luminosidade': 'LUMINOSIDADE',
        'ruido': 'RUÍDO', 
        'eco2': 'CO₂', 
        'etvoc': 'ETVOC',
    },
    inplace=True
)

st.dataframe(
    data=DF_TABLE, 
    use_container_width=True, 
    hide_index=True,
    selection_mode="single-row")

# ====================== GRÁFICO RADAR ======================
st.markdown(
    """
    <style>
    .title-radar {
        background-color: #222222;
        padding: 5px;
        color: #FFFFFF;
        text-align: center;
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    </style>
    <div class="title-radar">COMPARATIVO DE INDICADORES</div>
    """,
    unsafe_allow_html=True
)

COLS_GROUP_RADAR = ['BAIRRO']
COLS_VALUE_RADAR = [
    'TEMPERATURA', 'UMIDADE', 'LUMINOSIDADE', 'RUIDO', 'CO₂', 'ETVOC',
    'NRO_CRIMES', 
    'Sat_Bairro', 'Sat_Saúde', 'Pratica_Atividade', 'Sat_Financeira', 'Sat_atv_comercial',
    'Sat_Qual_Ar', 'Sat_Ruído', 'Sat_Lazer', 'Sat_col_Lixo', 'Sat_dist_bus_stop',
    'Sat_qual_bus_stop', 'Sat_Acesso', 'Sent_Segurança', 'Sent_Conf_Pessoas', 'Sat_Trat_Esgoto',
    'Tot_Pessoas', 'Tot_Domicílios', 'Tot_Domicílios_pvt', 'Tot_Domicílios_col', 'Med_pess_dom_pvt_ocup', 'Perc_Dom_pvt_ocup', 'Tot_Dom_pvt_ocup', 'Renda', 'Alfabetizados',
]

PROPS_GROUP_RADAR = 'BAIRRO'
PROPS_VALUE_RADAR = st.multiselect(
    label='Variáveis', 
    options=COLS_VALUE_RADAR, 
    placeholder="Selecione as variáveis",
    default=['TEMPERATURA', 'UMIDADE', 'LUMINOSIDADE', 'RUIDO', 'CO₂', 'ETVOC']
)

DF_AMV_RADAR = DF_DATA[[
    'BAIRRO',   
    'temperatura',
    'umidade',
    'luminosidade',
    'ruido',
    'eco2',
    'etvoc',
    'NRO_CRIMES', 
    'QTD_RESP', 
    'Sat_Bairro', 'Sat_Saúde', 'Pratica_Atividade', 'Sat_Financeira', 'Sat_atv_comercial',
    'Sat_Qual_Ar', 'Sat_Ruído', 'Sat_Lazer', 'Sat_col_Lixo', 'Sat_dist_bus_stop', 
    'Sat_qual_bus_stop', 'Sat_Acesso', 'Sent_Segurança', 'Sent_Conf_Pessoas', 'Sat_Trat_Esgoto', 
    'Tot_Pessoas', 'Tot_Domicílios', 'Tot_Domicílios_pvt', 'Tot_Domicílios_col', 'Med_pess_dom_pvt_ocup', 'Perc_Dom_pvt_ocup', 'Tot_Dom_pvt_ocup', 'Renda', 'Alfabetizados',
]]

DF_AMV_RADAR.rename(
    columns={
        'temperatura': 'TEMPERATURA',
        'umidade': 'UMIDADE',
        'luminosidade': 'LUMINOSIDADE',
        'ruido': 'RUIDO',
        'eco2': 'CO₂',
        'etvoc': 'ETVOC',
        # 'NRO_CRIMES': 'Qtd. Crimes',
        # 'Sat_Bairro': 'Satisfação Bairro', 
        # 'Sat_Saúde': 'Satisfação Saúde', 
        # 'Pratica_Atividade': 'Prática de Atividade Física', 
        # 'Sat_Financeira': 'Satisfação Financeira', 
        # 'Sat_atv_comercial': 'Satisfação com Atividade Comercial',
        # 'Sat_Qual_Ar': 'Satisfação com Qualidade do Ar', 
        # 'Sat_Ruído': 'Satisfação com Ruído', 
        # 'Sat_Lazer': 'Satisfação com Espaços de Lazer', 
        # 'Sat_col_Lixo': 'Satisfação com Coleta de Lixo', 
        # 'Sat_dist_bus_stop': 'Satisfação com Distância de Paradas de Ônibus', 
        # 'Sat_qual_bus_stop': 'Satisfação com Qualidade de Paradas de Ônibus', 
        # 'Sat_Acesso': 'Satisfação com Acesso a Locais Importantes da Cidade', 
        # 'Sent_Segurança': 'Sentimento de Segurança', 
        # 'Sent_Conf_Pessoas': 'Sentimento de Confiança nas Pessoas', 
        # 'Sat_Trat_Esgoto': 'Satisfação com Tratamento de Esgoto',
        # 'v0001': 'Total de Pessoas', 
        # 'v0002': 'Total de Domicílios', 
        # 'v0003': 'Total de Domicílios Particulares', 
        # 'v0004': 'Total de Domicílios COletivos', 
        # 'v0005': 'Média de Moradores em Domicílios Particulares Ocupados', 
        # 'v0006': 'Percentual de Domicílios Particulares Ocupados', 
        # 'v0007': 'Total de DomicÍlios Particulares Ocupados'
    },
    inplace=True
)

COLS_V = []
COLS_N = []
if(DF_AMV_RADAR.empty == False and FILTRO_BAIRRO != [] and PROPS_VALUE_RADAR != []):
    for fieldName in PROPS_VALUE_RADAR:
        DF_AMV_RADAR[f'N_{fieldName}'] = DF_AMV_RADAR[f'{fieldName}']
        COLS_N.append(f'N_{fieldName}')
        COLS_V.append(f'{fieldName}')

    scaler = MinMaxScaler()    
    DF_AMV_RADAR[COLS_N] = scaler.fit_transform(DF_AMV_RADAR[COLS_V])

DF_AMV_RADAR_PLOT = DF_AMV_RADAR[[PROPS_GROUP_RADAR] + COLS_N].groupby(PROPS_GROUP_RADAR).mean()
DF_AMV_RADAR_PLOT.rename(columns=lambda x: x[2:] if ('N_' in x) else x, inplace=True)
DF_AMV_RADAR_PLOT.reset_index(inplace=True)

chartMonitoramentoBairro = ChartUtils.createRadar(
    title=f'INDICADORES POR {PROPS_GROUP_RADAR}',
    dataframe=DF_AMV_RADAR_PLOT,
    fieldClasses=PROPS_GROUP_RADAR,
    colors=px.colors.qualitative.Light24,
    # theme='dark',
)

chartCols = st.columns(1)
chartCols[0].plotly_chart(chartMonitoramentoBairro, use_container_width=True)

# ====================== TABELA GRÁFICO RADAR ======================
if(DF_AMV_RADAR.empty == False and FILTRO_BAIRRO != [] and PROPS_VALUE_RADAR != []):
    DF_RADAR_TABLE = DF_AMV_RADAR[[PROPS_GROUP_RADAR] + COLS_V].groupby(PROPS_GROUP_RADAR).mean()
    DF_RADAR_TABLE.reset_index(inplace=True)
    st.dataframe(
        data=DF_RADAR_TABLE, 
        use_container_width=True, 
        hide_index=True,
        selection_mode="single-row")

# ====================== MAPA ======================
st.markdown(
    """
    <style>
    .title-radar {
        background-color: #CCCCCC;
        padding: 5px;
        color: #000000;
        text-align: center;
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    </style>
    <div class="title-radar">MAPA DE INDICADORES</div>
    """,
    unsafe_allow_html=True
)

mapIndicators = MapUtils.createMap(INITIAL_COORDS, 12, BASEMAPS[3], False, True, False, True)

lyrBairrosPLGStyle = {
    'fillColor': '#CCCCCC',    # Sem preenchimento
    'color': '#888',           # Cor da borda cinza
    'weight': 2,               # Espessura da borda
    'fillOpacity': 0.01        # Transparência do preenchimento
}

if(DF_BAIRROS_PLG.empty == False and FILTRO_BAIRRO != [] and PROPS_VALUE_RADAR != []):
    # ===== BAIRROS =====
    DF_BAIRROS_LYR = DF_BAIRROS_PLG[DF_BAIRROS_PLG['nome'].isin(FILTRO_BAIRRO)]
    DF_BAIRROS_LYR.rename(columns={'nome': 'BAIRRO'}, inplace=True)
    DF_BAIRROS_LYR = DF_BAIRROS_LYR.merge(DF_RADAR_TABLE, how='left', left_on='BAIRRO', right_on='BAIRRO')
    DF_BAIRROS_LYR['GEOID'] = DF_BAIRROS_LYR.index.astype(str)
    DF_BAIRROS_LYR['CENTROID'] = DF_BAIRROS_LYR.centroid
    DF_BAIRROS_LYR['LAT'] = DF_BAIRROS_LYR['CENTROID'].map(lambda c: c.y)
    DF_BAIRROS_LYR['LON'] = DF_BAIRROS_LYR['CENTROID'].map(lambda c: c.x)
    DF_BAIRROS_LYR.drop(columns=['CENTROID'], inplace=True)
    MapUtils.addLayer(
        geoDF=DF_BAIRROS_LYR,
        styleConfig=lyrBairrosPLGStyle,
        layerName='Limite de Bairros',
    ).add_to(mapIndicators)
    
    # ===== RÓTULOS =====
    folium.GeoJson(
        DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO'] + PROPS_VALUE_RADAR],
        style_function = lambda feature: {
            'fillColor': '#FFF',
            'color': '#888',
            'weight': 0,
            'fillOpacity': 0.01,
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=['BAIRRO'] + PROPS_VALUE_RADAR,
            # aliases=['Bairro: ']
        ),
        popup=folium.GeoJsonPopup(
            fields=['BAIRRO'] + PROPS_VALUE_RADAR,
            # aliases=['Bairro: ']
        ),
        name="Rótulos",
        show=True,
    ).add_to(mapIndicators, index=0)
    
    # ===== SEGURANÇA PÚBLICA =====
    DF_SEG_LYR = DF_SEG_FILTERED.copy()
    DF_SEG_LYR['GEOID'] = DF_SEG_LYR.index.astype(str)
    locationsSPCLUSTER = list(zip(DF_SEG_LYR['LAT'], DF_SEG_LYR['LON']))
    MarkerCluster(
        locations=locationsSPCLUSTER,
        name='Segurança Pública (Cluster)',
        popups=DF_SEG_LYR['BAIRRO'].tolist(),
        show=False,
    ).add_to(mapIndicators)
    
    crimesIM = {
        'Homicídio': {'color': '#330708', 'radius': 3},
        'Roubo': {'color': '#e87624', 'radius': 3},
        'Tentativa de Homicídio': {'color': '#e84624', 'radius': 3},
        'Tentativa de Roubo': {'color': '#e8a726', 'radius': 3}
    }
    crimesGRPLYR = folium.FeatureGroup(name='Segurança Pública (Localização)')
    crimesLYRS = {
        'Homicídio': FeatureGroupSubGroup(crimesGRPLYR, 'Homicídio'),
        'Roubo': FeatureGroupSubGroup(crimesGRPLYR, 'Roubo'),
        'Tentativa de Homicídio': FeatureGroupSubGroup(crimesGRPLYR, 'Tentativa de Homicídio'),
        'Tentativa de Roubo': FeatureGroupSubGroup(crimesGRPLYR, 'Tentativa de Roubo')
    }
    for idx, row in DF_SEG_LYR.iterrows():
        folium.CircleMarker(
            location=[row['LAT'], row['LON']],
            radius=crimesIM[row['F_CLASSIFICACAO']]['radius'],
            color=crimesIM[row['F_CLASSIFICACAO']]['color'],
            weight=0,
            fill=True,
            fill_color=crimesIM[row['F_CLASSIFICACAO']]['color'],
            fill_opacity=0.75,
        ).add_to(crimesLYRS[row['F_CLASSIFICACAO']])
    
    for crimeLYR in crimesLYRS.values():
        crimeLYR.add_to(crimesGRPLYR)
    
    crimesGRPLYR.show=False
    crimesGRPLYR.add_to(mapIndicators)
    
    symbolClasses = 5
    valuesSYMBOLS = {
        'Temperatura': {
            'type': ['PNT','PLG'],
            'PNT': {
                'color': '#f36364',
                'fill': True,
                'weight': 0,
                'opacity': 0.75,
                'sizes': [4, 8, 12, 16, 24]
            }
        },
        'Umidade': {
            'type': ['PNT','PLG'],
            'PNT': {
                'color': '#8dd0fc',
                'fill': True,
                'weight': 0,
                'opacity': 0.75,
                'sizes': [4, 8, 12, 16, 24]
            }
        },
        'Luminosidade': {
            'type': ['PNT','PLG'],
            'PNT': {
                'color': '#f4d444',
                'fill': True,
                'weight': 0,
                'opacity': 0.75,
                'sizes': [4, 8, 12, 16, 24]
            }
        },
        'Ruido': {
            'type': ['PNT','PLG'],
            'PNT': {
                'color': '#3e3b92',
                'fill': True,
                'weight': 0,
                'opacity': 0.75,
                'sizes': [4, 8, 12, 16, 24]
            }
        },
        'CO₂': {
            'type': ['PNT','PLG'],
            'PNT': {
                'color': '#9bb2e5',
                'fill': True,
                'weight': 0,
                'opacity': 0.75,
                'sizes': [4, 8, 12, 16, 24]
            }
        },
        'ETVOC': {
            'type': ['PNT','PLG'],
            'PNT': {
                'color': '#f74c06',
                'fill': True,
                'weight': 0,
                'opacity': 0.75,
                'sizes': [4, 8, 12, 16, 24]
            }
        },
        'SAT': {
            'type': ['PNT','PLG'],
            'PNT': {
                'color': '#f74c06',
                'fill': True,
                'weight': 0,
                'opacity': 0.75,
                'sizes': [4, 8, 12, 16, 24]
            }
        },
        'v': {
            'type': ['PNT','PLG'],
            'PNT': {
                'color': '#f74c06',
                'fill': True,
                'weight': 0,
                'opacity': 0.75,
                'sizes': [4, 8, 12, 16, 24]
            }
        },
    }
    for PROP in PROPS_VALUE_RADAR:
        # ===== TEMPERATURA =====
        if (PROP == 'TEMPERATURA'):
            if ('PNT' in valuesSYMBOLS['Temperatura']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Temperatura (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','TEMPERATURA','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['TEMPERATURA'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['Temperatura']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['Temperatura']['PNT']['color'],
                        weight=valuesSYMBOLS['Temperatura']['PNT']['weight'],
                        fill=valuesSYMBOLS['Temperatura']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['Temperatura']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['Temperatura']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['Temperatura']['type']):
                CLASS_BINS_TEMPERATURA = np.linspace(
                    DF_BAIRROS_LYR['TEMPERATURA'].min(), 
                    DF_BAIRROS_LYR['TEMPERATURA'].max(), 
                    NRO_CLASSES + 1)
                CMAP_TEMPERATURA = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_TEMPERATURA = mcolors.BoundaryNorm(CLASS_BINS_TEMPERATURA, CMAP_TEMPERATURA.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','TEMPERATURA']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_TEMPERATURA(NORM_TEMPERATURA(feature['properties']['TEMPERATURA']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','TEMPERATURA'],
                    #     aliases=['Bairro: ','Temperatura:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','TEMPERATURA'],
                    #     aliases=['Bairro: ','Temperatura:']
                    # ),
                    name="Temperatura",
                    show=False,
                ).add_to(mapIndicators, index=999)
        
        # ===== UMIDADE =====
        if (PROP == 'UMIDADE'):
            if ('PNT' in valuesSYMBOLS['Umidade']['type']):
                umidadeGRPLYR = folium.FeatureGroup(name='Umidade (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','UMIDADE','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['UMIDADE'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['Umidade']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['Umidade']['PNT']['color'],
                        weight=valuesSYMBOLS['Umidade']['PNT']['weight'],
                        fill=valuesSYMBOLS['Umidade']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['Umidade']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['Umidade']['PNT']['opacity'],
                    ).add_to(umidadeGRPLYR)
                umidadeGRPLYR.show=False
                umidadeGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['Umidade']['type']):
                CLASS_BINS_UMIDADE = np.linspace(
                    DF_BAIRROS_LYR['UMIDADE'].min(), 
                    DF_BAIRROS_LYR['UMIDADE'].max(), 
                    NRO_CLASSES + 1)
                CMAP_UMIDADE = mcolors.LinearSegmentedColormap.from_list('custom', ['#00ee6e', '#0c75e6'], N=NRO_CLASSES)
                NORM_UMIDADE = mcolors.BoundaryNorm(CLASS_BINS_UMIDADE, CMAP_UMIDADE.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','UMIDADE']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_UMIDADE(NORM_UMIDADE(feature['properties']['UMIDADE']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','UMIDADE'],
                    #     aliases=['Bairro: ','Umidade:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','UMIDADE'],
                    #     aliases=['Bairro: ','Umidade:']
                    # ),
                    name="Umidade",
                    show=False,
                ).add_to(mapIndicators, index=999)
            
        # ===== LUMINOSIDADE =====
        if (PROP == 'LUMINOSIDADE'):
            if ('PNT' in valuesSYMBOLS['Luminosidade']['type']):
                luminosidadeGRPLYR = folium.FeatureGroup(name='Luminosidade (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','LUMINOSIDADE','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['LUMINOSIDADE'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['Luminosidade']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['Luminosidade']['PNT']['color'],
                        weight=valuesSYMBOLS['Luminosidade']['PNT']['weight'],
                        fill=valuesSYMBOLS['Luminosidade']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['Luminosidade']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['Luminosidade']['PNT']['opacity'],
                    ).add_to(luminosidadeGRPLYR)
                luminosidadeGRPLYR.show=False
                luminosidadeGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['Luminosidade']['type']):
                CLASS_BINS_LUMINOSIDADE = np.linspace(
                    DF_BAIRROS_LYR['LUMINOSIDADE'].min(), 
                    DF_BAIRROS_LYR['LUMINOSIDADE'].max(), 
                    NRO_CLASSES + 1)
                CMAP_LUMINOSIDADE = mcolors.LinearSegmentedColormap.from_list('custom', ['#f7f2ab', '#bda734'], N=NRO_CLASSES)
                NORM_LUMINOSIDADE = mcolors.BoundaryNorm(CLASS_BINS_LUMINOSIDADE, CMAP_LUMINOSIDADE.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','LUMINOSIDADE']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_LUMINOSIDADE(NORM_LUMINOSIDADE(feature['properties']['LUMINOSIDADE']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','LUMINOSIDADE'],
                    #     aliases=['Bairro: ','Luminosidade:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','LUMINOSIDADE'],
                    #     aliases=['Bairro: ','Luminosidade:']
                    # ),
                    name="Luminosidade",
                    show=False,
                ).add_to(mapIndicators, index=999)
                
        # ===== RUIDO =====
        if (PROP == 'RUIDO'):
            if ('PNT' in valuesSYMBOLS['Ruido']['type']):
                ruidoGRPLYR = folium.FeatureGroup(name='Ruido (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','LUMINOSIDADE','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['LUMINOSIDADE'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['Ruido']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['Ruido']['PNT']['color'],
                        weight=valuesSYMBOLS['Ruido']['PNT']['weight'],
                        fill=valuesSYMBOLS['Ruido']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['Ruido']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['Ruido']['PNT']['opacity'],
                    ).add_to(ruidoGRPLYR)
                ruidoGRPLYR.show=False
                ruidoGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['Ruido']['type']):
                CLASS_BINS_RUIDO = np.linspace(
                    DF_BAIRROS_LYR['RUIDO'].min(), 
                    DF_BAIRROS_LYR['RUIDO'].max(), 
                    NRO_CLASSES + 1)
                CMAP_RUIDO = mcolors.LinearSegmentedColormap.from_list('custom', ['#6d90b9', '#bbc7dc'], N=NRO_CLASSES)
                NORM_RUIDO = mcolors.BoundaryNorm(CLASS_BINS_RUIDO, CMAP_RUIDO.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','RUIDO']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_RUIDO(NORM_RUIDO(feature['properties']['RUIDO']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','RUIDO'],
                    #     aliases=['Bairro: ','Ruído:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','RUIDO'],
                    #     aliases=['Bairro: ','Ruído:']
                    # ),
                    name="Ruído",
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== CO₂ =====
        if (PROP == 'CO₂'):
            if ('PNT' in valuesSYMBOLS['CO₂']['type']):
                co2GRPLYR = folium.FeatureGroup(name='CO₂ (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','CO₂','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['CO₂'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['CO₂']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['CO₂']['PNT']['color'],
                        weight=valuesSYMBOLS['CO₂']['PNT']['weight'],
                        fill=valuesSYMBOLS['CO₂']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['CO₂']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['CO₂']['PNT']['opacity'],
                    ).add_to(co2GRPLYR)
                co2GRPLYR.show=False
                co2GRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['CO₂']['type']):
                CLASS_BINS_CO2 = np.linspace(
                    DF_BAIRROS_LYR['CO₂'].min(), 
                    DF_BAIRROS_LYR['CO₂'].max(), 
                    NRO_CLASSES + 1)
                CMAP_CO2 = mcolors.LinearSegmentedColormap.from_list('custom', ['#6d90b9', '#bbc7dc'], N=NRO_CLASSES)
                NORM_CO2 = mcolors.BoundaryNorm(CLASS_BINS_CO2, CMAP_CO2.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','CO₂']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_CO2(NORM_CO2(feature['properties']['CO₂']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','CO₂'],
                    #     aliases=['Bairro: ','CO₂:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','CO₂'],
                    #     aliases=['Bairro: ','CO₂:']
                    # ),
                    name="CO₂",
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== ETVOC =====
        if (PROP == 'ETVOC'):
            if ('PNT' in valuesSYMBOLS['ETVOC']['type']):
                etvocGRPLYR = folium.FeatureGroup(name='ETVOC (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','ETVOC','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['ETVOC'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['ETVOC']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['ETVOC']['PNT']['color'],
                        weight=valuesSYMBOLS['ETVOC']['PNT']['weight'],
                        fill=valuesSYMBOLS['ETVOC']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['ETVOC']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['ETVOC']['PNT']['opacity'],
                    ).add_to(etvocGRPLYR)
                etvocGRPLYR.show=False
                etvocGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['ETVOC']['type']):
                CLASS_BINS_ETVOC = np.linspace(
                    DF_BAIRROS_LYR['ETVOC'].min(), 
                    DF_BAIRROS_LYR['ETVOC'].max(), 
                    NRO_CLASSES + 1)
                CMAP_ETVOC = mcolors.LinearSegmentedColormap.from_list('custom', ['#f74c06', '#f9bc2c'], N=NRO_CLASSES)
                NORM_ETVOC = mcolors.BoundaryNorm(CLASS_BINS_ETVOC, CMAP_ETVOC.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','ETVOC']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_ETVOC(NORM_ETVOC(feature['properties']['ETVOC']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','ETVOC'],
                    #     aliases=['Bairro: ','ETVOC:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','ETVOC'],
                    #     aliases=['Bairro: ','ETVOC:']
                    # ),
                    name="ETVOC",
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== Sat_Bairro =====
        if (PROP == 'Sat_Bairro'):
            if ('PNT' in valuesSYMBOLS['SAT']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Sat_Bairro (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Sat_Bairro','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Sat_Bairro'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['SAT']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['SAT']['PNT']['color'],
                        weight=valuesSYMBOLS['SAT']['PNT']['weight'],
                        fill=valuesSYMBOLS['SAT']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['SAT']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['SAT']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['SAT']['type']):
                CLASS_BINS_SAT01 = np.linspace(
                    DF_BAIRROS_LYR['Sat_Bairro'].min(), 
                    DF_BAIRROS_LYR['Sat_Bairro'].max(), 
                    NRO_CLASSES + 1)
                CMAP_SAT01 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_SAT01 = mcolors.BoundaryNorm(CLASS_BINS_SAT01, CMAP_SAT01.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Sat_Bairro']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_SAT01(NORM_SAT01(feature['properties']['Sat_Bairro']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Sat_Bairro'],
                    #     aliases=['Bairro: ','Sat_Bairro:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Sat_Bairro'],
                    #     aliases=['Bairro: ','Sat_Bairro:']
                    # ),
                    name='Sat_Bairro',
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== Sat_Saúde =====
        if (PROP == 'Sat_Saúde'):
            if ('PNT' in valuesSYMBOLS['SAT']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Sat_Saúde (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Sat_Saúde','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Sat_Saúde'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['SAT']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['SAT']['PNT']['color'],
                        weight=valuesSYMBOLS['SAT']['PNT']['weight'],
                        fill=valuesSYMBOLS['SAT']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['SAT']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['SAT']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['SAT']['type']):
                CLASS_BINS_SAT02 = np.linspace(
                    DF_BAIRROS_LYR['Sat_Saúde'].min(), 
                    DF_BAIRROS_LYR['Sat_Saúde'].max(), 
                    NRO_CLASSES + 1)
                CMAP_SAT02 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_SAT02 = mcolors.BoundaryNorm(CLASS_BINS_SAT02, CMAP_SAT02.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Sat_Saúde']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_SAT02(NORM_SAT02(feature['properties']['Sat_Saúde']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Sat_Saúde'],
                    #     aliases=['Bairro: ','Sat_Saúde:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Sat_Saúde'],
                    #     aliases=['Bairro: ','Sat_Saúde:']
                    # ),
                    name='Sat_Saúde',
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== Pratica_Atividade =====
        if (PROP == 'Pratica_Atividade'):
            if ('PNT' in valuesSYMBOLS['SAT']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Pratica_Atividade (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Pratica_Atividade','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Pratica_Atividade'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['SAT']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['SAT']['PNT']['color'],
                        weight=valuesSYMBOLS['SAT']['PNT']['weight'],
                        fill=valuesSYMBOLS['SAT']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['SAT']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['SAT']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['SAT']['type']):
                CLASS_BINS_SAT03 = np.linspace(
                    DF_BAIRROS_LYR['Pratica_Atividade'].min(), 
                    DF_BAIRROS_LYR['Pratica_Atividade'].max(), 
                    NRO_CLASSES + 1)
                CMAP_SAT03 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_SAT03 = mcolors.BoundaryNorm(CLASS_BINS_SAT03, CMAP_SAT03.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Pratica_Atividade']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_SAT03(NORM_SAT03(feature['properties']['Pratica_Atividade']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Pratica_Atividade'],
                    #     aliases=['Bairro: ','Pratica_Atividade:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Pratica_Atividade'],
                    #     aliases=['Bairro: ','Pratica_Atividade:']
                    # ),
                    name='Pratica_Atividade',
                    show=False,
                ).add_to(mapIndicators, index=999)
        
        # ===== Sat_Financeira =====
        if (PROP == 'Sat_Financeira'):
            if ('PNT' in valuesSYMBOLS['SAT']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Sat_Financeira (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Sat_Financeira','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Sat_Financeira'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['SAT']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['SAT']['PNT']['color'],
                        weight=valuesSYMBOLS['SAT']['PNT']['weight'],
                        fill=valuesSYMBOLS['SAT']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['SAT']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['SAT']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['SAT']['type']):
                CLASS_BINS_SAT04 = np.linspace(
                    DF_BAIRROS_LYR['Sat_Financeira'].min(), 
                    DF_BAIRROS_LYR['Sat_Financeira'].max(), 
                    NRO_CLASSES + 1)
                CMAP_SAT04 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_SAT04 = mcolors.BoundaryNorm(CLASS_BINS_SAT04, CMAP_SAT04.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Sat_Financeira']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_SAT01(NORM_SAT04(feature['properties']['Sat_Financeira']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Sat_Financeira'],
                    #     aliases=['Bairro: ','Sat_Financeira:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Sat_Financeira'],
                    #     aliases=['Bairro: ','Sat_Financeira:']
                    # ),
                    name='Sat_Financeira',
                    show=False,
                ).add_to(mapIndicators, index=999)
            
        # ===== Sat_atv_comercial =====
        if (PROP == 'Sat_atv_comercial'):
            if ('PNT' in valuesSYMBOLS['SAT']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Sat_atv_comercial (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Sat_atv_comercial','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Sat_atv_comercial'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['SAT']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['SAT']['PNT']['color'],
                        weight=valuesSYMBOLS['SAT']['PNT']['weight'],
                        fill=valuesSYMBOLS['SAT']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['SAT']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['SAT']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['SAT']['type']):
                CLASS_BINS_SAT05 = np.linspace(
                    DF_BAIRROS_LYR['Sat_atv_comercial'].min(), 
                    DF_BAIRROS_LYR['Sat_atv_comercial'].max(), 
                    NRO_CLASSES + 1)
                CMAP_SAT05 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_SAT05 = mcolors.BoundaryNorm(CLASS_BINS_SAT05, CMAP_SAT05.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Sat_atv_comercial']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_SAT01(NORM_SAT05(feature['properties']['Sat_atv_comercial']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Sat_atv_comercial'],
                    #     aliases=['Bairro: ','Sat_atv_comercial:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Sat_atv_comercial'],
                    #     aliases=['Bairro: ','Sat_atv_comercial:']
                    # ),
                    name='Sat_atv_comercial',
                    show=False,
                ).add_to(mapIndicators, index=999)
            
        # ===== Sat_Qual_Ar =====
        if (PROP == 'Sat_Qual_Ar'):
            if ('PNT' in valuesSYMBOLS['SAT']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Sat_Qual_Ar (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Sat_Qual_Ar','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Sat_Qual_Ar'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['SAT']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['SAT']['PNT']['color'],
                        weight=valuesSYMBOLS['SAT']['PNT']['weight'],
                        fill=valuesSYMBOLS['SAT']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['SAT']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['SAT']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['SAT']['type']):
                CLASS_BINS_SAT06 = np.linspace(
                    DF_BAIRROS_LYR['Sat_Qual_Ar'].min(), 
                    DF_BAIRROS_LYR['Sat_Qual_Ar'].max(), 
                    NRO_CLASSES + 1)
                CMAP_SAT06 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_SAT06 = mcolors.BoundaryNorm(CLASS_BINS_SAT06, CMAP_SAT06.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Sat_Qual_Ar']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_SAT06(NORM_SAT06(feature['properties']['Sat_Qual_Ar']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Sat_Qual_Ar'],
                    #     aliases=['Bairro: ','Sat_Qual_Ar:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Sat_Qual_Ar'],
                    #     aliases=['Bairro: ','Sat_Qual_Ar:']
                    # ),
                    name='Sat_Qual_Ar',
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== Sat_Ruído =====
        if (PROP == 'Sat_Ruído'):
            if ('PNT' in valuesSYMBOLS['SAT']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Sat_Ruído (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Sat_Ruído','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Sat_Ruído'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['SAT']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['SAT']['PNT']['color'],
                        weight=valuesSYMBOLS['SAT']['PNT']['weight'],
                        fill=valuesSYMBOLS['SAT']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['SAT']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['SAT']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['SAT']['type']):
                CLASS_BINS_SAT07 = np.linspace(
                    DF_BAIRROS_LYR['Sat_Ruído'].min(), 
                    DF_BAIRROS_LYR['Sat_Ruído'].max(), 
                    NRO_CLASSES + 1)
                CMAP_SAT07 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_SAT07 = mcolors.BoundaryNorm(CLASS_BINS_SAT07, CMAP_SAT07.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Sat_Ruído']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_SAT07(NORM_SAT07(feature['properties']['Sat_Ruído']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Sat_Ruído'],
                    #     aliases=['Bairro: ','Sat_Ruído:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Sat_Ruído'],
                    #     aliases=['Bairro: ','Sat_Ruído:']
                    # ),
                    name='Sat_Ruído',
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== Sat_Lazer =====
        if (PROP == 'Sat_Lazer'):
            if ('PNT' in valuesSYMBOLS['SAT']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Sat_Lazer (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Sat_Lazer','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Sat_Lazer'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['SAT']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['SAT']['PNT']['color'],
                        weight=valuesSYMBOLS['SAT']['PNT']['weight'],
                        fill=valuesSYMBOLS['SAT']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['SAT']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['SAT']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['SAT']['type']):
                CLASS_BINS_SAT08 = np.linspace(
                    DF_BAIRROS_LYR['Sat_Lazer'].min(), 
                    DF_BAIRROS_LYR['Sat_Lazer'].max(), 
                    NRO_CLASSES + 1)
                CMAP_SAT08 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_SAT08 = mcolors.BoundaryNorm(CLASS_BINS_SAT08, CMAP_SAT08.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Sat_Lazer']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_SAT08(NORM_SAT08(feature['properties']['Sat_Lazer']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Sat_Lazer'],
                    #     aliases=['Bairro: ','Sat_Lazer:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Sat_Lazer'],
                    #     aliases=['Bairro: ','Sat_Lazer:']
                    # ),
                    name='Sat_Lazer',
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== Sat_col_Lixo =====
        if (PROP == 'Sat_col_Lixo'):
            if ('PNT' in valuesSYMBOLS['SAT']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Sat_col_Lixo (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Sat_col_Lixo','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Sat_col_Lixo'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['SAT']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['SAT']['PNT']['color'],
                        weight=valuesSYMBOLS['SAT']['PNT']['weight'],
                        fill=valuesSYMBOLS['SAT']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['SAT']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['SAT']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['SAT']['type']):
                CLASS_BINS_SAT09 = np.linspace(
                    DF_BAIRROS_LYR['Sat_col_Lixo'].min(), 
                    DF_BAIRROS_LYR['Sat_col_Lixo'].max(), 
                    NRO_CLASSES + 1)
                CMAP_SAT09 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_SAT09 = mcolors.BoundaryNorm(CLASS_BINS_SAT09, CMAP_SAT09.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Sat_col_Lixo']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_SAT09(NORM_SAT09(feature['properties']['Sat_col_Lixo']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Sat_col_Lixo'],
                    #     aliases=['Bairro: ','Sat_col_Lixo:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Sat_col_Lixo'],
                    #     aliases=['Bairro: ','Sat_col_Lixo:']
                    # ),
                    name='Sat_col_Lixo',
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== Sat_dist_bus_stop =====
        if (PROP == 'Sat_dist_bus_stop'):
            if ('PNT' in valuesSYMBOLS['SAT']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Sat_dist_bus_stop (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Sat_dist_bus_stop','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Sat_dist_bus_stop'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['SAT']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['SAT']['PNT']['color'],
                        weight=valuesSYMBOLS['SAT']['PNT']['weight'],
                        fill=valuesSYMBOLS['SAT']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['SAT']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['SAT']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['SAT']['type']):
                CLASS_BINS_SAT10 = np.linspace(
                    DF_BAIRROS_LYR['Sat_dist_bus_stop'].min(), 
                    DF_BAIRROS_LYR['Sat_dist_bus_stop'].max(), 
                    NRO_CLASSES + 1)
                CMAP_SAT10 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_SAT10 = mcolors.BoundaryNorm(CLASS_BINS_SAT10, CMAP_SAT10.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Sat_dist_bus_stop']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_SAT10(NORM_SAT10(feature['properties']['Sat_dist_bus_stop']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Sat_dist_bus_stop'],
                    #     aliases=['Bairro: ','Sat_dist_bus_stop:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Sat_dist_bus_stop'],
                    #     aliases=['Bairro: ','Sat_dist_bus_stop:']
                    # ),
                    name='Sat_dist_bus_stop',
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== Sat_qual_bus_stop =====
        if (PROP == 'Sat_qual_bus_stop'):
            if ('PNT' in valuesSYMBOLS['SAT']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Sat_qual_bus_stop (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Sat_qual_bus_stop','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Sat_qual_bus_stop'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['SAT']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['SAT']['PNT']['color'],
                        weight=valuesSYMBOLS['SAT']['PNT']['weight'],
                        fill=valuesSYMBOLS['SAT']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['SAT']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['SAT']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['SAT']['type']):
                CLASS_BINS_SAT11 = np.linspace(
                    DF_BAIRROS_LYR['Sat_qual_bus_stop'].min(), 
                    DF_BAIRROS_LYR['Sat_qual_bus_stop'].max(), 
                    NRO_CLASSES + 1)
                CMAP_SAT11 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_SAT11 = mcolors.BoundaryNorm(CLASS_BINS_SAT11, CMAP_SAT11.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Sat_qual_bus_stop']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_SAT11(NORM_SAT11(feature['properties']['Sat_qual_bus_stop']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Sat_qual_bus_stop'],
                    #     aliases=['Bairro: ','Sat_qual_bus_stop:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Sat_qual_bus_stop'],
                    #     aliases=['Bairro: ','Sat_qual_bus_stop:']
                    # ),
                    name='Sat_qual_bus_stop',
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== Sat_Acesso =====
        if (PROP == 'Sat_Acesso'):
            if ('PNT' in valuesSYMBOLS['SAT']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Sat_Acesso (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Sat_Acesso','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Sat_Acesso'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['SAT']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['SAT']['PNT']['color'],
                        weight=valuesSYMBOLS['SAT']['PNT']['weight'],
                        fill=valuesSYMBOLS['SAT']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['SAT']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['SAT']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['SAT']['type']):
                CLASS_BINS_SAT12 = np.linspace(
                    DF_BAIRROS_LYR['Sat_Acesso'].min(), 
                    DF_BAIRROS_LYR['Sat_Acesso'].max(), 
                    NRO_CLASSES + 1)
                CMAP_SAT12 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_SAT12 = mcolors.BoundaryNorm(CLASS_BINS_SAT12, CMAP_SAT12.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Sat_Acesso']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_SAT12(NORM_SAT12(feature['properties']['Sat_Acesso']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Sat_Acesso'],
                    #     aliases=['Bairro: ','Sat_Acesso:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Sat_Acesso'],
                    #     aliases=['Bairro: ','Sat_Acesso:']
                    # ),
                    name='Sat_Acesso',
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== Sent_Segurança =====
        if (PROP == 'Sent_Segurança'):
            if ('PNT' in valuesSYMBOLS['SAT']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Sent_Segurança (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Sent_Segurança','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Sent_Segurança'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['SAT']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['SAT']['PNT']['color'],
                        weight=valuesSYMBOLS['SAT']['PNT']['weight'],
                        fill=valuesSYMBOLS['SAT']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['SAT']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['SAT']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['SAT']['type']):
                CLASS_BINS_SAT13 = np.linspace(
                    DF_BAIRROS_LYR['Sent_Segurança'].min(), 
                    DF_BAIRROS_LYR['Sent_Segurança'].max(), 
                    NRO_CLASSES + 1)
                CMAP_SAT13 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_SAT13 = mcolors.BoundaryNorm(CLASS_BINS_SAT13, CMAP_SAT13.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Sent_Segurança']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_SAT13(NORM_SAT13(feature['properties']['Sent_Segurança']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Sent_Segurança'],
                    #     aliases=['Bairro: ','Sent_Segurança:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Sent_Segurança'],
                    #     aliases=['Bairro: ','Sent_Segurança:']
                    # ),
                    name='Sent_Segurança',
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== Sent_Conf_Pessoas =====
        if (PROP == 'Sent_Conf_Pessoas'):
            if ('PNT' in valuesSYMBOLS['SAT']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Sent_Conf_Pessoas (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Sent_Conf_Pessoas','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Sent_Conf_Pessoas'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['SAT']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['SAT']['PNT']['color'],
                        weight=valuesSYMBOLS['SAT']['PNT']['weight'],
                        fill=valuesSYMBOLS['SAT']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['SAT']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['SAT']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['SAT']['type']):
                CLASS_BINS_SAT14 = np.linspace(
                    DF_BAIRROS_LYR['Sent_Conf_Pessoas'].min(), 
                    DF_BAIRROS_LYR['Sent_Conf_Pessoas'].max(), 
                    NRO_CLASSES + 1)
                CMAP_SAT14 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_SAT14 = mcolors.BoundaryNorm(CLASS_BINS_SAT14, CMAP_SAT14.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Sent_Conf_Pessoas']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_SAT14(NORM_SAT14(feature['properties']['Sent_Conf_Pessoas']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Sent_Conf_Pessoas'],
                    #     aliases=['Bairro: ','Sent_Conf_Pessoas:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Sent_Conf_Pessoas'],
                    #     aliases=['Bairro: ','Sent_Conf_Pessoas:']
                    # ),
                    name='Sent_Conf_Pessoas',
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== Sat_Trat_Esgoto =====
        if (PROP == 'Sat_Trat_Esgoto'):
            if ('PNT' in valuesSYMBOLS['SAT']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Sat_Trat_Esgoto (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Sat_Trat_Esgoto','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Sat_Trat_Esgoto'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['SAT']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['SAT']['PNT']['color'],
                        weight=valuesSYMBOLS['SAT']['PNT']['weight'],
                        fill=valuesSYMBOLS['SAT']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['SAT']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['SAT']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['SAT']['type']):
                CLASS_BINS_SAT15 = np.linspace(
                    DF_BAIRROS_LYR['Sat_Trat_Esgoto'].min(), 
                    DF_BAIRROS_LYR['Sat_Trat_Esgoto'].max(), 
                    NRO_CLASSES + 1)
                CMAP_SAT15 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_SAT15 = mcolors.BoundaryNorm(CLASS_BINS_SAT15, CMAP_SAT15.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Sat_Trat_Esgoto']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_SAT15(NORM_SAT15(feature['properties']['Sat_Trat_Esgoto']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Sat_Trat_Esgoto'],
                    #     aliases=['Bairro: ','Sat_Trat_Esgoto:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Sat_Trat_Esgoto'],
                    #     aliases=['Bairro: ','Sat_Trat_Esgoto:']
                    # ),
                    name='Sat_Trat_Esgoto',
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== Tot_Pessoas =====
        if (PROP == 'Tot_Pessoas'):
            if ('PNT' in valuesSYMBOLS['v']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Tot_Pessoas (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Tot_Pessoas','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Tot_Pessoas'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['v']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['v']['PNT']['color'],
                        weight=valuesSYMBOLS['v']['PNT']['weight'],
                        fill=valuesSYMBOLS['v']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['v']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['v']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['v']['type']):
                CLASS_BINS_v0001 = np.linspace(
                    DF_BAIRROS_LYR['Tot_Pessoas'].min(), 
                    DF_BAIRROS_LYR['Tot_Pessoas'].max(), 
                    NRO_CLASSES + 1)
                CMAP_v0001 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_v0001 = mcolors.BoundaryNorm(CLASS_BINS_v0001, CMAP_v0001.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Tot_Pessoas']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_v0001(NORM_v0001(feature['properties']['Tot_Pessoas']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Tot_Pessoas'],
                    #     aliases=['Bairro: ','Tot_Pessoas:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Tot_Pessoas'],
                    #     aliases=['Bairro: ','Tot_Pessoas:']
                    # ),
                    name='Tot_Pessoas',
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== Tot_Domicílios =====
        if (PROP == 'Tot_Domicílios'):
            if ('PNT' in valuesSYMBOLS['v']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Tot_Domicílios (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Tot_Domicílios','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Tot_Domicílios'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['v']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['v']['PNT']['color'],
                        weight=valuesSYMBOLS['v']['PNT']['weight'],
                        fill=valuesSYMBOLS['v']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['v']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['v']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['v']['type']):
                CLASS_BINS_v0002 = np.linspace(
                    DF_BAIRROS_LYR['Tot_Domicílios'].min(), 
                    DF_BAIRROS_LYR['Tot_Domicílios'].max(), 
                    NRO_CLASSES + 1)
                CMAP_v0002 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_v0002 = mcolors.BoundaryNorm(CLASS_BINS_v0002, CMAP_v0002.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Tot_Domicílios']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_v0002(NORM_v0002(feature['properties']['Tot_Domicílios']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Tot_Domicílios'],
                    #     aliases=['Bairro: ','Tot_Domicílios:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Tot_Domicílios'],
                    #     aliases=['Bairro: ','Tot_Domicílios:']
                    # ),
                    name='Tot_Domicílios',
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== Tot_Domicílios_pvt =====
        if (PROP == 'Tot_Domicílios_pvt'):
            if ('PNT' in valuesSYMBOLS['v']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Tot_Domicílios_pvt (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Tot_Domicílios_pvt','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Tot_Domicílios_pvt'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['v']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['v']['PNT']['color'],
                        weight=valuesSYMBOLS['v']['PNT']['weight'],
                        fill=valuesSYMBOLS['v']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['v']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['v']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['v']['type']):
                CLASS_BINS_v0003 = np.linspace(
                    DF_BAIRROS_LYR['Tot_Domicílios_pvt'].min(), 
                    DF_BAIRROS_LYR['Tot_Domicílios_pvt'].max(), 
                    NRO_CLASSES + 1)
                CMAP_v0003 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_v0003 = mcolors.BoundaryNorm(CLASS_BINS_v0003, CMAP_v0003.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Tot_Domicílios_pvt']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_v0003(NORM_v0003(feature['properties']['Tot_Domicílios_pvt']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Tot_Domicílios_pvt'],
                    #     aliases=['Bairro: ','Tot_Domicílios_pvt:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Tot_Domicílios_pvt'],
                    #     aliases=['Bairro: ','Tot_Domicílios_pvt:']
                    # ),
                    name='Tot_Domicílios_pvt',
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== Tot_Domicílios_col =====
        if (PROP == 'Tot_Domicílios_col'):
            if ('PNT' in valuesSYMBOLS['v']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Tot_Domicílios_col (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Tot_Domicílios_col','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Tot_Domicílios_col'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['v']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['v']['PNT']['color'],
                        weight=valuesSYMBOLS['v']['PNT']['weight'],
                        fill=valuesSYMBOLS['v']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['v']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['v']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['v']['type']):
                CLASS_BINS_v0004 = np.linspace(
                    DF_BAIRROS_LYR['Tot_Domicílios_col'].min(), 
                    DF_BAIRROS_LYR['Tot_Domicílios_col'].max(), 
                    NRO_CLASSES + 1)
                CMAP_v0004 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_v0004 = mcolors.BoundaryNorm(CLASS_BINS_v0004, CMAP_v0004.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Tot_Domicílios_col']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_v0004(NORM_v0004(feature['properties']['Tot_Domicílios_col']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Tot_Domicílios_col'],
                    #     aliases=['Bairro: ','Tot_Domicílios_col:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Tot_Domicílios_col'],
                    #     aliases=['Bairro: ','Tot_Domicílios_col:']
                    # ),
                    name='Tot_Domicílios_col',
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== Med_pess_dom_pvt_ocup =====
        if (PROP == 'Med_pess_dom_pvt_ocup'):
            if ('PNT' in valuesSYMBOLS['v']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Med_pess_dom_pvt_ocup (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Med_pess_dom_pvt_ocup','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Med_pess_dom_pvt_ocup'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['v']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['v']['PNT']['color'],
                        weight=valuesSYMBOLS['v']['PNT']['weight'],
                        fill=valuesSYMBOLS['v']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['v']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['v']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['v']['type']):
                CLASS_BINS_v0005 = np.linspace(
                    DF_BAIRROS_LYR['Med_pess_dom_pvt_ocup'].min(), 
                    DF_BAIRROS_LYR['Med_pess_dom_pvt_ocup'].max(), 
                    NRO_CLASSES + 1)
                CMAP_v0005 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_v0005 = mcolors.BoundaryNorm(CLASS_BINS_v0005, CMAP_v0005.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Med_pess_dom_pvt_ocup']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_v0005(NORM_v0005(feature['properties']['Med_pess_dom_pvt_ocup']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Med_pess_dom_pvt_ocup'],
                    #     aliases=['Bairro: ','Med_pess_dom_pvt_ocup:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Med_pess_dom_pvt_ocup'],
                    #     aliases=['Bairro: ','Med_pess_dom_pvt_ocup:']
                    # ),
                    name='Med_pess_dom_pvt_ocup',
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== Perc_Dom_pvt_ocup =====
        if (PROP == 'Perc_Dom_pvt_ocup'):
            if ('PNT' in valuesSYMBOLS['v']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Perc_Dom_pvt_ocup (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Perc_Dom_pvt_ocup','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Perc_Dom_pvt_ocup'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['v']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['v']['PNT']['color'],
                        weight=valuesSYMBOLS['v']['PNT']['weight'],
                        fill=valuesSYMBOLS['v']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['v']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['v']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['v']['type']):
                CLASS_BINS_v0006 = np.linspace(
                    DF_BAIRROS_LYR['Perc_Dom_pvt_ocup'].min(), 
                    DF_BAIRROS_LYR['Perc_Dom_pvt_ocup'].max(), 
                    NRO_CLASSES + 1)
                CMAP_v0006 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_v0006 = mcolors.BoundaryNorm(CLASS_BINS_v0006, CMAP_v0006.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Perc_Dom_pvt_ocup']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_v0006(NORM_v0006(feature['properties']['Perc_Dom_pvt_ocup']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Perc_Dom_pvt_ocup'],
                    #     aliases=['Bairro: ','Perc_Dom_pvt_ocup:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Perc_Dom_pvt_ocup'],
                    #     aliases=['Bairro: ','Perc_Dom_pvt_ocup:']
                    # ),
                    name='Perc_Dom_pvt_ocup',
                    show=False,
                ).add_to(mapIndicators, index=999)

        # ===== Tot_Dom_pvt_ocup =====
        if (PROP == 'Tot_Dom_pvt_ocup'):
            if ('PNT' in valuesSYMBOLS['v']['type']):
                temperaturaGRPLYR = folium.FeatureGroup(name='Tot_Dom_pvt_ocup (Ponto)')
                LYR_BAIRROS = DF_BAIRROS_LYR[['GEOID','geometry','BAIRRO','Tot_Dom_pvt_ocup','LAT','LON']].copy()
                LYR_BAIRROS['SYMBOL_CLASS'] = pd.cut(LYR_BAIRROS['Tot_Dom_pvt_ocup'], bins=symbolClasses, labels=np.arange(1, symbolClasses+1))
                for idx, row in LYR_BAIRROS.iterrows():
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=valuesSYMBOLS['v']['PNT']['sizes'][int(row['SYMBOL_CLASS']) - 1],
                        color=valuesSYMBOLS['v']['PNT']['color'],
                        weight=valuesSYMBOLS['v']['PNT']['weight'],
                        fill=valuesSYMBOLS['v']['PNT']['fill'],
                        fill_color=valuesSYMBOLS['v']['PNT']['color'],
                        fill_opacity=valuesSYMBOLS['v']['PNT']['opacity'],
                    ).add_to(temperaturaGRPLYR)
                temperaturaGRPLYR.show=False
                temperaturaGRPLYR.add_to(mapIndicators)
            if ('PLG' in valuesSYMBOLS['v']['type']):
                CLASS_BINS_v0007 = np.linspace(
                    DF_BAIRROS_LYR['Tot_Dom_pvt_ocup'].min(), 
                    DF_BAIRROS_LYR['Tot_Dom_pvt_ocup'].max(), 
                    NRO_CLASSES + 1)
                CMAP_v0007 = mcolors.LinearSegmentedColormap.from_list('custom', ['#f4d444', '#f86ca7'], N=NRO_CLASSES)
                NORM_v0007 = mcolors.BoundaryNorm(CLASS_BINS_v0007, CMAP_v0007.N)
                folium.GeoJson(
                    DF_BAIRROS_LYR[['geometry','GEOID','BAIRRO','Tot_Dom_pvt_ocup']],
                    style_function = lambda feature: {
                        'fillColor': mcolors.to_hex(CMAP_v0007(NORM_v0007(feature['properties']['Tot_Dom_pvt_ocup']))),
                        'color': 'black',
                        'weight': 0,
                        'fillOpacity': 0.7,
                    },
                    # tooltip=folium.features.GeoJsonTooltip(
                    #     fields=['BAIRRO','Tot_Dom_pvt_ocup'],
                    #     aliases=['Bairro: ','Tot_Dom_pvt_ocup:']
                    # ),
                    # popup=folium.GeoJsonPopup(
                    #     fields=['BAIRRO','Tot_Dom_pvt_ocup'],
                    #     aliases=['Bairro: ','Tot_Dom_pvt_ocup:']
                    # ),
                    name='Tot_Dom_pvt_ocup',
                    show=False,
                ).add_to(mapIndicators, index=999)

    folium.FitOverlays().add_to(mapIndicators)
    folium.LayerControl().add_to(mapIndicators)

folium_static(mapIndicators)