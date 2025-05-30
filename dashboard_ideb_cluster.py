# %%
# Step 1: Install necessary libraries (ensure dash-bootstrap-components is included)
#!pip install pysal scikit-learn --quiet

# %%

# Step 2: Import libraries
import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import geopandas as gpd
import folium
from branca.colormap import linear
import sys
import io

# %%

# Import for LISA, spatial weights, and scaling
import libpysal
from esda.moran import Moran_Local
from sklearn.preprocessing import scale # For scaling data

# %%

# --- Configuration & File Paths ---
IDEB_FILE_PATH = "ideb_escola_2021.txt"
MUNICIPIOS_GEOJSON_PATH = "municipios_br_2020.geojson"
ESTADOS_GEOJSON_PATH = "estados_br_2020.geojson"

# %%

# --- Adjusted GeoJSON column names based on your feedback ---
GEOJSON_MUN_CODE_COL = 'code_muni'
GEOJSON_MUN_NAME_COL = 'name_muni'
GEOJSON_STATE_ABBR_COL_MUN = 'abbrev_state'
GEOJSON_STATE_ABBR_COL_EST = 'abbrev_state'

# %%

# --- 1. Data Loading and Initial Preparation ---
print("Starting data loading process...") #
try:
    try:
        ideb_df = pd.read_csv(IDEB_FILE_PATH, delimiter='\t', encoding='utf-8') #
    except UnicodeDecodeError:
        print("UTF-8 decoding failed, trying latin-1 for IDEB file...") #
        ideb_df = pd.read_csv(IDEB_FILE_PATH, delimiter='\t', encoding='latin1') #
    print(f"IDEB data loaded. Shape: {ideb_df.shape}") #
except FileNotFoundError:
    print(f"ERROR: '{IDEB_FILE_PATH}' not found.") #
    sys.exit()
except Exception as e:
    print(f"Error loading IDEB data: {e}") #
    sys.exit()

try:
    municipios_gdf = gpd.read_file(MUNICIPIOS_GEOJSON_PATH)
    print(f"Municipalities GDF loaded. Shape: {municipios_gdf.shape}")
    estados_gdf = gpd.read_file(ESTADOS_GEOJSON_PATH)
    print(f"States GDF loaded. Shape: {estados_gdf.shape}")
except FileNotFoundError:
    print("ERROR: GeoJSON file(s) not found.")
    sys.exit()
except Exception as e:
    print(f"Error loading GeoJSON data: {e}")
    sys.exit()

# %%

# --- Data Cleaning and Preprocessing ---
print("Starting data cleaning and preprocessing...")
if 'ideb' in ideb_df.columns: #
    ideb_df['ideb'] = pd.to_numeric(ideb_df['ideb'], errors='coerce') #
    ideb_df.dropna(subset=['ideb'], inplace=True) #
    ideb_df = ideb_df[ideb_df['ideb'] != 0].copy() #
else:
    print("CRITICAL ERROR: 'ideb' column not found.") #
    sys.exit()

ideb_essential_cols = ['UF', 'cod_mun', 'nota_matem', 'nota_portugues', 'ideb']
for col in ideb_essential_cols:
    if col not in ideb_df.columns:
        print(f"CRITICAL ERROR: Column '{col}' not found in IDEB data.")
        sys.exit()
    if col in ['nota_matem', 'nota_portugues', 'ideb']:
        ideb_df[col] = pd.to_numeric(ideb_df[col], errors='coerce')
ideb_df.dropna(subset=['nota_matem', 'nota_portugues', 'ideb'], inplace=True)

try:
    ideb_df['cod_mun_int'] = pd.to_numeric(ideb_df['cod_mun'], errors='coerce').astype('Int64') #
    municipios_gdf[GEOJSON_MUN_CODE_COL + '_int'] = pd.to_numeric(municipios_gdf[GEOJSON_MUN_CODE_COL], errors='coerce').astype('Int64') #
    ideb_df.dropna(subset=['cod_mun_int'], inplace=True) #
    municipios_gdf.dropna(subset=[GEOJSON_MUN_CODE_COL + '_int'], inplace=True) #
except Exception as e:
    print(f"Error converting mun codes: {e}") #
    sys.exit()

if municipios_gdf.crs != "EPSG:4326":
    municipios_gdf = municipios_gdf.to_crs("EPSG:4326")
if estados_gdf.crs != "EPSG:4326":
    estados_gdf = estados_gdf.to_crs("EPSG:4326")

abrev_estados = sorted(list(estados_gdf[GEOJSON_STATE_ABBR_COL_EST].unique()))
if not abrev_estados:
    print("CRITICAL ERROR: No states found for dropdown.")
    sys.exit()
print("Data cleaning and preprocessing finished.")

# %%

# --- Dash App Initialization with Dark Theme ---
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG]) #
server = app.server #

# --- App Layout using Dash Bootstrap Components ---
app.layout = dbc.Container(
    fluid=True,
    className="dbc dbc-dark",
    style={'padding': '15px'},
    children=[
        dbc.Row( #
            dbc.Col(
                html.H1("Painel IDEB Brasil", className="text-center display-4 mb-4", style={'color': '#66D9EF'}), #
            )
        ),
        dbc.Row(
            dbc.Col(
                [
                    html.Label("Selecione um Estado:", className="mb-2"), #
                    dcc.Dropdown(
                        id='estado-dropdown',
                        options=[{'label': estado, 'value': estado} for estado in abrev_estados], #
                        value=abrev_estados[0] if abrev_estados else None, #
                        clearable=False,
                        # style={'color': '#333'} #
                    )
                ],
                width={'size': 6, 'offset': 3}, #
                md={'size': 4, 'offset': 4} #
            ),
            className="mb-4"
        ),
        dbc.Row( #
            [
                dbc.Col(
                    [
                        html.H4("Notas Médias por Município", className="mb-3", style={'color': '#A6E22E'}),
                        dash_table.DataTable( #
                            id='ideb-table',
                            page_size=10, #
                            style_table={ #
                                'overflowY': 'auto', #
                                'maxHeight': '560px', #
                                'minHeight': '560px' #
                            },
                            style_header={ #
                                'backgroundColor': 'rgb(30, 30, 30)', #
                                'color': 'white', #
                                'fontWeight': 'bold', #
                                'border': '1px solid rgb(50,50,50)' #
                            },
                            style_cell={ #
                                'backgroundColor': 'rgb(50, 50, 50)', #
                                'color': 'white', #
                                'textAlign': 'left', #
                                'padding': '10px', #
                                'minWidth': '80px', 'width': '120px', 'maxWidth': '180px', #
                                'border': '1px solid rgb(70,70,70)', #
                                'whiteSpace': 'normal', #
                                'height': 'auto' #
                            },
                            style_data_conditional=[ #
                                {
                                    'if': {'row_index': 'odd'}, #
                                    'backgroundColor': 'rgb(40, 40, 40)' #
                                }
                            ] #
                        )
                    ],
                    xs=12, sm=12, md=5, lg=4, #
                    className="mb-4 mb-md-0" #
                ),
                dbc.Col( #
                    [
                        dcc.Tabs(
                            id="map-tabs",
                            value='tab-mat', # Default tab
                            className="mb-3", #
                            children=[
                                dcc.Tab(label='Matemática', value='tab-mat', #
                                        children=[html.Iframe(id='map-mat', width='100%', height='600px', style={'border': 'none', 'borderRadius': '4px'})]), #
                                dcc.Tab(label='Português', value='tab-por', #
                                        children=[html.Iframe(id='map-por', width='100%', height='600px', style={'border': 'none', 'borderRadius': '4px'})]), #
                                dcc.Tab(label='IDEB', value='tab-ideb',
                                        children=[html.Iframe(id='map-ideb', width='100%', height='600px', style={'border': 'none', 'borderRadius': '4px'})]),
                                dcc.Tab(label='Cluster IDEB (LISA)', value='tab-lisa', # NEW TAB
                                        children=[html.Iframe(id='map-lisa', width='100%', height='600px', style={'border': 'none', 'borderRadius': '4px'})])
                            ] #
                        )
                    ],
                    xs=12, sm=12, md=7, lg=8 #
                ) #
            ]
        )
    ]
)

# --- Helper function to create Folium maps (scores) ---
def create_folium_map(gdf_data, score_column_name, legend_name, selected_estado_geom, mun_name_col):
    if selected_estado_geom is None or selected_estado_geom.is_empty:
        map_center = [-14.2350, -51.9253]
        m = folium.Map(location=map_center, zoom_start=3, tiles="CartoDB dark_matter", control_scale=True)
        folium.Marker(map_center, popup="Geometria do estado não disponível.").add_to(m)
        return m._repr_html_() #

    map_center = [selected_estado_geom.centroid.y, selected_estado_geom.centroid.x]
    m = folium.Map(location=map_center, tiles="CartoDB dark_matter", control_scale=True)
    try:
        bounds = selected_estado_geom.bounds
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    except Exception:
        m.zoom_start = 6

    if gdf_data.empty or score_column_name not in gdf_data.columns or gdf_data[score_column_name].isnull().all():
        if not selected_estado_geom.is_empty:
            folium.GeoJson(selected_estado_geom, style_function=lambda x: {'fillColor': '#444', 'color': '#777', 'weight': 1}).add_to(m) #
        folium.Marker(
            location=[selected_estado_geom.centroid.y, selected_estado_geom.centroid.x],
            popup=f"Dados de '{legend_name}' não disponíveis.",
            icon=folium.Icon(color='gray')
        ).add_to(m)
        return m._repr_html_()

    gdf_data[score_column_name] = pd.to_numeric(gdf_data[score_column_name], errors='coerce')
    valid_scores = gdf_data[score_column_name].dropna()

    if valid_scores.empty: #
        min_val, max_val = 0, 1
        colormap = linear.Greys_03.scale(min_val, max_val)
    else:
        min_val = valid_scores.min()
        max_val = valid_scores.max()
        if min_val == max_val:
            colormap = linear.viridis.scale(min_val - (0.5 if min_val > 0 else 0), max_val + 0.5 if max_val > 0 else 1)
            if min_val == 0 and max_val == 0: colormap = linear.viridis.scale(0,1) #
        else:
            colormap = linear.viridis.scale(min_val, max_val)
    colormap.caption = legend_name

    tooltip_style = "background-color: #2a2a2a; color: #f0f0f0; border: 1px solid #555; padding: 8px; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.5);" #

    folium.GeoJson(
        gdf_data,
        name=legend_name,
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties'][score_column_name]) if pd.notnull(feature['properties'][score_column_name]) else '#333333',
            'color': '#888888',
            'weight': 0.7,
            'fillOpacity': 0.75, #
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=[mun_name_col, score_column_name],
            aliases=['Município:', f'{legend_name}:'],
            localize=True, sticky=False, labels=True,
            style=tooltip_style
        ),
        highlight_function=lambda x: {'weight': 2, 'fillOpacity': 0.9, 'color': '#00FFFF'}
    ).add_to(m) #
    
    m.add_child(colormap)
    return m._repr_html_()

# --- Helper function to create Folium LISA Cluster map ---
def create_lisa_cluster_map(gdf_data, cluster_column_name, legend_name, selected_estado_geom, mun_name_col):
    if selected_estado_geom is None or selected_estado_geom.is_empty:
        map_center = [-14.2350, -51.9253]
        m = folium.Map(location=map_center, zoom_start=3, tiles="CartoDB dark_matter", control_scale=True)
        folium.Marker(map_center, popup="Geometria do estado não disponível.").add_to(m)
        return m._repr_html_()

    map_center = [selected_estado_geom.centroid.y, selected_estado_geom.centroid.x]
    m = folium.Map(location=map_center, tiles="CartoDB dark_matter", control_scale=True)
    try:
        bounds = selected_estado_geom.bounds
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    except Exception:
        m.zoom_start = 6

    if gdf_data.empty or cluster_column_name not in gdf_data.columns or gdf_data[cluster_column_name].isnull().all():
        if not selected_estado_geom.is_empty:
            folium.GeoJson(selected_estado_geom, style_function=lambda x: {'fillColor': '#444', 'color': '#777', 'weight': 1}).add_to(m)
        folium.Marker(
            location=[selected_estado_geom.centroid.y, selected_estado_geom.centroid.x],
            popup=f"Dados de '{legend_name}' não disponíveis ou insuficientes para clusterização.",
            icon=folium.Icon(color='gray')
        ).add_to(m)
        return m._repr_html_()

    # Define colors for clusters (inspired by PDF page 51)
    # Original R code uses: "red", "blue", "lightpink", "skyblue2", "white"
    # for HH, LL, HL, LH, Non-significant
    color_map = {
        1: 'red',        # High-High
        2: 'blue',       # Low-Low
        3: 'lightpink',  # High-Low
        4: 'skyblue',    # Low-High (skyblue2 changed to skyblue for web colors)
        5: '#DDDDDD',    # Non-significant (white might be too stark on dark theme)
        0: '#555555'     # Default/NoData (if any slip through)
    }
    
    cluster_labels = {
        1: "High-High",
        2: "Low-Low",
        3: "High-Low",
        4: "Low-High",
        5: "Non-Significant",
        0: "No Data"
    }

    tooltip_style = "background-color: #2a2a2a; color: #f0f0f0; border: 1px solid #555; padding: 8px; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.5);"

    folium.GeoJson(
        gdf_data,
        name=legend_name,
        style_function=lambda feature: {
            'fillColor': color_map.get(feature['properties'][cluster_column_name], '#333333'), # Use .get for safety
            'color': '#888888',
            'weight': 0.7,
            'fillOpacity': 0.75,
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=[mun_name_col, cluster_column_name, 'media_ideb_lisa', 'lisa_p_value'], # Add more info to tooltip
            aliases=['Município:', 'Cluster:', 'IDEB Médio:', 'LISA p-value:'],
            localize=True, sticky=False, labels=True,
            style=tooltip_style,
            fields_to_labels={cluster_column_name: cluster_labels} # Custom labels for cluster codes
        ),
        highlight_function=lambda x: {'weight': 2, 'fillOpacity': 0.9, 'color': '#00FFFF'}
    ).add_to(m)

    # Add a custom legend for categorical clusters (HTML based)
    legend_html = """
     <div style="position: fixed; 
                 bottom: 50px; left: 50px; width: 150px; height: auto; 
                 border:2px solid grey; z-index:9999; font-size:14px;
                 background-color:rgba(40,40,40,0.85); color:white; padding:5px; border-radius:5px;">
       &nbsp; <b>""" + legend_name + """</b> <br>
       &nbsp; <i style="background:red; opacity:0.75;"></i>&nbsp; High-High (HH) <br>
       &nbsp; <i style="background:blue; opacity:0.75;"></i>&nbsp; Low-Low (LL) <br>
       &nbsp; <i style="background:lightpink; opacity:0.75;"></i>&nbsp; High-Low (HL) <br>
       &nbsp; <i style="background:skyblue; opacity:0.75;"></i>&nbsp; Low-High (LH) <br>
       &nbsp; <i style="background:#DDDDDD; opacity:0.75;"></i>&nbsp; Non-Significant <br>
       &nbsp; <i style="background:#555555; opacity:0.75;"></i>&nbsp; No Data/Error
     </div>
     <style>
        i {
            width: 12px;
            height: 12px;
            float: left;
            margin-right: 5px;
            border: 1px solid black;
        }
     </style>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m._repr_html_()


# --- Callbacks ---
@app.callback(
    [Output('ideb-table', 'data'), Output('ideb-table', 'columns'),
     Output('map-mat', 'srcDoc'), Output('map-por', 'srcDoc'), Output('map-ideb', 'srcDoc'),
     Output('map-lisa', 'srcDoc')], # NEW OUTPUT for LISA map
    [Input('estado-dropdown', 'value')]
)
def update_dashboard(selected_estado_abbrev):
    empty_df_for_table = pd.DataFrame(columns=['Município', 'Média Mat.', 'Média Port.', 'Média IDEB']) #
    cols_for_table = [{"name": i, "id": i} for i in empty_df_for_table.columns]
    default_geom = estados_gdf.unary_union if not estados_gdf.empty else None
    default_map_html = create_folium_map(gpd.GeoDataFrame(), "", "Nenhum estado selecionado", default_geom, GEOJSON_MUN_NAME_COL)
    default_lisa_map_html = create_lisa_cluster_map(gpd.GeoDataFrame(), "", "Cluster IDEB (LISA)", default_geom, GEOJSON_MUN_NAME_COL)


    if not selected_estado_abbrev:
        return empty_df_for_table.to_dict('records'), cols_for_table, default_map_html, default_map_html, default_map_html, default_lisa_map_html

    ideb_estado_df = ideb_df[ideb_df['UF'] == selected_estado_abbrev].copy() #
    
    # Aggregate data per municipality
    media_mat = ideb_estado_df.groupby('cod_mun_int')['nota_matem'].mean().reset_index(name='media_mat')
    media_por = ideb_estado_df.groupby('cod_mun_int')['nota_portugues'].mean().reset_index(name='media_por')
    media_ideb_agg = ideb_estado_df.groupby('cod_mun_int')['ideb'].mean().reset_index(name='media_ideb')

    notas_df = pd.merge(media_mat, media_por, on='cod_mun_int', how='outer')
    notas_df = pd.merge(notas_df, media_ideb_agg, on='cod_mun_int', how='outer')

    estado_municipios_gdf = municipios_gdf[municipios_gdf[GEOJSON_STATE_ABBR_COL_MUN] == selected_estado_abbrev].copy()
    # Ensure unique geometries if there are duplicate mun_codes in geojson (take first)
    estado_municipios_gdf = estado_municipios_gdf.drop_duplicates(subset=GEOJSON_MUN_CODE_COL + '_int', keep='first')


    map_data_gdf = pd.merge(estado_municipios_gdf, notas_df, 
                            left_on=GEOJSON_MUN_CODE_COL + '_int', 
                            right_on='cod_mun_int', how='left') #

    table_display_df = map_data_gdf[[GEOJSON_MUN_NAME_COL, 'media_mat', 'media_por', 'media_ideb']].copy()
    table_display_df.columns = ['Município', 'Média Mat.', 'Média Port.', 'Média IDEB']
    for col_name in ['Média Mat.', 'Média Port.', 'Média IDEB']:
        table_display_df[col_name] = pd.to_numeric(table_display_df[col_name], errors='coerce').round(2)
    table_display_df.fillna('N/A', inplace=True)
    
    table_data = table_display_df.to_dict('records')
    table_columns = [{"name": i, "id": i} for i in table_display_df.columns]

    selected_estado_geom_series = estados_gdf[estados_gdf[GEOJSON_STATE_ABBR_COL_EST] == selected_estado_abbrev].geometry
    selected_estado_geom = selected_estado_geom_series.iloc[0] if not selected_estado_geom_series.empty else None #

    map_mat_html = create_folium_map(map_data_gdf, 'media_mat', 'Média Matemática', selected_estado_geom, GEOJSON_MUN_NAME_COL)
    map_por_html = create_folium_map(map_data_gdf, 'media_por', 'Média Português', selected_estado_geom, GEOJSON_MUN_NAME_COL)
    map_ideb_html = create_folium_map(map_data_gdf, 'media_ideb', 'Média IDEB', selected_estado_geom, GEOJSON_MUN_NAME_COL)

    # --- LISA Cluster Map Logic ---
    lisa_map_html = default_lisa_map_html # Default if something goes wrong
    # For LISA, we need 'media_ideb' and valid geometries
    # Ensure the GeoDataFrame for LISA has no missing geometries and has the target variable
    lisa_data_gdf = map_data_gdf[['geometry', GEOJSON_MUN_NAME_COL, GEOJSON_MUN_CODE_COL + '_int', 'media_ideb']].copy()
    lisa_data_gdf.rename(columns={'media_ideb': 'media_ideb_lisa'}, inplace=True) # Use a distinct name
    lisa_data_gdf.dropna(subset=['media_ideb_lisa', 'geometry'], inplace=True)
    lisa_data_gdf = lisa_data_gdf[lisa_data_gdf.geometry.is_valid & ~lisa_data_gdf.geometry.is_empty]
    lisa_data_gdf.set_index(GEOJSON_MUN_CODE_COL + '_int', inplace=True) # Index needed for pysal

    if not lisa_data_gdf.empty and len(lisa_data_gdf) > 4: # Need enough observations for spatial analysis
        try:
            # 1. Create Spatial Weights (Queen contiguity, row-standardized)
            # Ensure no islands, or handle them. For simplicity, we proceed.
            # If there are islands, `libpysal.weights.Queen.from_dataframe` might raise an error
            # or create weights that Moran_Local can't handle well without `allow_null_weights=True`
            # or `zero_policy=True` in older pysal or spdep R equivalent.
            # For `Moran_Local` in `esda`, it's generally robust to islands if they are disconnected.
            
            # Filter out invalid geometries before weights calculation
            lisa_data_gdf_clean = lisa_data_gdf[lisa_data_gdf.geometry.is_valid & ~lisa_data_gdf.geometry.is_empty].copy()
            
            if not lisa_data_gdf_clean.empty and len(lisa_data_gdf_clean) > 4 :
                w = libpysal.weights.Queen.from_dataframe(lisa_data_gdf_clean, ids=lisa_data_gdf_clean.index.tolist())
                w.transform = 'R' # Row-standardized, similar to style="W" in R spdep

                # 2. Calculate Local Moran's I
                y = lisa_data_gdf_clean['media_ideb_lisa'].values
                lisa = Moran_Local(y, w, permutations=999) # Use 999 permutations for p-value

                lisa_data_gdf_clean['lisa_Ii'] = lisa.Is # Local Moran values
                lisa_data_gdf_clean['lisa_p_value'] = lisa.p_sim # Simulated p-values
                lisa_data_gdf_clean['lisa_q'] = lisa.q # Quadrants (1=HH, 2=LH, 3=LL, 4=HL)

                # 3. Scale data (y) and calculate spatial lag (wy) for explicit cluster definition
                # As per PDF page 50 and 158 (Moran plot with scaled values)
                y_scaled = scale(y) # (value - mean) / std.dev
                wy_scaled = libpysal.weights.lag_spatial(w, y_scaled)

                lisa_data_gdf_clean['y_scaled'] = y_scaled
                lisa_data_gdf_clean['wy_scaled'] = wy_scaled
                
                # 4. Define Clusters based on PDF logic (page 50)
                # Quadrant definitions: 1=HH, 2=LL, 3=HL, 4=LH, 5=Non-significant
                lisa_data_gdf_clean['cluster_type'] = 5 # Default to Non-significant

                # High-High (AA)
                lisa_data_gdf_clean.loc[(lisa_data_gdf_clean['y_scaled'] >= 0) & (lisa_data_gdf_clean['wy_scaled'] >= 0) & (lisa_data_gdf_clean['lisa_p_value'] <= 0.05), 'cluster_type'] = 1
                # Low-Low (BB)
                lisa_data_gdf_clean.loc[(lisa_data_gdf_clean['y_scaled'] < 0) & (lisa_data_gdf_clean['wy_scaled'] < 0) & (lisa_data_gdf_clean['lisa_p_value'] <= 0.05), 'cluster_type'] = 2
                # High-Low (AB) - Outlier
                lisa_data_gdf_clean.loc[(lisa_data_gdf_clean['y_scaled'] >= 0) & (lisa_data_gdf_clean['wy_scaled'] < 0) & (lisa_data_gdf_clean['lisa_p_value'] <= 0.05), 'cluster_type'] = 3
                # Low-High (BA) - Outlier
                lisa_data_gdf_clean.loc[(lisa_data_gdf_clean['y_scaled'] < 0) & (lisa_data_gdf_clean['wy_scaled'] >= 0) & (lisa_data_gdf_clean['lisa_p_value'] <= 0.05), 'cluster_type'] = 4
                
                lisa_data_gdf_clean.reset_index(inplace=True) # To make mun_code_int a column again for merging/mapping
                lisa_map_html = create_lisa_cluster_map(lisa_data_gdf_clean, 'cluster_type', 'Cluster IDEB (LISA)', selected_estado_geom, GEOJSON_MUN_NAME_COL)

            else: # Not enough valid geometries after cleaning
                 print(f"Not enough valid geometries for LISA in {selected_estado_abbrev} after cleaning.")
                 lisa_map_html = create_lisa_cluster_map(gpd.GeoDataFrame(), 'cluster_type', 'Cluster IDEB (LISA) - Data Insufficient', selected_estado_geom, GEOJSON_MUN_NAME_COL)


        except Exception as e:
            print(f"Error during LISA calculation for {selected_estado_abbrev}: {e}")
            # Create an empty map or map with an error message
            lisa_map_html = create_lisa_cluster_map(gpd.GeoDataFrame(), 'cluster_type', 'Cluster IDEB (LISA) - Error', selected_estado_geom, GEOJSON_MUN_NAME_COL)
    else: # Not enough observations
        print(f"Not enough observations for LISA in {selected_estado_abbrev}. Need > 4, got {len(lisa_data_gdf)}")
        lisa_map_html = create_lisa_cluster_map(lisa_data_gdf, 'cluster_type', 'Cluster IDEB (LISA) - Data Insufficient', selected_estado_geom, GEOJSON_MUN_NAME_COL)


    return table_data, table_columns, map_mat_html, map_por_html, map_ideb_html, lisa_map_html

# --- Run the App ---
if __name__ == '__main__':
    print("Starting Dash app...")
    app.run(debug=True, port=8052) # [cite: 43]


