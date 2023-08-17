import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import networkx as nx
import shapely
from pathlib import Path
import random
import yaml
import config.agg_cfg as cfg

def load_parcels(parcel_path,piecewise = False, **kwargs):
    '''
    load all land parcels into parcel variable
    
    Inputs:
        parcel_path: if piecewice=False, expects a path to a full geodatabase (.gdb) file.
                    Otherwise, assumes that there are parcel shapefiles within that directory,
                    and merges them all into one variable.
        piecewise: whether or not parcel_path is a folder of .shp files. Default False, 
            meaning parcel_path points to a single .gdb file.
        **kwargs: additional args to pass to gpd.read_file
    Outputs:
        GeoDataFrame of parcels.
    '''
    
    if piecewise:
        parcel_paths = glob.glob(parcel_path + '*/*.shp')
        parcels = pd.DataFrame()
        for app in tqdm(parcel_paths):
            parcel_data = gpd.read_file(app)
            parcels = pd.concat((parcels,parcel_data), ignore_index=True)
    else:
        parcels = gpd.read_file(parcel_path, driver='FileGDB', **kwargs).to_crs(3071)

    parcels = gpd.GeoDataFrame(parcels).to_crs(3071)
    
    # Create indices for tracking downstream (WI-created PARCELID has duplicates)
    parcels['parcel_index'] = list(range(0, parcels.shape[0]))
    parcels['parcel_index'] = parcels['parcel_index'].astype(str)
    # Clean up parcel geometries
    parcels['geometry'] = shapely.make_valid(parcels['geometry'])
    parcels = parcels[(~parcels['geometry'].isna())]

    return parcels

def load_polygons(polygon_path):
    """
    Load polygons from .geojson file
    
    Inputs:
        polygon_path: Path to polygon .geojson file
    Outputs:
        GeoDataFrame of polygons.
    """
    polygons = gpd.read_file(polygon_path, driver='GeoJSON').to_crs(3071)
    # Create index for tracking downstream
    polygons['polygon_index'] = list(range(0, polygons.shape[0]))

    return polygons

def cluster(parcels, polygons, cluster_neighbors=None):
    """
    Joins polygons on the same parcel and neighboring parcels into clusters.

    Inputs:
        parcels: GeoDataFrame with parcel data
        polygons: GeoDataFrame with polygon data
        cluster_neighbors: String, how to cluster polygons in neighboring parcels.
            Default None does not cluster polygons in neighboring parcels.
            'chain' allows for chains of more than two parcels in a row.
            'pairwise_random' clusters polygons on the same parcel together, as well as 
                with polygons in one adjacent parcel. In cases where a parcel has more than
                one adjacent parcel with polygons, one adjacent parcel is chosen randomly,
                and all polygons on that parcel are clustered with polygons on the original
                parcel, and these polygons are not considered for further clustering.
            'pairwise_all' same as 'pairwise_random, except in cases where a parcel has more
                 than one neighboring parcel with polygons, all pairs are collected.
                 Note that this method duplicates polygons across multiple distinct clusters,
                 and thus should not be used for aggregated estimation. 
    Outputs:
        GeoDataFrame of clusters, with associated geometry and lists of component polygons
    """
    if cluster_neighbors not in [None, 'chain', 'pairwise_random', 'pairwise_all']:
        print('Error: parameter cluster_neighbors must be None or one of \'chain\', \'pairwise_random\', \'pairwise_all\'.')
        return
    
    # 1. Merge and preprocess parcel data
    print('PREPROCESSING PARCEL DATA...')

    # Join each polygon to the one or more parcels that it intersects
    polygons_merged = polygons.sjoin(parcels, how='inner')
    polygons_merged = polygons_merged[['geometry', 'polygon_index', 'parcel_index']]
    # Filter to only parcels intersecting polygons, in order to save memory
    parcels = parcels[parcels['parcel_index'].isin(polygons_merged['parcel_index'])].reset_index()

    if cluster_neighbors is not None:
        # Calculate neighbors for parcel subset
        parcels['neighbors'] = pd.Series()
        for i in tqdm(range(0, parcels.shape[0])):  
            neighbors = parcels[parcels.geometry.touches(parcels['geometry'].iloc[i])].parcel_index.tolist() 
            # Don't duplicate original parcel
            if (parcels.iloc[i].parcel_index in neighbors):
                neighbors = neighbors.remove(parcels.iloc[i].parcel_index)
            # If there are no neighbors intersecting polygons, move on
            if not (neighbors):
                continue
            else:
                parcels.at[i, "neighbors"] = ", ".join(neighbors)
        # Storing neighbors as a string of a list
        parcels['neighbors'] = parcels['neighbors'].apply(lambda x: x if pd.isna(x) else str(str.split(x, ', ')))

        # Re-merge with polygons to collect neighbors
        polygons_merged = polygons_merged.merge(parcels[['parcel_index', 'neighbors']], how='left', on='parcel_index')
        # Aggregate parcel join data so we have one row per polygon
        polygons_merged = polygons_merged.groupby(
            ['polygon_index', 'geometry']).agg(parcels=pd.NamedAgg(column='parcel_index', aggfunc='unique'),
                                            neighboring_parcels=pd.NamedAgg(column='neighbors', aggfunc='unique')).reset_index()
    else:
        polygons_merged = polygons_merged.groupby(['polygon_index', 'geometry']).agg(
            parcels=pd.NamedAgg(column='parcel_index', aggfunc='unique')).reset_index()
    print('PARCEL DATA PREPROCESSED')


    # 3. Calculate clusters using a Graph network
    print('CALCULATING CLUSTERS...')

    # Prepare geodataframe for collecting results
    clusters = gpd.GeoDataFrame({'polygon_indices':pd.Series(), 
                                 'parcel_indices':pd.Series()}, geometry=gpd.GeoSeries())

    # Create a graph with a node for each polygon
    polygon_graph = nx.Graph()
    for i in range(0, polygons_merged.shape[0]): 
        polygon_graph.add_node(polygons_merged['polygon_index'].iloc[i],
        # Retain polygon geometry
        geometry=polygons_merged['geometry'].iloc[i],
        # Retain intersecting parcels
        parcels=list(polygons_merged['parcels'].iloc[i]))

    # Loop through all polygons, connecting their nodes if they are in the same parcel
    # or optionally in neighboring parcels
    clustered_polygons = []
    for polygon_a in tqdm(polygons_merged['polygon_index']):
        parcels_a = list(polygons_merged[polygons_merged['polygon_index']==polygon_a]['parcels'].reset_index(drop=True)[0])
        # Connect polygons on same parcel
        same_parcel_polygons = polygons_merged[(polygons_merged['polygon_index'] != polygon_a) & 
                            (polygons_merged['parcels'].apply(
                                lambda x: len(set(list(x)).intersection(set(parcels_a))) > 0))]['polygon_index']
        for polygon_b in same_parcel_polygons:
            polygon_graph.add_edge(polygon_a, polygon_b)

        # Optionally connect polygons on neighboring parcels
        if cluster_neighbors is not None:
            neighboring_parcels = polygons_merged[polygons_merged['polygon_index']==polygon_a]['neighboring_parcels'].reset_index(drop=True)[0][0]
        if cluster_neighbors is not None and not pd.isna(neighboring_parcels):
            neighboring_parcels = eval(neighboring_parcels)
            # For polygons which cross parcel boundaries, make sure intersecting parcels
            # are not listed as neighboring parcels as well
            neighboring_parcels = list(set(neighboring_parcels)-set(parcels_a))
            adjacent_parcel_polygons = polygons_merged[(polygons_merged['polygon_index'] != polygon_a) & 
                            (polygons_merged['parcels'].apply(
                                lambda x: len(set(list(x)).intersection(set(neighboring_parcels))) > 0)) & 
                                (~polygons_merged['polygon_index'].isin(same_parcel_polygons))]
            # If the polygon has any polygons on adjacent parcels
            if adjacent_parcel_polygons.shape[0] > 0:
                # If chain clustering, connect polygons in adjoining parcels without checking if they've
                # already been connected to other polygons
                if cluster_neighbors == 'chain' or cluster_neighbors is None:
                    for polygon_b in adjacent_parcel_polygons['polygon_index']:
                        polygon_graph.add_edge(polygon_a, polygon_b)
                        clustered_polygons.append(polygon_b)
                # If pairwise clustering, consider connecting polygons in adjoining parcels
                #  only if they haven't been connected to another polygon already
                elif cluster_neighbors in ['pairwise_random', 'pairwise_all'] and polygon_a not in clustered_polygons:             
                        pairwise_cluster_candidates = adjacent_parcel_polygons[~adjacent_parcel_polygons['polygon_index'].isin(clustered_polygons)]
                        if pairwise_cluster_candidates.shape[0] > 0:
                            if cluster_neighbors == 'pairwise_random':
                                # Randomly sample a neighboring parcel for pairwise clustering,
                                # and connect to all of its polygons
                                parcel_pair = random.sample(neighboring_parcels, 1)  
                                polygons_parcel_pair = pairwise_cluster_candidates[
                                    (pairwise_cluster_candidates['parcels'].apply(lambda x: parcel_pair in x))]['polygon_index']
                                for polygon_b in polygons_parcel_pair:
                                    polygon_graph.add_edge(polygon_a, polygon_b)
                                    clustered_polygons.append(polygon_b)
                            else: 
                                # For pairwise-all clustering, we'll go straight to recording clusters
                                # in the results table (without creating a graph structure).
                                pairwise_cluster_candidate_parcels = list(set([item for sublist in pairwise_cluster_candidates['parcels'] for item in sublist]))
                                for parcel in pairwise_cluster_candidate_parcels:
                                    parcel_polygons = pairwise_cluster_candidates[  
                                        pairwise_cluster_candidates['parcels'].apply(lambda x: parcel in x)]  
                                    if len(parcel_polygons) > 0:
                                        cluster_polygons = [polygon_a] + list(same_parcel_polygons) + list(parcel_polygons['polygon_index'])
                                        cluster_parcels = parcels_a + [parcel]
                                        cluster_geometry = shapely.MultiPolygon(
                                                    list(polygons_merged[polygons_merged['polygon_index'].isin(same_parcel_polygons)]['geometry']) + 
                                                        list(polygons_merged[polygons_merged['polygon_index']==polygon_a]['geometry']) + 
                                                        list(parcel_polygons['geometry']))
                                        clusters = pd.concat([clusters, gpd.GeoDataFrame(
                                                                {'polygon_indices':pd.Series(str(tuple(cluster_polygons))), 
                                                                'parcel_indices':pd.Series(str(list(map(int, cluster_parcels))))},
                                                                geometry=gpd.GeoSeries(cluster_geometry))],
                                                        ignore_index=True)
                                        clustered_polygons.extend(list(parcel_polygons['polygon_index']))
                clustered_polygons.extend([polygon_a] + list(same_parcel_polygons))
        # When pairwise-all clustering, add same-parcel polygons not clustered across 
        # parcels to the resulting dataframe.
        if cluster_neighbors == 'pairwise_all' and polygon_a not in clustered_polygons:
            cluster_polygons = [polygon_a] + list(same_parcel_polygons)
            cluster_geometry = shapely.MultiPolygon(
                        list(polygons_merged[polygons_merged['polygon_index'].isin(same_parcel_polygons)]['geometry']) + 
                            list(polygons_merged[polygons_merged['polygon_index']==polygon_a]['geometry']))
            clusters = pd.concat([clusters, 
                                gpd.GeoDataFrame(
                {'polygon_indices':pd.Series(str(cluster_polygons)), 
                    'parcel_indices':pd.Series(str(list(map(int, parcels_a))))},
                    geometry=gpd.GeoSeries(cluster_geometry))],
                            ignore_index=True)
            clustered_polygons.extend(cluster_polygons)
    print('CLUSTERS CALCULATED')


    # 4. Collect and postprocess clusters 
    if cluster_neighbors is None or cluster_neighbors != 'pairwise_all':
        # Extract the connected components of the graph as subgraphs
        cluster_subgraphs = [polygon_graph.subgraph(c).copy() for c in nx.connected_components(polygon_graph)]
        # Save these subgraphs into a GeoDataFrame, along with their attributes
        clusters = gpd.GeoDataFrame({'polygon_indices':pd.Series(cluster_subgraphs)}, geometry=gpd.GeoSeries())
        clusters['parcel_indices'] = pd.Series(cluster_subgraphs).apply(lambda x: list(nx.get_node_attributes(x, 'parcels').values())).apply(lambda x: list(set([item for sublist in x for item in sublist])))
        clusters['geometry'] = pd.Series(cluster_subgraphs).apply(lambda x: shapely.MultiPolygon(list(nx.get_node_attributes(x, 'geometry').values())))
        clusters['polygon_indices'] = clusters['polygon_indices'].apply(lambda x: list(x.nodes()))

    clusters['cluster_area_m2'] = clusters['geometry'].area
    print('Number of clusters: ', clusters.shape[0])

    return(clusters)
    

if __name__ == '__main__':
    
    # read in configuration file
    with open(Path().resolve().parent / 'afo_vs_cafo/config/config.yml', 'r') as file:
        configs = yaml.safe_load(file)
    analysis_output_path = Path(configs['analysis_output_path'])

     # 1. Load and clean data
    print('LOADING DATA...')
    parcels = load_parcels(cfg.parcel_dir, whole_state=True)
    polygons = load_polygons(cfg.polygon_path)
    print('ALL DATA LOADED')

    # 2. Run different clustering algorithms
    clusters_no_neighbors = cluster(parcels, polygons)
    clusters_chain = cluster(parcels, polygons, cluster_neighbors='chain')
    random.seed(5)
    clusters_pairwise_random = cluster(parcels, polygons, cluster_neighbors='pairwise_random')
    clusters_pairwise_all = cluster(parcels, polygons, cluster_neighbors='pairwise_all')

    # 3. Save to disk
    clusters_no_neighbors.to_csv(analysis_output_path / 'all_cf_annotations_clusters_no_neighbors.csv', index=False)
    clusters_chain.to_csv(analysis_output_path / 'all_cf_annotations_clusters_chain.csv', index=False)
    clusters_pairwise_random.to_csv(analysis_output_path / 'all_cf_annotations_clusters_pairwise_random.csv', index=False)
    clusters_pairwise_all.to_csv(analysis_output_path / 'all_cf_annotations_clusters_pairwise_all.csv', index=False)