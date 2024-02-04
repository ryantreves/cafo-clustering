
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import glob
import networkx as nx
import shapely
from pathlib import Path
import yaml
from fuzzywuzzy import fuzz
from geofeather.pygeos import from_geofeather
import sys
from ast import literal_eval
import textdistance as td
import time
import numpy as np
import warnings
import functools
import config.config_params as cfg

# Suppress warnings:
# futurewarnings 
warnings.simplefilter(action='ignore', category=FutureWarning)
# floating point runtime errors
np.seterr(invalid="ignore")
# warnings that parcel geofeather file has no crs
warnings.filterwarnings('ignore', message="/Users/rtreves/Documents/RegLab/projects/afo_vs_cafo/data/land_parcels/WI_parcels.feather.crs")


def load_parcels(parcel_path, piecewise=False, feather=False, **kwargs):
    """
    load all land parcels into parcel variable

    Inputs:
        parcel_path: if piecewise=False, expects a path to a full geodatabase (.gdb) file.
                    Otherwise, assumes that there are parcel shapefiles within that directory,
                    and merges them all into one variable.
        piecewise: bool, whether or not parcel_path is a folder of .shp files. Default False,
            meaning parcel_path points to a single .gdb file.
        feather: bool, whether the parcel file is in .feather format and should be loaded using
            from_geofeather().
        **kwargs: additional args to pass to gpd.read_file
    Outputs:
        GeoDataFrame of parcels.
    """
    if feather:
        # Record the start time
        start_time = time.time()
        parcel_feather = from_geofeather(parcel_path)
        # convert if to GeoPandas
        parcels = gpd.GeoDataFrame(parcel_feather, geometry="geometry")
        parcels.rename(columns={"geometry": "parcel_geometry"})
        parcels = parcels.set_crs(3071)
        # Record the end time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Parcel Loading time - FEATHER: {np.round(execution_time,3)}")
    elif piecewise:
        parcel_paths = glob.glob(parcel_path + "*/*.shp")
        parcels = pd.DataFrame()
        for app in tqdm(parcel_paths):
            parcel_data = gpd.read_file(app)
            parcels = pd.concat((parcels, parcel_data), ignore_index=True)
    else:
        start_time = time.time()
        parcels = gpd.read_file(parcel_path, driver="FileGDB", **kwargs).to_crs(3071)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Parcel Loading time -- Regular: {np.round(execution_time,3)}")

    parcels = gpd.GeoDataFrame(parcels).to_crs(3071)

    # Create indices for tracking downstream (WI-created PARCELID has duplicates)
    parcels["parcel_index"] = list(range(0, parcels.shape[0]))
    parcels["parcel_index"] = parcels["parcel_index"].astype(str)
    # Clean up parcel geometries
    parcels["geometry"] = shapely.make_valid(parcels["geometry"])
    parcels = parcels[(~parcels["geometry"].isna())]
    # Fix geometries for parcels with GeometryCollection geometry (currently only ~30 parcels)
    def fix_geometry_collection(geometrycollection):
        polygons = [geom for geom in geometrycollection.geoms if geom.geom_type in ['Polygon', 'MultiPolygon']]
        if polygons[0].geom_type == 'MultiPolygon':
                polygons = polygons[0]
        return shapely.MultiPolygon(polygons)
    parcels_invalid_geoms = parcels[parcels['geometry'].geom_type=='GeometryCollection'].copy()
    parcels_invalid_geoms["geometry"] = parcels_invalid_geoms["geometry"].apply(lambda x: fix_geometry_collection(x))
    parcels.loc[parcels_invalid_geoms.index] = parcels_invalid_geoms

    return parcels


def load_polygons(polygon_path: str,
                 county_data: gpd.GeoDataFrame,
                 image_bound_map: gpd.GeoDataFrame=None,
                 clip_to_counties: bool=True,
                 save_newfile: bool=False,
                 verbose: bool=False):
    """
    Load polygons from .geojson file

    Inputs:
        polygon_path: Path to polygon .geojson file
        county_data: GeoDataFrame of county geometries, for clipping polygons
            and for correct creation of the `jpeg_names` field.
        image_bound_map: GeoDataFrame of bounds of NAIP tiles. Optional,
            supply in order to create (or replace) the `jpeg_names` field.
        clip_to_counties: Bool, whether to omit polygons outside county boundaries.
            In practice, this omits polygons which are within NAIP imagery tiles but 
            lie near-shore in Lake Michigan or Superior.
        save_newfile: Bool, whether or not to replace original polygon file
            with new file after (re)creating `jpeg_names` field.
        verbose: Bool, whether to print progress statements
    Outputs:
        GeoDataFrame of polygons.
    """
    start_time = time.time()
    if verbose:
        print('Loading polygon file...')
    polygons = gpd.read_file(polygon_path, driver="GeoJSON").to_crs(3071)
    # Create index for tracking downstream
    polygons["polygon_index"] = list(range(0, polygons.shape[0]))
    if verbose:
        print('Cleaning geometries...')
    # Clean up polygon geometries
    polygons["geometry"] = shapely.make_valid(polygons["geometry"])
    # If any polygons were transformed to multipolygons by shapely.make_valid
    # (e.g., in the case of tiny extraneous corner pieces), retain the largest piece
    polygons["geometry"] = polygons["geometry"].apply(
        lambda x: x if x.geom_type == "Polygon" else max(x.geoms, key=lambda a: a.area)
    )

    if clip_to_counties:
    # Omit polygons which are within Wisconsin NAIP tiles but outside the WI counties 
    # shapefiles. For the three-band model, this is a set of 591 polygons (0.56% of all polygons)
    # which are false predictions in Lake Superior and Lake Michigan.
        prev_count = polygons.shape[0]
        polygons =  (
            polygons.sjoin(county_data[['geometry']])
            .drop(["index_right"], axis=1)
            .drop_duplicates()
        )
        if verbose:
            print(f'Offshore polygons dropped: {prev_count-polygons.shape[0]}')

    # add in the jpeg names if needed:
    if image_bound_map is not None:
        if county_data is None:
            print('County data must be supplied for correct jpeg assocation.')
            return
        if verbose:
            print('Linking to NAIP tiles...')
        # Omit fully-masked images
        image_bound_map = image_bound_map[~image_bound_map['all_black']]
        
        # If jpeg_names column already exists, drop it
        if 'jpeg_names' in polygons.columns:
            polygons.drop('jpeg_names', axis=1, inplace=True)
        if 'jpeg_name' in polygons.columns:
            polygons.drop('jpeg_name', axis=1, inplace=True)

        # Join polygons with counties. Note this will create duplicates for polygons which cross county boundaries.
        # We'll handle this later.
        county_data['COUNTY_NAM'] = county_data['COUNTY_NAM'].apply(lambda x: 'St. Croix' if x == 'Saint Croix' else x)
        polygons = (
            polygons.sjoin(county_data[['geometry', 'COUNTY_NAM']])
            .drop(["index_right"], axis=1)
        )

        # Join polygons with image bound map to create jpeg_names column
        polygons = (
            polygons.sjoin(image_bound_map)
            .rename(columns={"filename": "jpeg_name"})
            .drop(["index_right"], axis=1)
        )

        # Drop polygon-image associations where image is from a different county
        polygons['image_county'] = polygons['jpeg_name'].apply(lambda x: x[3 : x.find("_", 3)])
        polygons = polygons[polygons['image_county']==polygons['COUNTY_NAM']]

        # If polygon intersects multiple images, save the images in a list
        polygons = (
            polygons.groupby(['polygon_index', 'geometry'])
            .agg(jpeg_names = pd.NamedAgg(column='jpeg_name', aggfunc='unique'))
            .sort_values(by='polygon_index')
            .reset_index()
        )
        polygons = gpd.GeoDataFrame(polygons, geometry='geometry', crs=cfg.WI_EPSG)

        if save_newfile:
            # save the new file:
            polygons.to_file(filename=polygon_path)
    end_time = time.time()
    execution_time = end_time - start_time
    if verbose:
        print(f"Polygon Loading time -- Regular: {np.round(execution_time,3)}")

    return polygons


def save_clusters(clusters, file_name, analysis_output_path):
    """Save clusters and change the geometry col
    Input:
        cluster: dataframe
        file_name: str
        analysis_output_path: pointer path
    """
    clusters = clusters.rename(columns={"geometry": "geometry2"})
    clusters.to_csv(analysis_output_path / file_name, index=False)


def load_clusters(cluster_path):
    """Used after cluster.cluster.py creates initial clusters
    Input: path to clusters CSV file.
    Return: loaded cluster as a geo DataFrame
    """
    df = pd.read_csv(
        cluster_path,
        converters={"jpeg_names": literal_eval, "polygon_indices": literal_eval},
    )

    clusters = gpd.GeoDataFrame(
        df.loc[:, [c for c in df.columns if c != "geometry2"]],
        geometry=gpd.GeoSeries.from_wkt(df["geometry2"]),
        crs="epsg:3071",
    )
    # fix the parcel indices data type
    clusters["parcel_indices"] = clusters["parcel_indices"].apply(eval)

    print(f"Number of Clusters: {len(clusters)}")
    return clusters


def join_polygons_parcels(
    parcels: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame, overlap_factor: float = 0.25
):
    """Join polygons dataframe with parcels dataframe.
    Args:
        parcels: GeoDataFrame of parcels
        polygons: GeoDataFrame of polygons
        overlap_factor: determines polygon-parcel association. If the proportion of a polygon's area
            intersecting a given parcel is greater than this factor, the polygon is considered a part of that parcel.
            This parameter exists primarily to account for minor discrepancies between barn locations and imprecise
            parcel shapefiles.
    Returns: Merged GeoDataFrame with one row for each unique polygon.
    """
    if overlap_factor > 1 or overlap_factor < 0:
        print("Error: overlap factor must be in [0, 1].")
        return

    # Join each polygon to the one or more parcels that it intersects
    parcels["parcel_geom"] = parcels["geometry"]
    polygons_merged = polygons.sjoin(parcels, how="inner")

    # Retain only polygon-parcel associations with sufficient overlap
    def sufficient_overlap(x, y):
        overlap = x.intersection(y).area
        return overlap > x.area * overlap_factor

    polygons_merged = polygons_merged[
        polygons_merged.apply(
            lambda x: sufficient_overlap(x.geometry, x.parcel_geom), axis=1
        )
    ]

    polygons_merged = polygons_merged[
        [
            "geometry",
            "polygon_index",
            "parcel_index",
            "parcel_geom",
            "OWNERNME1",
            "OWNERNME2",
            "jpeg_names",
        ]
    ]

    # Calculate adjacent parcels for parcels intersecting polygons
    parcels = parcels[
        parcels["parcel_index"].isin(polygons_merged["parcel_index"])
    ].reset_index()
    parcels["adjacent_parcels"] = pd.Series()
    for i in tqdm(range(0, parcels.shape[0])):
        adjacent_parcels = parcels[
            parcels.geometry.intersects(parcels["geometry"].iloc[i])
        ].parcel_index.tolist()
        # Don't duplicate original parcel
        if parcels.iloc[i].parcel_index in adjacent_parcels:
            adjacent_parcels.remove(parcels.iloc[i].parcel_index)
        # If there are no adjacent parcels intersecting polygons, move on
        if not adjacent_parcels:
            parcels.at[i, "adjacent_parcels"] = set()
        else:
            parcels.at[i, "adjacent_parcels"] = set(adjacent_parcels)
    # Re-merge with polygons to collect adjacent parcels
    polygons_merged = polygons_merged.merge(
        parcels[["parcel_index", "adjacent_parcels"]], how="left", on="parcel_index"
    )
    # Aggregate parcel join data so we have one row per polygon
    polygons_merged = (
        polygons_merged.groupby(["polygon_index", "geometry"])
        .agg(
            jpeg_names=pd.NamedAgg(column='jpeg_names', aggfunc='first'),
            parcel_indices=pd.NamedAgg(column="parcel_index", aggfunc="unique"),
            adjacent_parcels=pd.NamedAgg(
                column="adjacent_parcels",
                aggfunc=lambda x: functools.reduce(set.union, x),
            ),
            parcel_geoms=pd.NamedAgg(
                column="parcel_geom",
                aggfunc=lambda x: [geom for geom in x],
            ),
            parcel_owner1_names=pd.NamedAgg(column="OWNERNME1", aggfunc="unique"),
            parcel_owner2_names=pd.NamedAgg(column="OWNERNME2", aggfunc="unique"),
        )
        .reset_index()
    )
    # Fix adjacent parcel formatting
    polygons_merged["adjacent_parcels"] = polygons_merged["adjacent_parcels"].apply(
        lambda x: [] if pd.isna(x) else list(x)
    )
    return polygons_merged


def is_fuzzy_name_match(
    owner_name_a: str,
    owner_name_b: str,
    fuzzy_threshold: float = 60,
    words_to_remove: list = cfg.WORDS_TO_REMOVE,
    common_names: list = cfg.COMMON_NAMES,
):
    """Remove general terms from owner names and then check string similarity.

    Args:
        owner_name_a: first parcel owner name
        owner_name_b: second parcel owner name
        fuzzy_threshold: similarity threshold for fuzzy match (see fuzzywuzzy.fuzz.ratio)
        words_to_remove: list of words to remove before calculating similarity

    Returns: (similarity score, bool whether owner names are similar)
    """
    # Re-sort words to remove from longest to shortest, in order to avoid string fragments
    words_to_remove = sorted(words_to_remove, key=len, reverse=True)

    owner_name_a = str(owner_name_a)
    owner_name_b = str(owner_name_b)
    
    # Remove extraneous string/list characters
    owner_name_a = (
        owner_name_a.replace("]", "")
        .replace("[", "")
        .replace("'", "")
        .replace('"', "")
        .replace(".", "")
        .replace(",", "")
    )
    owner_name_b = (
        owner_name_b.replace("]", "")
        .replace("[", "")
        .replace("'", "")
        .replace('"', "")
        .replace(".", "")
        .replace(",", "")
    )

    # remove key words
    for keyword in words_to_remove:
        owner_name_a = owner_name_a.replace(keyword, "")
        owner_name_b = owner_name_b.replace(keyword, "")
    # remove double spaces
    owner_name_a = owner_name_a.replace("  ", " ")
    owner_name_b = owner_name_b.replace("  ", " ")

    # Calculate levenshtein distance
    simi_score = fuzz.ratio(owner_name_a, owner_name_b)

    # Remove common names
    for name in common_names:
        owner_name_a = owner_name_a.replace(name, "")
        owner_name_b = owner_name_b.replace(name, "")

    # Collect lists of all words >2 characters long in each owner name
    owner_name_a_words = owner_name_a.split()
    owner_name_a_words = [
        owner_name_a_words[i]
        for i in range(len(owner_name_a_words))
        if len(owner_name_a_words[i]) > 2
    ]
    owner_name_b_words = owner_name_b.split()
    owner_name_b_words = [
        owner_name_b_words[i]
        for i in range(len(owner_name_b_words))
        if len(owner_name_b_words[i]) > 2
    ]

    # Calculate jaccard distance
    jaccard_score = td.jaccard(owner_name_a_words, owner_name_b_words)

    # Determine match
    match = False
    if jaccard_score > 0 or simi_score > fuzzy_threshold:
        match = True

    return jaccard_score, simi_score, match


def cluster(
    parcels: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    cluster_same_parcel: bool = True,
    fuzzy_name_match: bool = True,
    fuzzy_threshold: float = 50,
    cluster_adjacent_parcels_same_name: bool = True,
    same_name_distance_threshold: float = 500,
    cluster_distance_threshold: float = 150,
    words_to_remove=cfg.WORDS_TO_REMOVE,
):
    """
    Joins polygons on the same parcel and neighboring parcels into clusters.

    Inputs:
        parcels: parcel data
        polygons: polygon data
        cluster_same_parcel: bool, whether to cluster polygons on the same parcel
        fuzzy_name_match: bool, enables fuzzy name matching.
        fuzzy_threshold: How similar names must be (with 100 = identical strings) when fuzzy_name_match=True.
        cluster_adjacent_parcels_same_name: should polygons in adjacent parcels matched by name be clustered?
        same_name_distance_threshold: distance threshold in meters for clustering polygons in non-adjacent parcels with the same owner.
            If `None`, polygons with the same owner beyond `cluster_distance_threshold` will not be clustered.
        cluster_distance_threshold: distance threshold in meters for clustering polygons in different parcels, regardless of owner name. If `None`, polygons
            with different owners in different parcels will not be clustered.
        words_to_remove: list of strings to remove from owner names before fuzzy matching, if fuzzy_name_match = True.

    Outputs:
        GeoDataFrame of clusters, with associated geometry and lists of component polygons
    """
    if fuzzy_threshold < 0 or fuzzy_threshold > 100:
        print("Error: parameter 'fuzzy_threshold' must be a float between 0 and 100.")
        return
    if (cluster_distance_threshold is not None and cluster_distance_threshold < 0) or (
        same_name_distance_threshold is not None and same_name_distance_threshold < 0
    ):
        print("Error: distance thresholds must be positive.")
        return

    # 1. Merge and preprocess parcel data
    print("PREPROCESSING PARCEL DATA...")
    polygons_merged = join_polygons_parcels(parcels, polygons)
    print("PARCEL DATA PREPROCESSED")

    # 3. Calculate clusters using a Graph network
    print("CALCULATING CLUSTERS...")
    hist_name_simi_score = []
    # Create a graph with a node for each polygon
    polygon_graph = nx.Graph()
    for i in range(0, polygons_merged.shape[0]):
        polygon_graph.add_node(
            polygons_merged["polygon_index"].iloc[i],
            # Retain polygon geometry and NAIP jpeg name
            geometry=polygons_merged["geometry"].iloc[i],
            jpeg_names=polygons_merged["jpeg_names"].iloc[i],
            # Retain intersecting parcel indices, names, and geometries
            parcel_indices=list(polygons_merged["parcel_indices"].iloc[i]),
            parcel_owner1_names=list(polygons_merged["parcel_owner1_names"].iloc[i]),
            parcel_owner2_names=list(polygons_merged["parcel_owner2_names"].iloc[i]),
            parcel_geoms=polygons_merged["parcel_geoms"].iloc[i],
            adjacent_parcels=list(polygons_merged["adjacent_parcels"].iloc[i]),
        )

    # Loop through all polygons, connecting their nodes if they are in the same parcel
    # or optionally in adjacent parcels
    for polygon_a in tqdm(polygons_merged["polygon_index"]):
        polygon_a_geometry = polygon_graph.nodes[polygon_a]["geometry"]
        parcels_a = polygon_graph.nodes[polygon_a]["parcel_indices"]
        # Collect owner names for parcels associated with this polygon, omitting `None`
        parcels_a_owners = [
            polygon_graph.nodes[polygon_a]["parcel_owner1_names"],
            polygon_graph.nodes[polygon_a]["parcel_owner2_names"],
        ]
        parcels_a_owners = [i for i in parcels_a_owners if i[0] is not None]
        if cluster_same_parcel:
            # Connect polygons on same parcel
            same_parcel_polygons = polygons_merged[
                (polygons_merged["polygon_index"] != polygon_a)
                & (
                    polygons_merged["parcel_indices"].apply(
                        lambda x: len(set(list(x)).intersection(set(parcels_a))) > 0
                    )
                )
            ]["polygon_index"]
            for polygon_b in same_parcel_polygons:
                polygon_graph.add_edge(polygon_a, polygon_b)

        polygons_to_cluster = gpd.GeoDataFrame()
        if cluster_adjacent_parcels_same_name:
            # Collect polygons on adjacent parcels
            adjacent_parcels = polygon_graph.nodes[polygon_a]["adjacent_parcels"]
            if adjacent_parcels:
                adjacent_parcels = list(set(adjacent_parcels) - set(parcels_a))
                polygons_to_cluster = polygons_merged[
                    (
                        polygons_merged["parcel_indices"].apply(
                            lambda x: len(
                                set(list(x)).intersection(set(adjacent_parcels))
                            )
                            > 0
                        )
                    )
                ]

        if same_name_distance_threshold is not None:
            # Force algorithm to split a specific edge case of two permitted CAFOs ~477m of each other
            alternate_same_name_distance_threshold = same_name_distance_threshold
            if len(set(parcels_a).intersection({'2179078', '2179071', '2179077'})) > 0:
                alternate_same_name_distance_threshold = 400
            # Collect nearby polygons
            polygons_indices_to_cluster = polygons.sjoin_nearest(
                gpd.GeoDataFrame(geometry=gpd.GeoSeries(polygon_a_geometry), crs=3071),
                max_distance=min(same_name_distance_threshold, alternate_same_name_distance_threshold),
            )["polygon_index"]
            polygons_to_cluster = pd.concat(
                [
                    polygons_to_cluster,
                    polygons_merged[
                        polygons_merged["polygon_index"].isin(
                            polygons_indices_to_cluster
                        )
                    ],
                ]
            )

        # Cluster polygons based on parcel owner names.
        for i, row in polygons_to_cluster.iterrows():
            a_b_match = False
            # check if there is one name match between owners 1 and 2, and group a and b.
            for owner_name_a in parcels_a_owners:
                # If owner names are coded as not available, don't fuzzy match
                if owner_name_a == ['NOT AVAILABLE']:
                    continue
                
                # remove None values
                parcels_b_owners = [
                    row["parcel_owner1_names"],
                    row["parcel_owner2_names"],
                ]
                parcels_b_owners = [i for i in parcels_b_owners if i[0] is not None]

                for owner_name_b in parcels_b_owners:
                    if owner_name_a[0] == owner_name_b[0]:
                        a_b_match = True

                    elif fuzzy_name_match:
                        jaccard_score, simi_score, a_b_match = is_fuzzy_name_match(
                            owner_name_a, owner_name_b, fuzzy_threshold, words_to_remove
                        )
                        hist_name_simi_score.append(
                            [simi_score, owner_name_a, owner_name_b]
                        )
                    if a_b_match:
                        break  # stop checking if an a-b pair of names are similar

            if a_b_match:
                # print(owner_name_a, owner_name_b)
                polygon_graph.add_edge(polygon_a, row["polygon_index"])

        # Cluster polygons within a certain maximum distance, regardless of parcel owner names
        if cluster_distance_threshold is not None:
            more_polygons_indices_to_cluster = polygons.sjoin_nearest(
                gpd.GeoDataFrame(geometry=gpd.GeoSeries(polygon_a_geometry), crs=3071),
                max_distance=cluster_distance_threshold,
            )["polygon_index"]
            for polygon_b in more_polygons_indices_to_cluster:
                polygon_graph.add_edge(polygon_a, polygon_b)

    print("CLUSTERS CALCULATED")

    # 4. Collect and postprocess clusters
    # Extract the connected components of the graph as subgraphs
    cluster_subgraphs = [
        polygon_graph.subgraph(c).copy() for c in nx.connected_components(polygon_graph)
    ]

    # Save these subgraphs into a GeoDataFrame, along with their attributes
    clusters = gpd.GeoDataFrame(
        {"polygon_indices": pd.Series(cluster_subgraphs)},
        geometry=gpd.GeoSeries(),
        crs=3071,
    )
    clusters["parcel_indices"] = (
        pd.Series(cluster_subgraphs)
        .apply(lambda x: list(nx.get_node_attributes(x, "parcel_indices").values()))
        .apply(lambda x: list(set([item for sublist in x for item in sublist])))
    )
    clusters["geometry"] = pd.Series(cluster_subgraphs).apply(
        lambda x: shapely.unary_union(
            list(nx.get_node_attributes(x, "geometry").values())
        )
    )
    clusters["parcel_owner1_names"] = (
        pd.Series(cluster_subgraphs)
        .apply(
            lambda x: list(nx.get_node_attributes(x, "parcel_owner1_names").values())
        )
        .apply(lambda x: list(set([item for sublist in x for item in sublist])))
    )
    clusters["parcel_owner2_names"] = (
        pd.Series(cluster_subgraphs)
        .apply(
            lambda x: list(nx.get_node_attributes(x, "parcel_owner2_names").values())
        )
        .apply(lambda x: list(set([item for sublist in x for item in sublist])))
    )
    # Collecting parcel geoms is slightly tricky
    clusters["parcel_geoms"] = pd.Series(cluster_subgraphs).apply(
        lambda x: list(nx.get_node_attributes(x, "parcel_geoms").values())
    )
    clusters["parcel_geoms"] = (clusters["parcel_geoms"]
                                # Flatten list of parcel geoms, drop duplicates
                                .apply(lambda x: list(set([multipolygon for multipolygon_list in x for multipolygon in multipolygon_list])))
                                # Explode multipolygons into list of polygons
                                .apply(lambda x: [list(shapely.get_parts(multipolygon)) for multipolygon in x])
                                # Create single new multipolygon from list of polygons
                                .apply(lambda x: shapely.MultiPolygon([poly for poly_list in x for poly in poly_list]))
    )
    clusters["polygon_indices"] = clusters["polygon_indices"].apply(
        lambda x: list(x.nodes())
    )
    clusters["jpeg_names"] = pd.Series(cluster_subgraphs).apply(
        lambda x: list((nx.get_node_attributes(x, "jpeg_names").values()))
    )
    clusters["jpeg_names"] = clusters["jpeg_names"].apply(
        lambda x: list(set([jpeg_name for jpeg_name_list in x for jpeg_name in jpeg_name_list])))
    clusters["cluster_area_m2"] = clusters["geometry"].area

    print("Number of clusters: ", clusters.shape[0])

    return clusters


if __name__ == "__main__":
    # 0. Load paths from config file
    with open(Path().resolve().parent / "afo_vs_cafo/config/config.yml", "r") as file:
        configs = yaml.safe_load(file)
    data_path = Path(configs["data_path"])
    land_parcel_path = Path(configs['land_parcel_path'])
    cluster_path = Path(configs['cluster_path'])
    analysis_output_path = Path(configs['analysis_output_path'])

    # 1. Load and clean data
    print("LOADING DATA...")
    counties = gpd.read_file(data_path / 'County_Boundaries_24K/County_Boundaries_24K.shp')
    image_bound_map = gpd.read_file(analysis_output_path / 'image_bound_map.geojson')
    parcels = load_parcels(land_parcel_path / 'WI_parcels.feather', feather=True)

    outside_ewg_annotations = load_polygons(data_path / 'full_state_cf_annotations.geojson',
                                            counties, image_bound_map)
    inside_ewg_annotations = load_polygons(data_path / 'ewg_region_train_labels.geojson',
                                            counties, image_bound_map)
    all_annotations = pd.concat([outside_ewg_annotations, inside_ewg_annotations], axis=0)
    all_annotations['polygon_index'] = range(0, all_annotations.shape[0])
    print("ALL DATA LOADED")

    # 2. Run clustering algorithm
    all_cf_annotation_clusters = cluster(parcels, all_annotations)

    # 3. Save to disk
    save_clusters(all_cf_annotation_clusters, "all_CF_clusters.csv", cluster_path)
