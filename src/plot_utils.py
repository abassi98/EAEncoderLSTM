import struct
import warnings
from pathlib import Path

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    import contextily as cx
except ImportError:
    cx = None

DEFAULT_RESULTS_DIR = Path("analysis/results_data")
DEFAULT_OUTPUT_DIR = Path("analysis/figures")
STATS_DIR = Path("analysis/stats/test")
BOOTSTRAP_DIR = Path("analysis/bootstrap")
BASIN_LIST_PATH = Path("data/basin_list.txt")

US_STATES_SHP = Path("data/usa-states-census-2014.shp")


def get_basemap_settings(basemap: str) -> dict[str, object] | None:
    basemap_settings = {
        "light": {
            "source": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            "zoom_adjust": 3,
        },
        "terrain": {
            "source": "https://tile.opentopomap.org/{z}/{x}/{y}.png",
            "zoom_adjust": 3,
        },
        "satellite": {
            "source": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "zoom_adjust": 3,
        },
    }

    if basemap not in basemap_settings:
        return None

    return basemap_settings[basemap]


def add_basemap(ax, basemap: str) -> None:
    if basemap == "none":
        return

    if cx is None:
        raise ImportError(
            "Basemap requested, but `contextily` is not installed in the active Python environment. "
            "Install `contextily` in the same environment used to run this script, or run with `--basemap none`."
        )

    basemap_settings = get_basemap_settings(basemap)
    cx.add_basemap(
        ax,
        source=basemap_settings["source"],
        crs="EPSG:4326",
        attribution=False,
        zoom="auto",
        zoom_adjust=basemap_settings["zoom_adjust"],
    )


def load_us_states(required: bool = True):
    if gpd is None:
        message = (
            "geopandas is not installed in the active Python environment, so the US states shapefile "
            "cannot be loaded."
        )
        if required:
            raise ImportError(message)
        warnings.warn(message)
        return None

    if not US_STATES_SHP.is_file():
        message = f"Missing US states shapefile at {US_STATES_SHP}."
        if required:
            raise FileNotFoundError(message)
        warnings.warn(message)
        return None

    return gpd.read_file(US_STATES_SHP)


def load_state_polygons_local(shp_path: Path | None = None) -> list:
    shp_path = shp_path or US_STATES_SHP
    polygons = []
    if not shp_path.is_file():
        return polygons

    with shp_path.open("rb") as shp_file:
        shp_file.seek(100)
        while True:
            record_header = shp_file.read(8)
            if len(record_header) < 8:
                break

            _, content_length_words = struct.unpack(">2i", record_header)
            content = shp_file.read(content_length_words * 2)
            if len(content) < 44:
                continue

            shape_type = struct.unpack("<i", content[:4])[0]
            # PolygonZ (15), used by our states file, has the same XY parts;
            # the additional elevation/measure arrays can be ignored for maps.
            if shape_type not in (3, 5, 15):
                continue

            num_parts, num_points = struct.unpack("<2i", content[36:44])
            parts_offset = 44
            points_offset = parts_offset + 4 * num_parts

            parts = []
            for part_idx in range(num_parts):
                start = parts_offset + 4 * part_idx
                parts.append(struct.unpack("<i", content[start:start + 4])[0])

            points = []
            for point_idx in range(num_points):
                start = points_offset + 16 * point_idx
                points.append(struct.unpack("<2d", content[start:start + 16]))

            for part_idx, start in enumerate(parts):
                stop = parts[part_idx + 1] if part_idx + 1 < num_parts else num_points
                part_points = points[start:stop]
                if len(part_points) >= 3:
                    polygons.append(part_points)

    return polygons


cmodels = {
    "pub": {
        0: "#e38050",
        1: "#c86448",
        2: "#984240",
        None: "#30210d",
    },
    "global": {
        0: "#d2d484",
        1: "#b5eacc",
        2: "#8ed9d1",
        #3: "#67c7d5",
        #4: "#54b2cc",
        26: "#3292c2",
        None: "#1e5fac",
    },
    "global_linear": {
        2: "#27F568",
    },
    "pub_linear": {
        2: "#E727F5",
    },
}


BOOTSTRAP_RANDOM_FEATURE_MODELS = (
    {
        "id": "global_ae_2",
        "family": "global",
        "file_experiment": "global",
        "encoded_features": 2,
        "color": "#6f4bf2",
        "label": "GLOBAL-RF-2",
    },
)


BOOTSTRAP_COMPARISON_FEATURES = {"global": (0, 1, 2, 4, 26), "pub": (0, 1, 2)}
BOOTSTRAP_ATTRIBUTE_FEATURES = {"global": (0, 1, 2, 4), "pub": (0, 1, 2)}


def get_bootstrap_comparison_pairs(experiment, kind):
    """Shared map/table pairs, oriented current minus previous or attributes minus encoded."""
    if kind == "previous":
        features = BOOTSTRAP_COMPARISON_FEATURES[experiment]
        pairs = [(current, previous) for previous, current in zip(features, features[1:])]
    elif kind == "attr":
        pairs = [("A", features) for features in BOOTSTRAP_ATTRIBUTE_FEATURES[experiment]]
    else:
        raise ValueError(f"Unknown bootstrap comparison kind: {kind}")
    return [(f"{experiment.upper()}-{a}", f"{experiment.upper()}-{b}") for a, b in pairs]


def format_bootstrap_model_label(experiment: str, encoded_features, random_features: bool = False) -> str:
    experiment_name = experiment.upper()
    if random_features:
        if encoded_features is None:
            return f"{experiment_name}-RF-A"
        return f"{experiment_name}-RF-{encoded_features}"

    if encoded_features is None:
        return f"{experiment_name}-A"
    return f"{experiment_name}-{encoded_features}"


def get_bootstrap_random_feature_model_ids() -> tuple[str, ...]:
    """Return the CLI-selectable random-feature bootstrap model IDs."""
    return tuple(spec["id"] for spec in BOOTSTRAP_RANDOM_FEATURE_MODELS)


def get_bootstrap_plot_models(
    random_feature_models: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    """Return standard models plus explicitly selected random-feature models."""
    specs = []

    for experiment, encoded_features_dict in cmodels.items():
        for encoded_features, color in encoded_features_dict.items():
            specs.append(
                {
                    "family": "pub" if "pub" in experiment else "global",
                    "file_experiment": experiment,
                    "display_experiment": experiment,
                    "encoded_features": encoded_features,
                    "color": color,
                    "label": format_bootstrap_model_label(experiment, encoded_features),
                    "random_features": False,
                }
            )

    selected_random_feature_models = set(random_feature_models)
    for spec in BOOTSTRAP_RANDOM_FEATURE_MODELS:
        if spec["id"] in selected_random_feature_models:
            specs.append(
                {
                    **spec,
                    "display_experiment": spec["family"],
                    "label": spec.get("label")
                    or format_bootstrap_model_label(
                        spec["family"],
                        spec["encoded_features"],
                        random_features=True,
                    ),
                    "random_features": True,
                }
            )

    return specs


