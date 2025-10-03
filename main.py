"""Proof of concept of echoSMs anatomical data store RESTful API using FastAPI."""

from fastapi import FastAPI, Query, Path as fPath
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
import numpy as np
import io
from typing import Annotated
from pathlib import Path
import json
import pandas as pd
from datetime import datetime as dt
from stat import S_IFDIR, S_IFREG
from stream_zip import ZIP_64, stream_zip


base_dir = Path(r'C:\Users\GavinMacaulay\OneDrive - Aqualyd Limited\Documents\Aqualyd'
                r'\Projects\2024-05 NOAA modelling\working\anatomical data store')
base_dir = Path('.')
datasets_dir = base_dir/'datasets'

#datasets_dir = Path('.')/'data_store'/'resources'

with open(datasets_dir/'all-datasets-automatically-generated.json', 'r') as f:
    all_datasets = json.load(f)

# make a Pandas version of the dataset attributes that can be searched through easily
searchable_attrs = ['dataset_id', 'species', 'imaging_method', 'model_type',
                    'anatomical_category', 'shape_method', 'aphiaID']
searchable_data = [{key: d[key] for key in searchable_attrs if key in d} for d in all_datasets]
df = pd.DataFrame(searchable_data).set_index('dataset_id')

# For the v2 API, make a dataframe with one row per specimen and the columns containing 
# all dataset and specimen metadata, excluding the shapes.
rows = []
for d in all_datasets:
    for s in d['specimens']:
        row = {'id': d['dataset_id'] + '_' + s['specimen_id']} | d | s
        # Remove unneeded columns in the flattened version
        for r in ['specimens', 'shapes', 'shape_types']:
            row.pop(r)
        rows.append(row)

df_flat = pd.DataFrame(rows).fillna('')

schema_url = 'https://ices-tools-dev.github.io/echoSMs/schema/data_store_schema/'

app = FastAPI(title='The echoSMs web API',
              openapi_tags=[{'name': 'v1',
                             'description': 'Provides data via a dataset/specimen structure'},
                            {'name': 'v2',
                             'description': 'Provides data via a flat specimen structure'},])

@app.get("/v1/datasets",
         summary="Get dataset_ids with optional filtering",
         response_description='A list of dataset_ids',
         tags=['v1'])
async def get_datasets(species: Annotated[str | None, Query(  # noqa
                           title='Species',
                           description="The scientific species name")] = None,
                       imaging_method: Annotated[str | None, Query(
                           title='Imaging method',
                           description="The imaging method used")] = None,
                       model_type: Annotated[str | None, Query(
                           title='Model type',
                           description="The model type used")] = None,
                       anatomical_category: Annotated[str | None, Query(
                           title='Anatomical category',
                           description="The anatomical category")] = None,
                       shape_method: Annotated[str | None, Query(
                           title='Shape method',
                           description="The shape method")] = None,
                       aphiaID: Annotated[int | None, Query(
                           title='AphiaID',
                           description='The [aphiaID](https://www.marinespecies.org/aphia.php)')]
                               = None):

    q = ''
    for attr in searchable_attrs[1:]:  # excludes 'dataset_id'
        q += '' if eval(attr) is None else f'{attr} == @{attr} & '

    if len(q) == 0:
        return df.index.tolist()

    return df.query(q[:-3]).index.tolist()


@app.get("/v1/dataset/{dataset_id}",
         summary='Get the dataset with the given dataset_id',
         response_description='A dataset structured as per the echoSMs data store '
                              f'[schema]({schema_url})',
         tags=['v1'])
async def get_dataset(dataset_id: Annotated[str, fPath(description='The dataset ID')], # noqa
                      full_data: Annotated[bool, Query(description='If true, all raw data for the '
                                    'dataset will be returned as a zipped file')] = False):

    ds = get_ds(dataset_id)
    if not ds:
        return {"message": "Dataset not found"}

    if full_data:
        return {"message": "Not available on this testing server"}
        # zip up the dataset and stream out
        return StreamingResponse(stream_zip(get_dir_items(datasets_dir/dataset_id)),
                                 media_type='application/zip',
                                 headers={'Content-Disposition':
                                          f'attachment; filename={dataset_id}.zip'})
    return ds[0]


@app.get("/v1/specimens/{dataset_id}",
         summary='Get the specimen_ids from the dataset with the given dataset_id',
         response_description='A list of specimen_ids',
         tags=['v1'])
async def get_specimens(dataset_id: Annotated[str, fPath(description='The dataset ID')]): # noqa

    ds = get_ds(dataset_id)
    if not ds:
        return {"message": "Dataset not found"}

    return [s['specimen_id'] for s in ds[0]['specimens']]


@app.get("/v1/specimen/{dataset_id}/{specimen_id}",
         summary='Get specimen data with the given dataset_id and specimen_id',
         response_description='A specimen dataset structured as per the echoSMs data '
                              f'store [schema]({schema_url})',
         tags=['v1'])
async def get_specimen(dataset_id: Annotated[str, fPath(description='The dataset ID')], # noqa
                       specimen_id: Annotated[str, fPath(description='The specimen ID')]):

    ds = get_ds(dataset_id)
    if not ds:
        return {"message": "Dataset not found"}

    return get_sp(ds[0], specimen_id)


@app.get("/v1/specimen_image/{dataset_id}/{specimen_id}",
         summary='Get an image of the specimen shape, with the given dataset_id and specimen_id',
         response_description='An image of the specimen shape',
         tags=['v1'],
         response_class=Response,
         responses={200: {'content': {'image/png': {}}}})
async def get_specimen_image(dataset_id: Annotated[str, fPath(description='The dataset ID')], # noqa
                             specimen_id: Annotated[str, fPath(description='The specimen ID')]):

    ds = get_ds(dataset_id)
    if ds:
        s = get_sp(ds[0], specimen_id)
        if s:
            img = plot_specimen(s[0], dataset_id=ds[0]['dataset_id'], stream=True)
            return Response(img, media_type="image/png")

#============================================================================
# An alternative way to access the specimens, without using datasets
# These are all under the /v2 path

# /v2/specimens endpoint query parameter definitions via a Pydantic model
class SpecimenQuery_v2(BaseModel):
    species: str | None = Field(None, title='Species', description="The scientific species name")
    dataset_id: str | None = Field(None, title='Dataset ID', description="The dataset ID")
    length_type: str | None = Field(None, title='Length type', description="The length type")
    anatomical_category: str | None = Field(None, title='Anatomical category',
                                           description="The anatomical category")
    family: str | None = Field(None, title='Family', description="The scientific family name")                                           
    genus: str | None = Field(None, title='Genus', description="The scientific genus name")
    verncaular_name: str | None = Field(None, title='Vernacular name', description="The common name")
    activity_name: str | None = Field(None, title='Activity name', description="The activity name")
    sound_speed_method: str | None = Field(None, title='Sound speed method',
                                          description="The sound speed method")
    mass_density_method: str | None = Field(None, title='Mass density method',
                                           description="The mass density method")
    sex: str | None = Field(None, title='Sex of the organism', description='The sex of the organism')
    imaging_method: str | None = Field(None, title='Imaging method', description="The imaging method used")
    specimen_condition: str | None = Field(None, title='Specimen condition',
                                            description="The specimen condition")                                      
    model_type: str | None = Field(None, title='Model type', description="The model type used")
    shape_type: str | None = Field(None, title='Shape type', description="The shape type used")
    anatomical_category: str | None = Field(None, title='Anatomical category',
                                           description="The anatomical category")
    shape_method: str | None = Field(None, title='Shape method', description="The shape method")
    aphiaID: int | None = Field(None, title='AphiaID',
                               description='The [aphiaID](https://www.marinespecies.org/aphia.php)')

@app.get("/v2/specimens",
         summary="Get specimen metadata with optional filtering. Does not return shapes.",
         response_description='A list of specimen metadata',
         tags=['v2'])
async def get_specimens_v2(query: Annotated[SpecimenQuery_v2, Query()]):
        # Return all specimens if no query parameters are given
        if not query.model_fields_set:
            return df_flat.to_dict(orient='records')

        # Buuld a DataFrame query string from the query parameters
        # attr is a tuple of (query_parameter, value)
        q = [f"{attr[0]} == '{attr[1]}'" for attr in query if attr[1] is not None]

        return df_flat.query(' & '.join(q)).to_dict(orient='records')


@app.get("/v2/specimen/{id}/shape",
         summary='Get specimen shape with the given id',
         response_description='A specimen shape structured as per the echoSMs data '
                              f'store [schema]({schema_url})',
         tags=['v2'])
async def get_specimen_shape_v2(id: Annotated[str, fPath(description='The specimen ID')]):

    s = get_sp_from_id(id)
    if not s:
        return {"message": "Specimen not found"}

    return s[0]['shapes']


@app.get("/v2/specimen/{id}/image",
         summary='Get an image of the specimen shape with the given id',
         response_description='An image of the specimen shape',
         tags=['v2'],
         response_class=Response,
         responses={200: {'content': {'image/png': {}}}})
async def get_specimen_image_v2(id: Annotated[str, fPath(description='The specimen ID')]):

    s = get_sp_from_id(id)
    if not s:
        return {"message": "Specimen not found"}

    img = plot_specimen(s[0], title=id, stream=True)
    return Response(img, media_type="image/png")


#============================================================================
# Helper functions

def get_ds(dataset_id):
    """Find datasets with given dataset_id."""
    return [ds for ds in all_datasets if ds['dataset_id'] == dataset_id]


def get_sp(ds, specimen_id):
    """Find specimen with given specimen_id in the given dataset."""
    return [s for s in ds['specimens'] if s['specimen_id'] == specimen_id]

def get_sp_from_id(id):
    """Find specimen with given id in the flattened dataframe."""
    s = df_flat.query(f"id == '{id}'")

    if s.empty:
        return None

    ds = get_ds(s['dataset_id'].iloc[0])

    if not ds:
        return None

    return get_sp(ds[0], s['specimen_id'].iloc[0])

def plot_specimen(specimen, dataset_id='', title='', stream=False):
    """Plot the specimen shape."""
    match specimen['shape_type']:
        case 'outline':
            fig, axs = plt.subplots(2, 1, sharex=True, layout='constrained')
            plot_shape_outline(specimen['shapes'], axs)
            axs[0].text(0, 1.05, 'Dorsal', transform=axs[0].transAxes)
            axs[1].text(0, 1.05, 'Lateral', transform=axs[1].transAxes)
            fig.supxlabel('[mm]')
            fig.supylabel('[mm]')
        case 'surface':
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
            plot_shape_surface(specimen['shapes'], ax)
            plt.tight_layout()
        case 'voxels':
            fig, axs = plt.subplots(2, 1, sharex=True, layout='constrained')
            plot_shape_voxels(specimen['shapes'], axs)

    t = title if title else dataset_id + ' ' + specimen['specimen_id']
    fig.suptitle(t)

    if stream:
        with io.BytesIO() as buffer:
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            return buffer.getvalue()
    else:
        plt.show()


def plot_shape_outline(shapes, axs):
    """Plot the specimen's outline shape."""
    for s in shapes:
        c = 'C0' if s['boundary'] == 'fluid' else 'C1'
        x = np.array(s['x'])*1e3
        z = np.array(s['z'])*1e3
        y = np.array(s['y'])*1e3
        width_2 = np.array(s['width'])*1e3/2
        zU = (z + np.array(s['height'])*1e3/2)
        zL = (z - np.array(s['height'])*1e3/2)

        # Dorsal view
        axs[0].plot(x, y, c='grey', linestyle='--', linewidth=1)  # centreline
        axs[0].plot(x, y+width_2, c=c)
        axs[0].plot(x, y-width_2, c=c)

        # Lateral view
        axs[1].plot(x, z, c='grey', linestyle='--', linewidth=1)  # centreline
        axs[1].plot(x, zU, c=c)
        axs[1].plot(x, zL, c=c)

        # close the ends of the shapes
        for i in [0, -1]:
            axs[1].plot([x[i], x[i]], [zU[i], zL[i]], c=c)
            axs[0].plot([x[i], x[i]], [(y+width_2)[i], (y-width_2)[i]], c=c)
            axs[i].set_aspect('equal')
            axs[i].xaxis.set_inverted(True)


def plot_shape_surface(shapes, ax):
    """Plot the specimen's 3D triangular shape."""
    for s in shapes:
        # c = 'C0' if s['boundary'] == 'fluid' else 'C1'
        facets = np.array([s['facets_0'], s['facets_1'], s['facets_2']]).transpose()
        x = 1e3 * np.array(s['x'])
        y = 1e3 * np.array(s['y'])
        z = 1e3 * np.array(s['z'])

        ax.plot_trisurf(x, y, z, triangles=facets)
        ax.view_init(elev=210, azim=-60, roll=0)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.set_aspect('equal')
        ax.xaxis.set_inverted(True)
        ax.yaxis.set_inverted(True)


def plot_shape_voxels(s, axs):
    """Plot the specimen's voxels."""
    pass


def get_dir_items(base_path: Path):
    """Create an iterable of file/directory info for use by stream-zip."""
    for item in base_path.rglob('*'):
        a_name = item.relative_to(base_path).as_posix()  # path within the zip archive
        # need a tuple of (archive_name, modified_time, mode, compression_method, data_source)
        # For directories, data_source must be empty
        if item.is_file():
            with open(item, 'rb') as f:
                yield (a_name, dt.fromtimestamp(item.stat().st_mtime),
                       S_IFREG | 0o644,  # regular file with read/write permissions
                       ZIP_64, (chunk for chunk in iter(lambda: f.read(65536*64), b'')))
        elif item.is_dir():
            yield (a_name + '/',  # trailing slash for directories
                   dt.fromtimestamp(item.stat().st_mtime), S_IFDIR | 0o755, ZIP_64, ())
