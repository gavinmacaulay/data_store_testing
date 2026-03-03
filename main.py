"""Proof of concept of echoSMs anatomical data store RESTful API using FastAPI."""

from fastapi import FastAPI, Query, HTTPException, Path as fPath
from fastapi.responses import Response, FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Annotated
from pathlib import Path
import orjson
import jmespath
from datetime import datetime as dt
from stat import S_IFDIR, S_IFREG
from stream_zip import ZIP_64, stream_zip
from urllib.request import urlretrieve
from zipfile import ZipFile
import os
import shutil


cdn_url = 'https://echosms-datastore.syd1.cdn.digitaloceanspaces.com/'
schema_url = 'https://ices-tools-dev.github.io/echoSMs/schema/data_store_schema/'

zipfile = Path('echosms_datastore_final.zip')
metadata_filename = 'metadata_all_autogen.json'

datasets_dir = zipfile.with_suffix('')

favicon_path = 'echoSMs_logo_auto_colour.svg'

# If woring on a DigitalOcean droplet, set to download datastore from URL, else
# expect the data to be available in a local directory
from_url = True if os.getenv('HOME') == '/workspace' else False

# Note. This initialising code gets run twice sometimes - this needs to be fixed by using
# FastAPI lifespan events.

"""Obtain the datastore and load into memory."""
if from_url:
    print('Deleting old local datastore (if present)')
    zipfile.unlink(missing_ok=True)
    shutil.rmtree(datasets_dir, ignore_errors=True)

    print('Downloading datastore data')
    urlretrieve(cdn_url + str(zipfile), filename=zipfile)

    print('Uncompressing datastore data')
    with ZipFile(zipfile, 'r') as zip_object:
        zip_object.extractall(datasets_dir)

    zipfile.unlink()

print('Loading datastore from local files')
with open(datasets_dir/metadata_filename, 'rb') as f:
    json_bytes = f.read()
    all_datasets = orjson.loads(json_bytes)


####################################################################################################
app = FastAPI(title='The echoSMs web API',
              openapi_tags=[{'name': 'v2',
                             'description': ''},])


# /v2/specimens endpoint query parameter definitions via a Pydantic model
class SpecimenQuery_v2(BaseModel):  # noqa
    species: str | None = Field(None, title='Species', description="The scientific species name")
    uuid: str | None = Field(None, title='Specimen UUID', description="The specimen UUID")
    specimen_name: str | None = Field(None, title='Specimen name', description="The specimen name")
    dataset_uuid: str | None = Field(None, title='Dataset UUID', description="The dataset UUID")
    dataset_name: str | None = Field(None, title='Dataset name', description="The dataset name")
    family: str | None = Field(None, title='Family', description="The scientific family name")
    genus: str | None = Field(None, title='Genus', description="The scientific genus name")
    activity_name: str | None = Field(None, title='Activity name', description="The activity name")
    sex: str | None = Field(None, title='Sex of the organism', description='The sex of the organism')
    imaging_method: str | None = Field(None, title='Imaging method', description="The imaging method used")
    specimen_condition: str | None = Field(None, title='Specimen condition',
                                            description="The specimen condition")
    model_type: str | None = Field(None, title='Model type', description="The model type used")
    shape_type: str | None = Field(None, title='Shape type', description="The shape type used")
    shape_method: str | None = Field(None, title='Shape method', description="The shape method")
    vernacular_names: str | None = Field(None, title='Vernacular name',
                                         description="A vernacular name")
    anatomical_category: str | None = Field(None, title='Anatomical category',
                                            description="The anatomical category")
    anatomical_feature: str | None = Field(None, title='Anatomical feature', 
                                description="Specimen contains a shape with this anatomical feature")
    boundary: str | None = Field(None, title='Shape boundary',
                                 description="The shape boundary")
    version_investigators: str | None = Field(None, title='Investigator name',
                                description="An investigator name")
    aphia_id: int | None = Field(None, title='AphiaID',
                               description='The [aphiaID](https://www.marinespecies.org/aphia.php)')

# Hacky way to indicate how some attributes should be treated when they are queried for
nested = {'anatomical_feature': 'shapes',
          'boundary': 'shapes'}

number = {'aphia_id'}

array = ['version_investigators', 'vernacular_names']

####################################################################################################
@app.get("/v2/specimens",
         summary="Get specimen metadata with optional filtering. Does not return shape data.",
         response_description='A list of specimen metadata',
         tags=['v2'])
async def get_specimens_v2(query: Annotated[SpecimenQuery_v2, Query()]):  # noqa
        # Return all specimens if no query parameters are given
        if not query.model_fields_set:
            return all_datasets

        # Build a jmespath query string from the query parameters
        q = []
        for attr in query:  # attr is a tuple of (query_parameter, value)
            if attr[1] is None:
                continue

            if attr[0] in array:
                q.append(f"{attr[0]}[?contains(@, '{attr[1]}')]")
                continue

            # Can't currently have nested arrays
            if attr[0] in nested:
                q.append(f"{nested[attr[0]]}[?{attr[0]} == '{attr[1]}']")
                continue

            if attr[0] in number:
                q.append(f"{attr[0]} == `{attr[1]}`")
                continue

            # A normal top level attribute
            q.append(f"{attr[0]} == '{attr[1]}'")
            
        specimens = jmespath.search('[?' + ' && '.join(q) + ']', all_datasets)

        # remove shape data except for some of the metadata
        for sp in specimens:
            s_metadata = []
            for s in sp['shapes']:
                ss = {k: v for k, v in s.items()
                        if k in ['anatomical_feature', 'name', 'boundary']}
                s_metadata.append(ss)
            sp['shapes'] = s_metadata

        return specimens


####################################################################################################
@app.get("/v2/specimen/{uuid}/data",
         summary='Get all specimen data with the given UUID',
         response_description='Specimen data structured as per the echoSMs data '
                              f'store [schema]({schema_url})',
         tags=['v2'])
async def get_specimen_shape_v2(uuid: Annotated[str, fPath(description='The specimen UUID')]):  # noqa

    s = specimen(uuid)
    if not s:
        raise HTTPException(status_code=404, detail=f'Specimen {uuid} not found')

    return s


####################################################################################################
@app.get("/v2/specimen/{uuid}/image",
         summary='Get an image of the specimen shape with the given UUID',
         response_description='An image of the specimen shape',
         tags=['v2'],
         response_class=Response,
         responses={200: {'content': {'image/png': {}}}})
async def get_specimen_image_v2(uuid: Annotated[str, fPath(description='The specimen UUID')]):  # noqa

    image_file = Path(f'{datasets_dir/uuid}.png')
    return FileResponse(image_file)


####################################################################################################
@app.get("/v2/dataset/{dataset_uuid}/all",
         summary='Get all data with the given dataset_uuid, including any raw data',
         response_description='A zipped file containing all data for the dataset',
         tags=['v2'])
async def get_dataset(dataset_uuid: Annotated[str, fPath(description='The dataset UUID')]):  # noqa

    return {"message": "Not yet implemented"}

    # The plan: zip up all files in the directory with the same name as the given
    # dataset_uuid. If such a directory doesn't exist, raise HTTPException

    # zip up the dataset and stream out
    return StreamingResponse(stream_zip(get_dir_items(datasets_dir/dataset_uuid)),
                             media_type='application/zip',
                             headers={'Content-Disposition':
                                      f'attachment; filename={dataset_uuid}.zip'})

####################################################################################################
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():  # noqa
    return FileResponse(favicon_path, media_type="image/svg+xml")

#============================================================================
# Helper functions

def specimen(sid):
    """Find specimen with given uuid, reading the shape from file if needed."""
    s = jmespath.search(f"[?uuid == '{sid}']", all_datasets)

    if not s:
        return None

    s = s[0]

    # If the shape is not in all_datasets (because it is large), load it
    if ref_key := 'large_shape_ref' in s:
        if isinstance(s[ref_key], str):
            with open(datasets_dir/s[ref_key], 'r') as f:
                json_bytes = f.read()  # loads it all into memory
                s['shapes'] = orjson.loads(json_bytes)
            del s[ref_key]

    return s

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
