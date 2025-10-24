"""Proof of concept of echoSMs anatomical data store RESTful API using FastAPI."""

from fastapi import FastAPI, Query, HTTPException, Path as fPath
from fastapi.responses import Response, FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import numpy as np
from typing import Annotated
from pathlib import Path
import orjson
import pandas as pd
from datetime import datetime as dt
from stat import S_IFDIR, S_IFREG
from stream_zip import ZIP_64, stream_zip
from echosms import plot_specimen
from urllib.request import urlretrieve
from zipfile import ZipFile
import os
import shutil


cdn_url = 'https://echosms-datastore.syd1.cdn.digitaloceanspaces.com/'
schema_url = 'https://ices-tools-dev.github.io/echoSMs/schema/data_store_schema/'

zipfile = Path('echosms_datastore_final.zip')
metadata_filename = 'metadata_all_autogen.json'

datasets_dir = zipfile.with_suffix('')

favicon_path = 'favicon.ico'

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
# now prepare the data for use via the API

# Replace nan with None - done like this because .fillna(None) doesn't work.
# None is needed because fastAPI won't serialise np.nan into JSON and the alternative
# of converting to '' causes problems with using the data later on (one gets
# columns with a mixture of numbers and text).
df = pd.DataFrame(all_datasets).fillna(np.nan).replace([np.nan], [None])

del all_datasets

####################################################################################################
app = FastAPI(title='The echoSMs web API',
              openapi_tags=[{'name': 'v2',
                             'description': 'Provides data via a flat specimen structure'},])


# /v2/specimens endpoint query parameter definitions via a Pydantic model
class SpecimenQuery_v2(BaseModel):  # noqa
    species: str | None = Field(None, title='Species', description="The scientific species name")
    id: str | None = Field(None, title='Specimen ID', description="The specimen ID")
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


####################################################################################################
@app.get("/v2/specimens",
         summary="Get specimen metadata with optional filtering. Does not return shapes.",
         response_description='A list of specimen metadata',
         tags=['v2'])
async def get_specimens_v2(query: Annotated[SpecimenQuery_v2, Query()]):  # noqa
        # Return all specimens if no query parameters are given
        if not query.model_fields_set:
            return df.loc[:, df.columns != 'shapes'].to_dict(orient='records')

        # Buuld a DataFrame query string from the query parameters
        # attr is a tuple of (query_parameter, value)
        q = [f"{attr[0]} == '{attr[1]}'" for attr in query if attr[1] is not None]

        return df.query(' & '.join(q)).loc[:, df.columns != 'shapes'].to_dict(orient='records')


####################################################################################################
@app.get("/v2/specimen/{id}/data",
         summary='Get all specimen data with the given id',
         response_description='Specimen data structured as per the echoSMs data '
                              f'store [schema]({schema_url})',
         tags=['v2'])
async def get_specimen_shape_v2(id: Annotated[str, fPath(description='The specimen ID')]):  # noqa

    s = specimen(id)
    if not s:
        raise HTTPException(status_code=404, detail=f'Specimen {id} not found')

    return s


####################################################################################################
@app.get("/v2/specimen/{id}/image",
         summary='Get an image of the specimen shape with the given id',
         response_description='An image of the specimen shape',
         tags=['v2'],
         response_class=Response,
         responses={200: {'content': {'image/png': {}}}})
async def get_specimen_image_v2(id: Annotated[str, fPath(description='The specimen ID')]):  # noqa

    image_file = (datasets_dir/id).with_suffix('.png')

    if not image_file.exists():
        s = specimen(id)
        if not s:
            raise HTTPException(status_code=404, detail=f'Specimen {id} not found')

        print('Making and caching the image')
        plot_specimen(s, title=id, savefile=image_file, dpi=200)

    return FileResponse(image_file)


####################################################################################################
@app.get("/v2/dataset/{dataset_id}/all",
         summary='Get all data with the given dataset_id, including any raw data',
         response_description='A zipped file containing all data for the dataset',
         tags=['v2'])
async def get_dataset(dataset_id: Annotated[str, fPath(description='The dataset ID')]):  # noqa

    return {"message": "Not yet implemented"}

    # The plan: zip up all files in the directory with the same name as the given
    # dataset_id. If such a directory doesn't exist, raise HTTPException

    # zip up the dataset and stream out
    return StreamingResponse(stream_zip(get_dir_items(datasets_dir/dataset_id)),
                             media_type='application/zip',
                             headers={'Content-Disposition':
                                      f'attachment; filename={dataset_id}.zip'})

####################################################################################################
@app.get("/v2/last-updated",
         summary='Date of most recent datastore contents update',
         response_description='The date when the datastore contents were last updated',
         tags=['v2'])
async def last_updated():  # noqa

    # Using the most recent date from the datasets might not work that well - it relies
    # on that field in the datasets being updated. Consider maintaining a separate
    # last updated time, independent of individual datasets.
    return max(df.date_last_modified)


####################################################################################################
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():  # noqa
    return FileResponse(favicon_path)

#============================================================================
# Helper functions

def specimen(sid):
    """Find specimen with given id, reading the shape from file if needed."""
    s = df.query(f"id == '{sid}'")

    if s.empty:
        return None

    sp = s.to_dict(orient='records')[0]

    # If the shape is not in df (because it is large), load it
    if isinstance(sp['shapes'], str):
        with open(datasets_dir/sp['shapes'], 'r') as f:
            json_bytes = f.read()  # loads it all into memory
            sp['shapes'] = orjson.loads(json_bytes)

    return sp

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
