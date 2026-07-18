"""Some simple tests that the FastApi server functions."""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}

def test_read_docs():
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger-ui" in response.text

def test_read_specimens():
    response = client.get("/v2/specimens")
    assert response.status_code == 200
    assert len(response.json()) > 0

def test_read_specimen():
    response = client.get("/v2/specimens")

    # test that we get some specimen data
    s = response.json()[0]
    response = client.get("/v2/specimens/" + s['uuid'] + "/data")
    assert response.status_code == 200
    assert response.json()['uuid'] == s['uuid']
    # could test that it validates against the schema...

    # test that we get an image for a specimen
    response = client.get("/v2/specimens/" + s['uuid'] + "/image")
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('image/')

def test_read_query():
    response = client.get("/v2/specimens?shape_type=outline")

    assert response.status_code == 200
    assert len(response.json()) > 0
