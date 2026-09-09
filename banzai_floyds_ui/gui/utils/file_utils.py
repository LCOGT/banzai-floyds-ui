from astropy.io import fits
from django.conf import settings
import asyncio
import httpx
import io
import requests
from django.core.cache import cache
from io import BytesIO
import pickle


def _open_fits(data):
    return fits.open(io.BytesIO(data))


def _data_url(response, list_endpoint):
    if list_endpoint:
        return response.json()['results'][0]['url']
    return response.json()['url']


async def fetch(client, url, params, headers):
    response = await client.get(url, params=params, headers=headers)
    return response


async def fetch_all(archive_header, request_params):
    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = [fetch(client, settings.ARCHIVE_URL, params, archive_header) for params in request_params]
        return await asyncio.gather(*tasks)


async def async_download_frame(client, headers, url=f'{settings.ARCHIVE_URL}', params=None, list_endpoint=False):
    response = await client.get(url, params=params, headers=headers)
    response.raise_for_status()

    data_response = await client.get(_data_url(response, list_endpoint))
    data_response.raise_for_status()
    return _open_fits(data_response.content)


async def async_get_related_frame(client, frame_id, archive_header, related_frame_key):
    response = await client.get(f'{settings.ARCHIVE_URL}{frame_id}/headers', headers=archive_header)
    response.raise_for_status()

    related_frame_filename = response.json()['data'][related_frame_key]
    frame = await async_download_frame(
        client,
        archive_header,
        params={'basename_exact': related_frame_filename},
        list_endpoint=True
    )
    return frame, related_frame_filename


async def async_fetch_plot_frame(frame_id, archive_header, related_frame_key=None):
    async with httpx.AsyncClient(follow_redirects=True) as client:
        if related_frame_key is not None:
            return await async_get_related_frame(client, frame_id, archive_header, related_frame_key)
        return await async_download_frame(client, archive_header, url=f'{settings.ARCHIVE_URL}{frame_id}/')


def download_frame(headers, url=f'{settings.ARCHIVE_URL}', params=None, list_endpoint=False):
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()

    data = requests.get(_data_url(response, list_endpoint)).content
    return _open_fits(data)


def get_related_frame(frame_id, archive_header, related_frame_key):
    # Get the related frame from the archive that matches related_frame_key in the header.
    response = requests.get(f'{settings.ARCHIVE_URL}{frame_id}/headers', headers=archive_header)
    response.raise_for_status()
    related_frame_filename = response.json()['data'][related_frame_key]
    params = {'basename_exact': related_frame_filename}
    return download_frame(archive_header, params=params, list_endpoint=True), related_frame_filename


def cache_fits(key_name, hdulist, timeout=None):
    buffer = BytesIO()
    hdulist.writeto(buffer)
    buffer.seek(0)
    cache.set(key_name, buffer.read(), timeout=timeout)


def get_cached_fits(key_name):
    cached_value = cache.get(key_name)
    if cached_value is None:
        return None
    buffer = BytesIO(cached_value)
    buffer.seek(0)
    return fits.open(buffer)


def get_cached_frame(key_name):
    cached_value = cache.get(key_name)
    if cached_value is None:
        return None
    buffer = BytesIO(cached_value)
    buffer.seek(0)
    return pickle.load(buffer)


def cache_frame(key_name, frame, timeout=None):
    buffer = BytesIO()
    pickle.dump(frame, buffer)
    buffer.seek(0)
    cache.set(key_name, buffer.read(), timeout=timeout)
