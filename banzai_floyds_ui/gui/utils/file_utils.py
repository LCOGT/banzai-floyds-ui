from astropy.io import fits
from django.conf import settings
import asyncio
import httpx
import io
import requests


async def fetch(url, params, headers):
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, headers=headers)
    return response


async def fetch_all(archive_header, request_params):
    tasks = [fetch(settings.ARCHIVE_URL, params, archive_header) for params in request_params]
    return await asyncio.gather(*tasks)


def download_frame(headers, url=f'{settings.ARCHIVE_URL}', params=None, list_endpoint=False):
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()

    buffer = io.BytesIO()
    if list_endpoint:
        buffer.write(requests.get(response.json()['results'][0]['url'], stream=True).content)
    else:
        buffer.write(requests.get(response.json()['url'], stream=True).content)
    buffer.seek(0)

    hdu = fits.open(buffer)
    return hdu


def get_related_frame(frame_id, archive_header, related_frame_key):
    # Get the related frame from the archive that matches related_frame_key in the header.
    response = requests.get(f'{settings.ARCHIVE_URL}{frame_id}/headers', headers=archive_header)
    response.raise_for_status()
    related_frame_filename = response.json()['data'][related_frame_key]
    params = {'basename_exact': related_frame_filename}
    return download_frame(archive_header, params=params, list_endpoint=True), related_frame_filename
