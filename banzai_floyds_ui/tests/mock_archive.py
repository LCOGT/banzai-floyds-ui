from flask import Flask, request, jsonify, send_file, url_for
from glob import glob
import io
import os
from astropy.io import fits

file_list = glob('**/*.fits*', recursive=True)

app = Flask(__name__)

frames = {}
for i, filename in enumerate(file_list):
    hdu = fits.open(filename)
    if 'SCI' in hdu:
        hdu_index = 'SCI'
    else:
        hdu_index = 0
    try:
        frames[i] = {'id': i,
                     'filename': os.path.basename(filename),
                     'INSTRUME': hdu[hdu_index].header['INSTRUME'],
                     'SITEID': hdu[hdu_index].header['INSTRUME'],
                     'filepath': os.path.dirname(filename),}
    except KeyError:
        continue


@app.route('/', methods=['GET'])
@app.route('/<int:frame_id>/', methods=['GET'])
def get_frame_by_params(frame_id=None):
    if frame_id is not None:
        if frame_id in frames:
            frame = frames[frame_id]
            frame['url'] = url_for('download_frame', frame_id=frame_id, _external=True)
            return jsonify(frame), 201
        else:
            return jsonify({"error": "Frame not found"}), 404
    instrument = request.args.get('instrument_id')
    basename = request.args.get('basename')
    basename_exact = request.args.get('basename_exact')
    response_frames = []
    for frame_id, frame in frames.items():
        frame_criteria = frame['INSTRUME'] == instrument
        if basename is not None:
            frame_criteria = frame_criteria and basename in frame['filename']
        if basename_exact is not None:
            frame_criteria = basename_exact == frame['filename']
        if frame_criteria:
            frame['url'] = url_for('download_frame', frame_id=frame_id, _external=True)
            response_frames.append(frame)

    return jsonify({'results': response_frames}), 201


@app.route('/download/<int:frame_id>', methods=['GET'])
def download_frame(frame_id):

    if frame_id in frames:
        with open(os.path.join(frames[frame_id]['filepath'], frames[frame_id]['filename']), 'rb') as f:
            return send_file(
                io.BytesIO(f.read()),
                download_name=frames[frame_id]['filename'],
                mimetype='application/fits'
            ), 200
    else:
        return jsonify({"error": "Frame not found"}), 404


@app.route('/<int:frame_id>/headers', methods=['GET'])
def get_header(frame_id):
    if frame_id in frames:
        hdu = fits.open(os.path.join(frames[frame_id]['filepath'], frames[frame_id]['filename']))
        if 'SCI' in hdu:
            hdu_index = 'SCI'
        else:
            hdu_index = 0
        return jsonify({'data': dict(hdu[hdu_index].header)}), 200
    else:
        return jsonify({"error": "Frame not found"}), 404


if __name__ == '__main__':
    app.run(debug=True)
