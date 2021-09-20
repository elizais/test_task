import os

from flask import render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

from . import app
from .mlmodel.predictor import Predictor
from .utils import allowed_file


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        if filename != '' and file and allowed_file(file.filename):
            file_path = os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save(file_path)
            predictor = Predictor(app.config['NAME_FINAL_MODEL_DIR'])
            data_pred = predictor(file_path,
                                  app.config['ID_COL'],
                                  app.config['NAME_DATA_TYPE_COL'],
                                  app.config['DEL_COLUMN'],
                                  app.config['DATA_TYPE'][1])
            os.remove(file_path)
            return render_template('result_table.html', title='Renewed', predict=data_pred)
    return render_template('upload_file.html', title='Upload File')
