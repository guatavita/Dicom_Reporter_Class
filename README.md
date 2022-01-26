# DICOM REPORTER-CONVERTER

## Table of contents
* [General info](#general-info)
* [Example](#example)
* [Example for radiotherapy](#Example-for-radiotherapy-DICOM-data)
* [Example for mammography](#Example-for-radiotherapy-DICOM-data)
* [Example for cardiac phases](#Example-for-radiotherapy-DICOM-data)
* [Dependencies](#dependencies)

## General info
This class is a DICOM listener that will create a report of all the dicom seriesID/patientID in the output directory as well as possibibly convert them to NIfTI images.

The dicom report will be available both in ".json" and ".txt" ("," delimiter)

The ".run_conversion()" command line supports CT/MR/MG modalities, RTDOSE and RTSTRUCT DICOM formats. Other modalities may be supported but not tested.

This class support unorganized DICOM data as each study will be matched by their "StudyInstanceUID"

Bastien Rigaud, PhD
Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
Campus de Beaulieu, Université de Rennes 1
35042 Rennes, FRANCE
bastien.rigaud@univ-rennes1.fr

## Parameter example

```
:param input_dir: input folder where (unorganized) dicom can be found
:param output_dir: output directory to save dcm_report.json and conversion output following \PatientID\Date\StudyORSeriesDescription
:param contour_names: list of contour names that will be written, ALL if empty
:param contour_association: dictionary of contour names association
:param force_rewrite: for rewrite of NIfTI images (user should remove dcm_report.json)
:param extension: output image extension ['.nii.gz', '.nii', '.mhd', '.nrrd', '.mha']
:param force_uint16: force_uint16 output pixel value representation
:param force_int16: force_int16 output pixel value representation
:param image_series_id: True if you want the series id in the image filename, if you expect multiple series in study output dir
:param study_desc_folder_name: True if you want the output folder to be named after the StudyDescription (False -> SeriesDescription)
:param merge_study_serie_desc: merge study and series descript for image folder name
:param include_patient_name: include patient name with MRN in output folder name
:param avoid_duplicate: True if you want to add _N after folder name in case duplicate output foldername
:param split_by_cardiac_phase: split SeriesInstanceUID using NominalPercentageOfCardiacPhase
:param save_json: save dcm_report.json in output_dir
:param load_json: reload previous dcm_report.json
:param supp_tags: extract DICOM metadata for in-house usage, format dict such as {'SOPClassUID': '0008|0016',}
:param nb_threads: nb_thread to run processes in multithreads
:param verbose: True to have output prints
```

## Example for radiotherapy DICOM data

```
from Dicom_Reporter_Class import *

def main():
    supp_tags = {
        'MammoDesc': '0055|1001'
    }

    contour_names = ['Bag_Bowel']
    contour_association = {
        'CTV45_Pelvic Nodes': 'CTVN',
        'CTV45_Pelvis_Nodes': 'CTVN',
        'CTVn45': 'CTVN',
        'ITV45_Uterus/Cervix/Vagina/Parametria': 'ITV45',
    }
    rt_template = {
        'image_series_id': False,
        'study_desc_name': True,
        'merge_study_serie_desc': True,
        'force_uint16': False,
        'force_int16': True,
        'include_patient_name': True,
        'avoid_duplicate': True,
    }
        
    time_start = time.time()
    dicom_explorer = Dicom_Reporter(input_dir=input_dir,
                                    output_dir=output_dir,
                                    contour_names=contour_names,
                                    contour_association=contour_association,
                                    force_rewrite=True,
                                    save_json=True,
                                    load_json=True,
                                    supp_tags=supp_tags, nb_threads=1,
                                    verbose=True, **rt_template)
    dicom_explorer.run_conversion()
    print("     Elapse time {}".format(time.time() - time_start))
```

## Example for mammogram DICOM data

```
from Dicom_Reporter_Class import *

def main():
    supp_tags = {
        'Manufacturer': '0008|0070',
        'KVP': '0018|0060',
        'ExposureTime': '0018|9328',
        'ExposureTimeInms': '0018|1150',
        'XRayTubeCurrent': '0018|1151',
        'XRayTubeCurrentInmA': '0018|9330',
        'Exposure': '0018|1152',
        'ExposureInmAs': '0018|9332',
        'BodyPartThickness': '0018|11a0',
        'CompressionForce': '0018|11a2',
        'RelativeXRayExposure': '0018|1405',
        'Rows': '0028|0010',
        'Columns': '0028|0011',
        'PixelSpacing': '0028|0030',
        'OrganDose': '0040|0316',
    }
    contour_names = []
    contour_association = {}
    mg_template = {
        'image_series_id': True,
        'study_desc_name': False,
        'merge_study_serie_desc': False,
        'force_uint16': True,
        'force_int16': False,
        'include_patient_name': False,
        'avoid_duplicate': False,
        'split_by_cardiac_phase': False,
    }

    time_start = time.time()   
    dicom_explorer = Dicom_Reporter(input_dir=input_dir,
                                    output_dir=output_dir,
                                    contour_names=contour_names,
                                    contour_association=contour_association,
                                    force_rewrite=True,
                                    save_json=True,
                                    load_json=True,
                                    supp_tags=supp_tags, nb_threads=1,
                                    verbose=True, **mg_template)
    dicom_explorer.run_conversion()
    print("     Elapse time {}".format(time.time() - time_start))
```

## Example for cardiac images (multiple phases) DICOM data
```
    tv_template = {
        'image_series_id': False,
        'study_desc_name': True,
        'merge_study_serie_desc': False,
        'force_uint16': False,
        'force_int16': True,
        'include_patient_name': True,
        'avoid_duplicate': False,
        'split_by_cardiac_phase': True,
    }

    dicom_explorer = Dicom_Reporter(input_dir=input_dir,
                                    output_dir=output_dir,
                                    contour_names=contour_names,
                                    contour_association=contour_association,
                                    force_rewrite=True,
                                    save_json=True,
                                    load_json=True,
                                    supp_tags=supp_tags, nb_threads=10,
                                    verbose=True, **tv_template)
    dicom_explorer.run_conversion()
```
## Dependencies
```
pip install -r requirements.txt
```
```
tqdm
numpy
SimpleITK
pydicom>=2.2.1
opencv-python
matplotlib
ipywidgets
pathvalidate
python-gdcm
PlotScrollNumpyArrays
```
