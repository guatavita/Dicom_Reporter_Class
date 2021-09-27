# Dicom_Reporter_Class

This class is a DICOM listener that will create a report of all the dicom seriesID/patientID in the output directory as well as possibibly convert them to nifti images.

The ".run_conversion()" command line supports CT/MR/MG DICOM, RTDOSE DICOM and RTSTRUCT DICOM.

This class support unorganized DICOM data as each study will be matched by their "StudyInstanceUID"

### Example

```
from Dicom_Reporter_Class import *

'''
:param input_dir: input folder where (unorganized) dicom can be found
:param output_dir: output directory to save dcm_report.json and conversion output following \PatientID\StudyDate\SeriesDescription
:param contour_names: list of contour names that will be written, ALL if empty
:param contour_association: dictionary of contour names association
:param force_rewrite: for rewrite of NIfTI images (user should remove dcm_report.json)
:param save_json: save dcm_report.json in output_dir
:param load_json: reload previous dcm_report.json
:param supp_tags: extract DICOM metadata for in-house usage
:param nb_threads: nb_thread to run processes in multithreads
:param verbose: True to have output prints
'''

def main():
    supp_tags = {
        'MammoDesc': '0055|1001'
    }

    contour_names = ['Bag_Bowel']
    contour_association = {
        'CTV45_Pelvis Nodes': 'CTVNs',
        'CTV45_Pelvic Nodes': 'CTVN',
        'CTV45_Pelvis_Nodes': 'CTVN',
        'CTVn45': 'CTVN',
        'ITV45_Uterus/Cervix/Vagina/Parametria': 'ITV45',
    }
    time_start = time.time()
    dicom_explorer = Dicom_Reporter(input_dir=input_dir,
                                    output_dir=output_dir,
                                    contour_names=contour_names,
                                    contour_association=contour_association,
                                    force_rewrite=True,
                                    save_json=True,
                                    load_json=True,
                                    supp_tags=supp_tags,
                                    nb_threads=1, verbose=False)
    dicom_explorer.run_conversion()
    print("     Elapse time {}".format(time.time() - time_start))
```

### Dependencies

```
pip install -r requirements.txt
```
