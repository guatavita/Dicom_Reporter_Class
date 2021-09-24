# Dicom_Reporter_Class

This class is a DICOM listener that will create a report of all the dicom seriesID/patientID in the output directory as well as possibibly convert them to nifti images.

### Example

```
from Dicom_Reporter_Class import *

'''
:param input_dir:
:param output_dir:
:param force_rewrite:
:param save_json:
:param load_json:
:param supp_tags:
:param nb_threads:
:param verbose:
'''

def main():
    supp_tags = {
        'MammoDesc': '0055|1001'
    }

    time_start = time.time()
    dicom_explorer = Dicom_Reporter(input_dir=input_dir,
                                    output_dir=output_dir,
                                    force_rewrite=True,
                                    save_json=True,
                                    load_json=True,
                                    supp_tags=supp_tags,
                                    nb_threads=1, verbose=False)
    dicom_explorer.run_conversion()
```

### Dependencies

```
pip install -r requirements.txt
```
