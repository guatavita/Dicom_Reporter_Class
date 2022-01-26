from Dicom_Reporter_Class import Dicom_Reporter


def main():
    # input_dir = r'Z:\Morfeus\Bastien\MERIT\test_tom'
    # output_dir = r'Z:\Morfeus\Bastien\MERIT\test_tom\nifti'

    # input_dir = r'Z:\Morfeus\Bastien\DICOM\AIP_LACC\test'
    # output_dir = r'Z:\Morfeus\Bastien\DICOM\AIP_LACC\test\nifti'

    input_dir = r'/workspace/Morfeus/Bastien/MERIT/DICOM_study_v1/Consolidated/DATA'
    output_dir = r'/workspace/Morfeus/Bastien/MERIT/nifti_consolidated'
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

    # README.md
    contour_names = []
    contour_association = {}

    # mg_template = {
    #     'image_series_id': True,
    #     'study_desc_name': False,
    #     'merge_study_serie_desc': False,
    #     'force_uint16': True,
    #     'force_int16': False,
    # }

    rt_template = {
        'image_series_id': False,
        'study_desc_name': True,
        'merge_study_serie_desc': True,
        'force_uint16': False,
        'force_int16': True,
        'include_patient_name': True,
        'avoid_duplicate': True,
    }

    dicom_explorer = Dicom_Reporter(input_dir=input_dir,
                                    output_dir=output_dir,
                                    contour_names=contour_names,
                                    contour_association=contour_association,
                                    force_rewrite=True,
                                    save_json=True,
                                    load_json=True,
                                    supp_tags=supp_tags, nb_threads=10,
                                    verbose=True, **rt_template)

    # dicom_explorer.run_conversion()


if __name__ == '__main__':
    main()
