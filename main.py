from Dicom_Reporter_Class import Dicom_Reporter


def main():
    input_dir = r'Z:\Morfeus\Bastien\MERIT\DICOM_explore_example'
    output_dir = r'Z:\Morfeus\Bastien\MERIT\DICOM_explore_example\nifti'
    supp_tags = {}
    contour_names = []
    contour_association = {}
    rt_template = {'image_series_id': False,
                   'study_desc_name': True,
                   'merge_study_serie_desc': True}
    mg_template = {'image_series_id': True,
                   'study_desc_name': False,
                   'merge_study_serie_desc': False}

    dicom_explorer = Dicom_Reporter(input_dir=input_dir,
                                    output_dir=output_dir,
                                    contour_names=contour_names,
                                    contour_association=contour_association,
                                    force_rewrite=True,
                                    save_json=True,
                                    load_json=True,
                                    supp_tags=supp_tags,
                                    nb_threads=24, verbose=True, **mg_template)
    dicom_explorer.run_conversion()


if __name__ == '__main__':
    main()
